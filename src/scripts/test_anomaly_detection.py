import os
from pathlib import Path
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

from alpiq.data.sequence_dataset import SequenceDataset
from alpiq.model.causal_model import CausalModel
from alpiq.evaluation.anomaly_detection import compute_score, windowed_threshold_batch


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, outputs_list, targets_list):
        self.outputs_list = outputs_list
        self.targets_list = targets_list

    def __len__(self):
        return len(self.outputs_list)

    def __getitem__(self, idx):
        return {"outputs": self.outputs_list[idx], "targets": self.targets_list[idx]}


@hydra.main(config_path="configs", config_name="test_model_VG5_anom_01a")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize WandB if enabled
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load model outputs from the csv file
    results_file = os.path.join(cfg.testing.results_dir, cfg.testing.results_file)

    # Load the data from the pickle file
    with open(results_file, 'rb') as f:
        outputs_list, targets_list, GT_list = pickle.load(f)

    # Example usage of the loaded data
    print(f"Loaded {len(outputs_list)} outputs and {len(targets_list)} targets.")
    if GT_list is not None:
        print(f"Loaded {len(GT_list)} ground truth values.")

    outputs_list = np.array(outputs_list)
    targets_list = np.array(targets_list)
    if cfg.data.load_GT:
        GT_list = np.array(GT_list)

    # Create a simple data loader to iterate over outputs_list, targets_list and GT_list
    prerun_dataset = SimpleDataset(outputs_list, targets_list)

    # Load datasets and create data loaders
#    test_dataset = SequenceDataset(
 #       data_path=Path(cfg.data.test_file),
  #      context_length=cfg.model.context_length,
   #     input_feature_col_names=cfg.data.input_feature_cols,
    #    current_value_col_names=cfg.data.current_value_cols,
     #   next_value_col_names=cfg.data.next_value_cols,
      #  padding_value=cfg.data.padding_value,
       # load_GT=cfg.data.load_GT
    #)

    prerun_loader = DataLoader(prerun_dataset, batch_size=cfg.testing.batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=cfg.testing.batch_size, shuffle=False)


    outputs_buffer = []
    scores_buffer = []
    targets_buffer = []
    states_buffer = []

    scores_list = []
    anomalies_list = []

    ope_mod_list = []
    #for batch in tqdm(test_loader):
     #   ope_mod_list.extend(batch["operating_mode"].numpy())


    buffer_size = cfg.testing.buffer_size

    for batch1 in tqdm(prerun_loader):
        targets = batch1["targets"]
        outputs = batch1["outputs"]

        batch_size = outputs.size(0)

        # Compute the score
        scores = compute_score(outputs, targets)

        # Save Outputs Scores and Targets
        outputs_buffer.extend(outputs.cpu().numpy())
        scores_buffer.extend(scores.cpu().numpy())
        targets_buffer.extend(targets.cpu().numpy())
        scores_list.extend(scores.cpu().numpy())

        if len(outputs_buffer) > buffer_size:
            outputs_buffer = outputs_buffer[-buffer_size:]
            scores_buffer = scores_buffer[-buffer_size:]
            targets_buffer = targets_buffer[-buffer_size:]

        # Compute the anomaly state
        if len(scores_buffer) > cfg.testing.window_size:
            states_buffer = windowed_threshold_batch(scores_buffer, cfg.testing.threshold, cfg.testing.window_size, states_buffer, cfg.testing.nb_consecutive_anomalies, batch_size)

            # Check if any of the new states are anomalies
            new_states = [state[-batch_size:] for state in states_buffer]
            anomalies_batch = [any([state[t] == 2 for state in new_states]) for t in range(batch_size)]
            anomalies_list.extend(anomalies_batch)

    print(f"Anomalies shape: {len(anomalies_list)}")
    print(f"GT shape: {len(GT_list)}")

    if cfg.data.load_GT:
        # Create a figure with n+2 subplots, n being the number of cols in cfg.data.next_value_cols,
        # so in each plot you will the output and the target and the score for each of the cols,
        # in the n+1 plot you will see the total score
        # and in the last plot you will see the anomalies and the GT
        t_plot = 40000
        t_max = min(t_plot, len(outputs_list))
        n = len(cfg.data.next_value_cols)
        fig, axs = plt.subplots(n+2, 1, figsize=(15, (n+1)*5))
        for i in range(n):
            axs[i].plot([outputs_list[j][i] for j in range(t_max)], label='Outputs')
            axs[i].plot([targets_list[j][i] for j in range(t_max)], label='Targets')
            axs[i].plot([scores_list[j][i] for j in range(t_max)], label='Scores')
            axs[i].legend()
            axs[i].set_title(cfg.data.next_value_cols[i])

        axs[n].plot([scores_list[j][-1] for j in range(t_max)], label='Total score')
        axs[n].set_title('Total score')

        axs[n+1].plot(anomalies_list[:t_max], label='Anomalies')
        axs[n+1].plot(GT_list[:t_max], label='GT')
        #axs[n+1].plot(ope_mod_list[:t_max], label='Operating Mode')
        axs[n+1].legend()
        axs[n+1].set_title('Anomalies and GT')

        # save the figure
        fig_file = os.path.join(cfg.testing.results_dir, cfg.testing.fig_file)
        plt.savefig(fig_file)
        plt.show()

        # Find the indices where the ground truth is 1
        GT_indices = np.where(GT_list == 1)[0]
        # Find continuous sequences of indices
        continuous_sequences = []
        start = GT_indices[0]
        for i in range(1, len(GT_indices)):
            if GT_indices[i] != GT_indices[i-1] + 1:
                continuous_sequences.append((max(0, start-500), min(GT_indices[i-1]+500, len(GT_list)-1)))
                start = GT_indices[i]
        continuous_sequences.append((start, GT_indices[-1]))

        # Show the same figure but only for the indices
        # where the ground truth is 1 continuously
        seq2show = continuous_sequences[3]

        n = len(cfg.data.next_value_cols)
        fig, axs = plt.subplots(n+2, 1, figsize=(15, (n+1)*5))
        for i in range(n):
            axs[i].plot([outputs_list[j][i] for j in range(seq2show[0], seq2show[1])], label='Outputs')
            axs[i].plot([targets_list[j][i] for j in range(seq2show[0], seq2show[1])], label='Targets')
            axs[i].plot([scores_list[j][i] for j in range(seq2show[0], seq2show[1])], label='Scores')
            axs[i].legend()
            axs[i].set_title(cfg.data.next_value_cols[i])

        axs[n].plot([scores_list[j][-1] for j in range(seq2show[0], seq2show[1])], label='Total score')
        axs[n].set_title('Total score')

        axs[n+1].plot(anomalies_list[seq2show[0]:seq2show[1]], label='Anomalies')
        axs[n+1].plot(GT_list[seq2show[0]:seq2show[1]], label='GT')
        #axs[n+1].plot(ope_mod_list[seq2show[0]:seq2show[1]], label='Operating Mode')
        axs[n+1].legend()
        axs[n+1].set_title('Anomalies and GT')
        plt.show()


        # plot the ROC curve
        #fpr, tpr, _ = roc_curve(GT_list, anomalies_list)
        #plt.figure()
        #plt.plot(fpr, tpr)
        #plt.show()

        print(f"ROC AUC: {roc_auc_score(GT_list, anomalies_list)}")
        print(f"Precision: {precision_score(GT_list, anomalies_list)}")
        print(f"Recall: {recall_score(GT_list, anomalies_list)}")
        print(f"F1: {f1_score(GT_list, anomalies_list)}")
        print(f"Accuracy: {accuracy_score(GT_list, anomalies_list)}")



    # End WandB run
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()