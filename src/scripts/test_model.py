import os
from pathlib import Path
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

from alpiq.data.sequence_dataset import SequenceDataset
from alpiq.model.causal_model import CausalModel
from alpiq.evaluation.anomaly_detection import compute_score, windowed_threshold_batch


@hydra.main(config_path="configs", config_name="test_model_VG5_anom_01a")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize WandB if enabled
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load datasets and create data loaders
    test_dataset = SequenceDataset(
        data_path=Path(cfg.data.test_file),
        context_length=cfg.model.context_length,
        input_feature_col_names=cfg.data.input_feature_cols,
        current_value_col_names=cfg.data.current_value_cols,
        next_value_col_names=cfg.data.next_value_cols,
        padding_value=cfg.data.padding_value,
        load_GT=cfg.data.load_GT
    )

    # Data loaders
    test_loader = DataLoader(test_dataset, batch_size=cfg.testing.batch_size, shuffle=False)

    # Model, optimizer, and loss function
    model = CausalModel(
        input_dim=len(cfg.data.input_feature_cols),
        num_operating_modes=cfg.data.num_operating_modes,
        sequence_length=cfg.model.context_length,
        embedding_dim=cfg.model.embedding_dim,
        state_dim=cfg.model.state_dim,
        num_s5_layers=cfg.model.num_s5_layers,
        num_control_variates=len(cfg.data.next_value_cols),
        s5_dropout=cfg.model.s5_dropout,
        fc_hidden_dims=cfg.model.fc_hidden_dims,
        fc_dropout=cfg.model.fc_dropout
    ).to(device)

    # Load the checkpoint
    checkpoint = torch.load(cfg.testing.model_path, map_location=device)

    # Restore the model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    outputs_buffer = []
    scores_buffer = []
    targets_buffer = []
    states_buffer = []

    outputs_list = []
    scores_list = []
    targets_list = []
    anomalies_list = []
    GT_list = []

    buffer_size = cfg.testing.buffer_size

    with torch.no_grad():
        for batch in tqdm(test_loader):
            if cfg.data.load_GT:
                GT_X = batch["ground_truth"]
            op_x = batch["operating_mode"].to(device)
            input_signals = batch["input_sequence"].to(device)
            cur_control_values = batch["current_values"].to(device)
            targets = batch["next_values"].to(device)

            # Forward pass
            outputs = model(op_x, input_signals, cur_control_values)

            # Compute the score
            scores = compute_score(outputs, targets)

            # Save Outputs Scores and Targets
            outputs_buffer.extend(outputs.cpu().numpy())
            scores_buffer.extend(scores.cpu().numpy())
            targets_buffer.extend(targets.cpu().numpy())
            outputs_list.extend(outputs.cpu().numpy())
            scores_list.extend(scores.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

            if len(outputs_buffer) > buffer_size:
                outputs_buffer = outputs_buffer[-buffer_size:]
                scores_buffer = scores_buffer[-buffer_size:]
                targets_buffer = targets_buffer[-buffer_size:]

            # Compute the anomaly state
            if len(scores_buffer) > cfg.testing.window_size:
                states_buffer = windowed_threshold_batch(scores_buffer, cfg.testing.threshold, cfg.testing.window_size, states_buffer, cfg.testing.nb_consecutive_anomalies, cfg.testing.batch_size)

                # Check if any of the new states are anomalies
                new_states = [state[-cfg.testing.batch_size:] for state in states_buffer]
                anomalies_batch = [any([state[t] == 2 for state in new_states]) for t in range(cfg.testing.batch_size)]
                anomalies_list.extend(anomalies_batch)
                if cfg.data.load_GT:
                    GT_list.extend(GT_X.cpu().numpy())

            if len(GT_list) > 10000:
                break

    # Save the results to a file
    if not os.path.exists(cfg.testing.results_dir):
        os.makedirs(cfg.testing.results_dir)

    # Save the outputs, scores, targets, anomalies, and GT to a file
    results_file = os.path.join(cfg.testing.results_dir, cfg.testing.results_file)
    with open(results_file, 'w') as f:
        f.write("Outputs, Scores, Targets, Anomalies, GT\n")
        for i in range(len(outputs_list)):
            f.write(f"{outputs_list[i]}, {scores_list[i]}, {targets_list[i]}, {anomalies_list[i]}, {GT_list[i]}\n")

    print(f"Anomalies shape: {len(anomalies_list)}")
    print(f"GT shape: {len(GT_list)}")

    if cfg.data.load_GT:
        # Create a figure with n+2 subplots, n being the number of cols in cfg.data.next_value_cols,
        # so in each plot you will the output and the target and the score for each of the cols,
        # in the n+1 plot you will see the total score
        # and in the last plot you will see the anomalies and the GT
        n = len(cfg.data.next_value_cols)
        fig, axs = plt.subplots(n+2, 1, figsize=(15, (n+1)*5))
        for i in range(n):
            axs[i].plot([outputs_list[j][i] for j in range(len(outputs_list))], label='Outputs')
            axs[i].plot([targets_list[j][i] for j in range(len(targets_list))], label='Targets')
            axs[i].plot([scores_list[j][i] for j in range(len(scores_list))], label='Scores')
            axs[i].legend()
            axs[i].set_title(cfg.data.next_value_cols[i])

        axs[n].plot([scores_list[j][-1] for j in range(len(scores_list))], label='Total score')

        axs[n+1].plot(anomalies_list, label='Anomalies')
        axs[n+1].plot(GT_list, label='GT')
        axs[n+1].legend()
        axs[n+1].set_title('Anomalies and GT')

        # save the figure
        fig_file = os.path.join(cfg.testing.results_dir, cfg.testing.fig_file)
        plt.savefig(fig_file)
        plt.show()

        # plot the ROC curve
        fpr, tpr, _ = roc_curve(GT_list, anomalies_list)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.show()

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