import os
from pathlib import Path
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from alpiq.data.sequence_dataset import SequenceDataset
from alpiq.model.causal_model import CausalModel
from alpiq.evaluation.anomaly_detection import compute_score, windowed_threshold_batch


@hydra.main(config_path="configs", config_name="test_model_VG6")
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
        padding_value=cfg.data.padding_value
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
    #checkpoint = torch.load(cfg.testing.model_path, map_location=device)

    # Restore the model state
    #model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    outputs_buffer = []
    scores_buffer = []
    targets_buffer = []
    state_buffer = [0]
    buffer_size = 1000

    with torch.no_grad():
        for batch in test_loader:
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

            if len(outputs_buffer) > buffer_size:
                outputs_buffer = outputs_buffer[-buffer_size:]
                scores_buffer = scores_buffer[-buffer_size:]
                targets_buffer = targets_buffer[-buffer_size:]

            # Compute the anomaly state
            if len(scores_buffer) > cfg.testing.window_size:
                state_buffer = windowed_threshold_batch(scores_buffer, cfg.testing.threshold, cfg.testing.window_size, state_buffer, cfg.testing.nb_consecutive_anomalies, cfg.testing.batch_size)
                print(state_buffer)

    # End WandB run
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()