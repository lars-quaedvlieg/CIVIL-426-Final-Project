import os
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from alpiq.data.sequence_dataset import SequenceDataset
from alpiq.model.causal_model import CausalModel
from alpiq.training.trainer import Trainer


@hydra.main(config_path="configs", config_name="train_model_VG6")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize WandB if enabled
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load datasets and create data loaders
    train_dataset = SequenceDataset(
        data_path=Path(cfg.data.train_file),
        context_length=cfg.model.context_length,
        input_feature_col_names=cfg.data.input_feature_cols,
        current_value_col_names=cfg.data.current_value_cols,
        next_value_col_names=cfg.data.next_value_cols,
        padding_value=cfg.data.padding_value
    )
    do_eval = cfg.data.eval_file is not None
    if do_eval:
        eval_dataset = SequenceDataset(
            data_path=Path(cfg.data.eval_file),
            context_length=cfg.model.context_length,
            input_feature_col_names=cfg.data.input_feature_cols,
            current_value_col_names=cfg.data.current_value_cols,
            next_value_col_names=cfg.data.next_value_cols,
            padding_value=cfg.data.padding_value
        )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    eval_loader = None
    if do_eval:
        eval_loader = DataLoader(eval_dataset, batch_size=cfg.training.batch_size, shuffle=False)

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
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Train the model
    os.makedirs(cfg.training.checkpoint_folder, exist_ok=True)
    trainer.train(num_steps=cfg.training.num_gradient_steps, checkpoint_folder_path=cfg.training.checkpoint_folder)

    # End WandB run
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
