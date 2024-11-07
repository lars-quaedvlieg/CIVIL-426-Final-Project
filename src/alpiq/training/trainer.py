from typing import Callable, Optional

import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            criterion: Callable,
            train_loader: DataLoader,
            eval_loader: DataLoader,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Trainer for the CausalModel.

        Args:
            model (nn.Module): The model to train.
            optimizer (Optimizer): Optimizer for updating model weights.
            criterion (Callable): Loss function.
            train_loader (DataLoader): DataLoader for the training set.
            eval_loader (DataLoader): DataLoader for the validation set.
            device (torch.device): Device to run training on (CPU or GPU).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device

    def train(self, num_epochs: int = 10, checkpoint_path: Optional[str] = None):
        """
        Runs the full training loop for the specified number of epochs.

        Args:
            num_epochs (int): Number of epochs for training.
            checkpoint_path (Optional[str]): Path to save model checkpoints.
        """
        best_eval_loss = float("inf")

        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch(epoch)
            eval_loss = self._evaluate(self.eval_loader)

            # Log to WandB if active
            if wandb.run is not None:
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "eval_loss": eval_loss})

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

            # Save the model checkpoint if evaluation loss improves
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                if checkpoint_path:
                    self._save_checkpoint(epoch, best_eval_loss, checkpoint_path)
                    if wandb.run is not None:
                        wandb.log({"best_eval_loss": best_eval_loss})

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (op_x, input_signals, targets) in enumerate(self.train_loader):
            op_x, input_signals, targets = op_x.to(self.device), input_signals.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(op_x, input_signals)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Log batch loss to WandB every 10 batches
            if wandb.run is not None and batch_idx % 10 == 0:
                wandb.log({"batch_loss": loss.item()})

            if batch_idx % 10 == 0:
                print(
                    f"Train Epoch [{epoch + 1}], Batch [{batch_idx}/{len(self.train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def _evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluates the model on a given dataset.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to evaluate (validation or test).

        Returns:
            float: Average loss on the dataset.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for op_x, input_signals, targets in data_loader:
                op_x, input_signals, targets = op_x.to(self.device), input_signals.to(self.device), targets.to(
                    self.device)

                # Forward pass
                outputs = self.model(op_x, input_signals)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        return avg_loss

    def _save_checkpoint(self, epoch: int, loss: float, checkpoint_path: str):
        """
        Saves the model checkpoint.

        Args:
            epoch (int): Epoch number.
            loss (float): Loss value to store with the checkpoint.
            checkpoint_path (str): Path to save the checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1} with loss: {loss:.4f}")

    def test(self, test_loader: DataLoader):
        """
        Tests the model on the test set and logs the average test loss.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
        """
        test_loss = self._evaluate(test_loader)

        # Log test loss to WandB if active
        if wandb.run is not None:
            wandb.log({"test_loss": test_loss})

        print(f"Test Loss: {test_loss:.4f}")
        return test_loss
