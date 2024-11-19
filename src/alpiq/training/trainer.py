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
            eval_loader: Optional[DataLoader] = None,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model_save_freq_epochs = 1,
            batch_loss_log_freq = 10,
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
        self.model_save_freq_epochs = model_save_freq_epochs
        self.batch_loss_log_freq = batch_loss_log_freq

    def train(self, num_epochs: int = 10, checkpoint_path: Optional[str] = None):
        """
        Runs the full training loop for the specified number of epochs.

        Args:
            num_epochs (int): Number of epochs for training.
            checkpoint_path (Optional[str]): Path to save model checkpoints.
        """
        best_loss = float("inf")
        do_eval = self.eval_loader is not None

        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch(epoch)
            if do_eval:
                eval_loss = self._evaluate(self.eval_loader)

            # Log to WandB if active
            if wandb.run is not None:
                metrics_dict = {"epoch": epoch + 1, "train_loss": train_loss}
                if do_eval:
                    metrics_dict["eval_loss"] = eval_loss
                wandb.log(metrics_dict)

            print_str = f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}"
            if do_eval:
                print_str += f", Eval Loss: {eval_loss:.4f}"
            print(print_str)

            # Save the model checkpoint if evaluation loss improves
            cur_loss = train_loss if not do_eval else eval_loss
            if cur_loss < best_loss or epoch % self.model_save_freq_epochs == 0:
                best_loss = cur_loss
                if checkpoint_path:
                    self._save_checkpoint(epoch, best_loss, checkpoint_path)
                    if do_eval and wandb.run is not None:
                        wandb.log({"best_eval_loss": best_loss})

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

        for batch_idx, batch in enumerate(self.train_loader):
            op_x = batch["operating_mode"].to(self.device)
            input_signals = batch["input_sequence"].to(self.device)
            cur_control_values = batch["current_values"].to(self.device)
            targets = batch["next_values"].to(self.device)

            # Forward pass
            outputs = self.model(op_x, input_signals, cur_control_values)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Log batch loss to WandB every 10 batches
            if wandb.run is not None and batch_idx % 10 == 0:
                wandb.log({"batch_loss": loss.item()})

            if batch_idx % self.batch_loss_log_freq == 0:
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
            for batch in data_loader:
                op_x = batch["operating_mode"].to(self.device)
                input_signals = batch["input_sequence"].to(self.device)
                cur_control_values = batch["current_values"].to(self.device)
                targets = batch["next_values"].to(self.device)

                # Forward pass
                outputs = self.model(op_x, input_signals, cur_control_values)
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
