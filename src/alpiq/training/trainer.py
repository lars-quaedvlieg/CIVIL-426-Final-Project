import os
from typing import Callable, Optional
from tqdm import tqdm
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
            eval_freq_steps: int = 100,
            save_freq_steps: int = 100,
            batch_loss_log_freq: int = 10,
    ):
        """
        Trainer for the CausalModel with step-based training.

        Args:
            model (nn.Module): The model to train.
            optimizer (Optimizer): Optimizer for updating model weights.
            criterion (Callable): Loss function.
            train_loader (DataLoader): DataLoader for the training set.
            eval_loader (DataLoader): DataLoader for the validation set.
            device (torch.device): Device to run training on (CPU or GPU).
            eval_freq_steps (int): Frequency of steps to evaluate the model.
            save_freq_steps (int): Frequency of steps to save the model checkpoint.
            batch_loss_log_freq (int): Frequency of steps to log batch loss to WandB.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.eval_freq_steps = eval_freq_steps
        self.save_freq_steps = save_freq_steps
        self.batch_loss_log_freq = batch_loss_log_freq
        assert self.eval_freq_steps % self.batch_loss_log_freq == 0

        self.global_step = 0

    def train(self, num_steps: int, checkpoint_folder_path: Optional[str] = None):
        """
        Runs the training loop for the specified number of steps.

        Args:
            num_steps (int): Number of gradient steps for training.
            checkpoint_folder_path (Optional[str]): Path to save model checkpoints folder.
        """

        best_loss = float("inf")
        train_iterator = iter(self.train_loader)
        pbar = tqdm(total=num_steps, desc="Training Progress")

        self.model.train()
        while self.global_step < num_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            op_x = batch["operating_mode"].to(self.device)
            input_signals = batch["input_sequence"].to(self.device)
            cur_control_values = batch["current_values"].to(self.device) + torch.randn_like(batch["current_values"]).to(
                self.device) * 1
            targets = batch["next_values"].to(self.device)

            # Forward pass
            outputs = self.model(op_x, input_signals, cur_control_values)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log batch loss to WandB every `batch_loss_log_freq` steps
            log_dict = {}
            if wandb.run is not None and self.global_step % self.batch_loss_log_freq == 0:
                log_dict = log_dict | {"batch_loss": loss.item(), "step": self.global_step}

            # Evaluate model every `eval_freq_steps` steps
            if self.eval_loader is not None and self.global_step % self.eval_freq_steps == 0:
                eval_loss = self._evaluate()
                self.model.train()
                if wandb.run is not None:
                    log_dict = log_dict | {"eval_loss": eval_loss, "step": self.global_step}
                print(f"Step [{self.global_step}], Eval Loss: {eval_loss:.4f}")

            if wandb.run is not None and self.global_step % self.batch_loss_log_freq == 0:
                wandb.log(log_dict)

            # Save checkpoint every `save_freq_steps` steps or if it’s the best eval loss
            cur_loss = loss.item() if self.eval_loader is None else eval_loss
            if checkpoint_folder_path is not None and self.global_step % self.save_freq_steps == 0:
                ckpt_path = os.path.join(checkpoint_folder_path, f"causal-ckpt-{self.global_step}.pth")
                self._save_checkpoint(self.global_step, cur_loss, ckpt_path)
            if checkpoint_folder_path is not None and self.eval_loader and cur_loss < best_loss:
                best_loss = cur_loss
                ckpt_path = os.path.join(checkpoint_folder_path, f"causal-model-best.pth")
                self._save_checkpoint(self.global_step, best_loss, ckpt_path)

            # Update progress bar
            pbar.set_description(f"Step [{self.global_step}], Train Loss: {loss.item():.4f}")
            pbar.update(1)
            self.global_step += 1

        pbar.close()

    def _evaluate(self) -> float:
        """
        Evaluates the model on the evaluation dataset.

        Returns:
            float: Average loss on the evaluation set.
        """
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                op_x = batch["operating_mode"].to(self.device)
                input_signals = batch["input_sequence"].to(self.device)
                cur_control_values = batch["current_values"].to(self.device)
                targets = batch["next_values"].to(self.device)

                # Forward pass
                outputs = self.model(op_x, input_signals, cur_control_values)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

        avg_loss = running_loss / len(self.eval_loader)
        return avg_loss

    def _save_checkpoint(self, step: int, loss: float, checkpoint_path: str):
        """
        Saves the model checkpoint.

        Args:
            step (int): Current training step.
            loss (float): Loss value to store with the checkpoint.
            checkpoint_path (str): Path to save the checkpoint.
        """
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at step {step} with loss: {loss:.4f}")

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
