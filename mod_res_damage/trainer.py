import logging
import operator
import os
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from mod_res_damage.utils.utils import RunningAverageMeter, compute_loss

      
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module | None,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        ckpt_dir: pathlib.Path | str,
        device: torch.device,
        distributed: bool,
        cudnn_backend: bool,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
    ):
        self.rank = int(os.environ["RANK"])
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.n_epochs = n_epochs
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.distributed = distributed
        self.cudnn_backend = cudnn_backend
        self.use_wandb = use_wandb
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.best_metric_key = best_metric_key
        
        self.logger = logging.getLogger()
        self.training_stats = {
            name: RunningAverageMeter(length=len(self.train_loader))
            for name in ["loss", "data_time", "batch_time", "eval_time"]
        }
        self.best_metric = -1
        self.best_metric_comp = operator.gt
        self.num_predictands = len(self.train_loader.dataset.predictands)

        assert precision in [
            "fp32",
            "fp16",
            "bfp16",
        ], f"Invalid precision {precision}, use 'fp32', 'fp16' or 'bfp16'."
        
        torch.backends.cudnn.enabled = self.cudnn_backend
        
        self.enable_mixed_precision = precision != "fp32"
        self.precision = torch.float16 if (precision == "fp16") else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_mixed_precision)
        self.start_epoch = 0

        if self.use_wandb:
            import wandb
            self.wandb = wandb
            
    def train(self) -> None:
        for epoch in range(self.start_epoch, self.n_epochs):
            if epoch % self.eval_interval == 0:
                metrics, used_time = self.evaluator(self.model, f"epoch {epoch}")
                self.training_stats["eval_time"].update(used_time)
                self.save_best_checkpoint(metrics, epoch)

            self.logger.info("============ Starting epoch %i ... ============" % epoch)
            self.t = time.time()
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if epoch % self.ckpt_interval == 0 and epoch != self.start_epoch:
                self.save_model(epoch)

        metrics, used_time = self.evaluator(self.model, f"epoch {epoch}")
        self.training_stats["eval_time"].update(used_time)
        self.save_best_checkpoint(metrics, self.n_epochs)
        self.save_model(self.n_epochs, is_final=True)

    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            inputs, target, no_data = data["inputs"], data["target"], data["no_data"]
            inputs = {modality: value.to(self.device) for modality, value in inputs.items()}
            target = target.to(self.device)
            no_data = no_data.to(self.device)

            self.training_stats["data_time"].update(time.time() - end_time)

            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                logits = self.model(inputs)
                loss = compute_loss(self.criterion, logits, target, no_data)

            self.optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.training_stats['loss'].update(loss.item())
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            self.lr_scheduler.step()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=epoch * len(self.train_loader) + batch_idx,
                )

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()

    def get_checkpoint(self, epoch: int) -> dict[str, dict | int]:
        state_dict = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        checkpoint = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
        }
        return checkpoint

    def save_model(
        self,
        epoch: int,
        is_final: bool = False,
        is_best: bool = False,
        checkpoint: dict[str, dict | int] | None = None,
    ):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
            return
        checkpoint = self.get_checkpoint(epoch) if checkpoint is None else checkpoint
        name = f"epoch-{epoch}_{self.best_metric_key}_best" if is_best else f"epoch-{epoch}_final" if is_final else f"epoch-{epoch}"
        checkpoint_path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}"
        )
        if self.distributed:
            torch.distributed.barrier()
        return

    def load_model(self, resume_path: str | pathlib.Path) -> None:
        model_dict = torch.load(resume_path, map_location=self.device, weights_only=False)
        if "model" in model_dict:
            if self.distributed:
                self.model.module.load_state_dict(model_dict["model"])
            else:
                self.model.load_state_dict(model_dict["model"])
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
            self.scaler.load_state_dict(model_dict["scaler"])
            self.start_epoch = model_dict["epoch"] + 1
        else:
            if self.distributed:
                self.model.module.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(model_dict)
            self.start_epoch = 0

        self.logger.info(
            f"Loaded model from {resume_path}. Resume training from epoch {self.start_epoch}"
        )

    def save_best_checkpoint(
        self, eval_metrics: dict[float, list[float]], epoch: int
    ) -> None:
        curr_metric = eval_metrics[self.best_metric_key]
        if isinstance(curr_metric, list):
            curr_metric = curr_metric[1] if self.num_predictands == 1 else np.mean(curr_metric)
        if self.best_metric_comp(curr_metric, self.best_metric):
            self.best_metric = curr_metric
            best_ckpt = self.get_checkpoint(epoch)
            self.save_model(
                epoch, is_best=True, checkpoint=best_ckpt
            )

    def log(self, batch_idx: int, epoch) -> None:
        """Log the information.

        Args:
            batch_idx (int): number of the batch.
            epoch (_type_): number of the epoch.
        """
        basic_info = (
            "Epoch [{epoch}-{batch_idx}/{len_loader}]\t"
            "Time [{batch_time.avg:.3f}|{data_time.avg:.3f}]\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "lr {lr:.3e}".format(
                epoch=epoch,
                len_loader=len(self.train_loader),
                batch_idx=batch_idx,
                batch_time=self.training_stats["batch_time"],
                data_time=self.training_stats["data_time"],
                loss=self.training_stats["loss"],
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )

        log_info = basic_info
        self.logger.info(log_info)

    def reset_stats(self) -> None:
        """Reset the training stats and metrics."""
        for v in self.training_stats.values():
            v.reset()
