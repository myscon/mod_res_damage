import logging
import os
import time
import math
import numpy as np
import rasterio
import torch
import torch.nn as nn

from pathlib import Path
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader

from mod_res_damage.utils.utils import mutual_information


class Evaluator:
    def __init__(
            self,
            dataloader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            distributed: bool = False,
            inference_mode: str = 'sliding',
            num_monte_carlo: int | None = None,
            random_min_redundancy: int | None = None,
            random_max_redundancy: int | None = None,
            sliding_stride: int | None = None,
            inference_batch: int = None,
            use_wandb: bool = False,
            save_probs: bool = False,
            criterion: nn.Module | None = None,
            activation: nn.Module | None = None,
    ) -> None:
        self.rank = int(os.environ["RANK"])
        
        self.dataloader = dataloader
        self.exp_dir = exp_dir
        self.device = device
        self.distributed = distributed
        self.inference_mode = inference_mode
        self.num_monte_carlo = num_monte_carlo
        self.random_min_redundancy = random_min_redundancy
        self.random_max_redundancy = random_max_redundancy
        self.sliding_stride = sliding_stride
        self.inference_batch = inference_batch
        self.use_wandb = use_wandb
        self.save_probs = save_probs
        self.criterion = criterion
        self.activation = activation
        
        if self.save_probs:
            self.probs_dir = exp_dir / "save_probs"
            self.probs_dir.mkdir(exist_ok=True)
            
        if self.num_monte_carlo is not None:
            self.mc_pred_unc_path = self.exp_dir / "pred_uncertainty"
            self.mc_model_unc_path = self.exp_dir / "model_uncertainty"
            self.mc_pred_unc_path.mkdir(exist_ok=True)
            self.mc_model_unc_path.mkdir(exist_ok=True)

        self.input_size = self.dataloader.dataset.input_size
        self.classes = self.dataloader.dataset.classes
        self.split = self.dataloader.dataset.split
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])

        self.logger = logging.getLogger()
        if self.use_wandb:
            import wandb
            self.wandb = wandb

    def evaluate(
            self,
            model: torch.nn.Module,
            model_name: str,
            model_ckpt_path: str | Path | None = None,
    ) -> None:
        raise NotImplementedError

    def __call__(self, model, model_name, model_ckpt_path):
        self.evaluate(model, model_name, model_ckpt_path)

    def compute_metrics(self, confusion_matrix):
        pass

    def log_metrics(self, metrics):
        pass

    @staticmethod
    def sliding_inference(model, input, input_size, stride=None, max_batch=None):
        b, _, _, height, width = input[list(input.keys())[0]].shape

        if stride is None:
            h = int(math.ceil(height / input_size))
            w = int(math.ceil(width / input_size))
        else:
            h = math.ceil((height - input_size) / stride) + 1
            w = math.ceil((width - input_size) / stride) + 1

        h_grid = torch.linspace(0, height - input_size, h).round().long()
        w_grid = torch.linspace(0, width - input_size, w).round().long()

        num_crops_per_input = h * w

        input_cropped = {}
        for k, v in input.items():
            input_crops = []
            for i in range(h):
                for j in range(w):
                    input_crops.append(v[:, :, :, h_grid[i]:h_grid[i] + input_size, w_grid[j]:w_grid[j] + input_size])
            input_cropped[k] = torch.cat(input_crops, dim=0)

        pred = []
        max_batch = max_batch if max_batch is not None else b * num_crops_per_input
        batch_num = int(math.ceil(b * num_crops_per_input / max_batch))
        for i in range(batch_num):
            input_ = {k: v[max_batch * i: min(max_batch * i + max_batch, b * num_crops_per_input)] for k, v in input_cropped.items()}
            pred_ = model.forward(input_)
            pred.append(pred_)
        pred = torch.cat(pred, dim=0)
        pred = pred.view(num_crops_per_input, b, -1, input_size, input_size).transpose(0, 1)

        merged_pred = torch.zeros((b, pred.shape[2], height, width), device=pred.device)
        pred_count = torch.zeros((b, height, width), dtype=torch.long, device=pred.device)

        for i in range(h):
            for j in range(w):
                merged_pred[:, :, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += pred[:, (w * i) + j]
                pred_count[:, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += 1

        merged_pred = merged_pred / pred_count.unsqueeze(1)

        return merged_pred
    

class SegEvaluator(Evaluator):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )

        if self.num_monte_carlo is not None:
            output_pred_unc = []
            output_model_unc = []
            
        loss = []
        for batch_idx, data in enumerate(self.dataloader):
            input, target = data["input"], data["target"]
            input = {k: v.to(self.device) for k, v in input.items()}
            target = target.to(self.device)

            if self.num_monte_carlo is not None:        
                output_mc = []
                logits_mc = []
                end_time = time.time()
                self.logger.info(f"Starting monte carlo inference for batch {batch_idx} / {len(self.dataloader)} {model_name} for {self.num_monte_carlo} runs")
                for _ in range(self.num_monte_carlo):
                    logits = self._inference(model, input)
                    logits_mc.append(logits)
                    output_mc.append(self.activation(logits))
                self.logger.info(f"Completed monte carlo inference for batch {batch_idx} / {len(self.dataloader)} {model_name} for {self.num_monte_carlo} runs in {time.time()-end_time:.3f}")
                output_mc = torch.stack(output_mc)
                pred_unc, model_unc = mutual_information(output_mc.flatten(-2).data.cpu().numpy())
                output_pred_unc.append(pred_unc)
                output_model_unc.append(model_unc)
                
                logits = torch.stack(logits_mc).mean(dim=0)
                probs = output_mc.mean(dim=0)
            else:
                logits = self._inference(model, input)
                probs = self.activation(logits) 
            loss.append(self.criterion(logits, target))
            
            if self.save_probs:
                self._save_probs(probs, data, model_name)
            
            if probs.shape[1] == 1:
                pred = (probs > 0.5).type(torch.int64)
            else:
                pred = torch.argmax(probs, dim=1, keepdim=True)
            count = torch.bincount(
                (pred * self.num_classes + target.long()).flatten(), minlength=self.num_classes ** 2
            )
            confusion_matrix += count.view(self.num_classes, self.num_classes)

        if self.num_monte_carlo is not None:
            output_pred_unc = np.concatenate(output_pred_unc, axis=-1)
            output_model_unc = np.concatenate(output_model_unc, axis=-1)
            np.save(self.mc_pred_unc_path / f"{model_name}.npy", output_pred_unc)
            np.save(self.mc_model_unc_path / f"{model_name}.npy", output_model_unc)
        
        if self.distributed:
            torch.distributed.all_reduce(
                confusion_matrix, op=torch.distributed.ReduceOp.SUM
            )
        loss = torch.stack(loss).mean(dim=0)
        metrics = self.compute_metrics(confusion_matrix.cpu())
        metrics["loss"] = loss.item()
        metrics.update({"loss": loss.item()})
        self.log_metrics(metrics)

        used_time = time.time() - t
        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)
    
    def _inference(self, model, input):
        if self.inference_mode == "sliding":
            logits = self.sliding_inference(model, input, self.input_size, stride=self.sliding_stride, max_batch=self.inference_batch)
        elif self.inference_mode == "whole":
            logits = model(input)
        else:
            raise NotImplementedError((f"Inference mode {self.inference_mode} is not implemented."))
        return logits
    
    def _save_probs(self, probs, data, model_name):
        probs = probs.cpu().numpy()
        indxes = data["index"]
        tif_list = [self.dataloader.dataset.tif_list[i] for i in indxes]
        out_dir = self.probs_dir / f"{model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for tif, prob in zip(tif_list, probs):
            with rasterio.open(tif) as src:
                profile = src.profile.copy()
                transform = from_bounds(
                    *src.bounds,
                    width=prob.shape[-1],
                    height=prob.shape[-2]
                )
            profile.update(
                dtype=rasterio.float32,
                count=self.num_classes,
                compress="lzw",
                width=prob.shape[-1],
                height=prob.shape[-2],
                transform=transform
            )
            out_path = out_dir / Path(tif).name
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(prob)

    def compute_metrics(self, confusion_matrix):
        # Calculate IoU for each class
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6)) * 100

        # Calculate precision and recall for each class
        precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
        recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        miou = iou.mean().item()
        mf1 = f1.mean().item()
        macc = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100

        # Convert metrics to CPU and to Python scalars
        iou = [m.item() for m in iou]
        f1 = [m.item() for m in f1]
        precision = [m.item() for m in precision]
        recall = [m.item() for m in recall]
        
        # Prepare the metrics dictionary
        metrics = {
            "IoU": iou,
            "mIoU": miou,
            "F1": f1,
            "mF1": mf1,
            "mAcc": macc,
            "Precision": precision,
            "Recall": recall,
        }

        return metrics

    def log_metrics(self, metrics):
        def format_metric(name, values, mean_value):
            header = f"[{self.split}] ------- {name} --------\n"
            metric_str = (
                    "\n".join(
                        c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                        for c, num in zip(self.classes, values)
                    )
                    + "\n"
            )
            mean_str = (
                    f"[{self.split}]-------------------\n"
                    + f"[{self.split}] Mean".ljust(self.max_name_len, " ")
                    + "\t{:>7}".format("%.3f" % mean_value)
            )
            return header + metric_str + mean_str

        loss_str = f"[{self.split}] ------- Loss --------\n"+"\t{:>7}".format("%.3f" % metrics["loss"])
        iou_str = format_metric("IoU", metrics["IoU"], metrics["mIoU"])
        f1_str = format_metric("F1-score", metrics["F1"], metrics["mF1"])

        precision_mean = sum(metrics["Precision"]) / len(metrics["Precision"]) if metrics["Precision"] else 0.0
        recall_mean = sum(metrics["Recall"]) / len(metrics["Recall"]) if metrics["Recall"] else 0.0

        precision_str = format_metric("Precision", metrics["Precision"], precision_mean)
        recall_str = format_metric("Recall", metrics["Recall"], recall_mean)

        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"

        self.logger.info(loss_str)
        self.logger.info(iou_str)
        self.logger.info(f1_str)
        self.logger.info(precision_str)
        self.logger.info(recall_str)
        self.logger.info(macc_str)

        if self.use_wandb and self.rank == 0:
            self.wandb.log(
                {
                    f"{self.split}_loss": metrics["loss"],
                    f"{self.split}_mIoU": metrics["mIoU"],
                    f"{self.split}_mF1": metrics["mF1"],
                    f"{self.split}_mAcc": metrics["mAcc"],
                    **{
                        f"{self.split}_IoU_{c}": v
                        for c, v in zip(self.classes, metrics["IoU"])
                    },
                    **{
                        f"{self.split}_F1_{c}": v
                        for c, v in zip(self.classes, metrics["F1"])
                    },
                    **{
                        f"{self.split}_Precision_{c}": v
                        for c, v in zip(self.classes, metrics["Precision"])
                    },
                    **{
                        f"{self.split}_Recall_{c}": v
                        for c, v in zip(self.classes, metrics["Recall"])
                    },
                },
            )