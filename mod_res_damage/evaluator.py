import logging
import os
import time
import math
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            sliding_stride: int | None = None,
            inference_batch: int = None,
            use_wandb: bool = False,
            save_output: bool = False,
            task: str = 'segmentation',
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
        self.sliding_stride = sliding_stride
        self.inference_batch = inference_batch
        self.use_wandb = use_wandb
        self.save_output = save_output
        self.task = task
        self.criterion = criterion
        self.activation = activation
        
        if self.save_output:
            self.output_dir = exp_dir / "save_output"
            self.output_dir.mkdir(exist_ok=True)
            
        if self.num_monte_carlo is not None:
            self.mc_pred_unc_path = self.exp_dir / "pred_uncertainty"
            self.mc_model_unc_path = self.exp_dir / "model_uncertainty"
            self.mc_pred_unc_path.mkdir(exist_ok=True)
            self.mc_model_unc_path.mkdir(exist_ok=True)

        self.input_size = self.dataloader.dataset.input_size
        self.predictands = self.dataloader.dataset.predictands
        self.split = self.dataloader.dataset.split
        self.num_predictands = len(self.predictands)
        self.max_name_len = max([len(name) for name in self.predictands])

        self.logger = logging.getLogger()
        if self.use_wandb:
            import wandb
            self.wandb = wandb

    def __call__(self, model, model_name, model_ckpt_path):
        self.evaluate(model, model_name, model_ckpt_path)

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model_dict = model_dict["model"]
                
            if self.distributed:
                model.module.load_state_dict(model_dict)
            else:
                model.load_state_dict(model_dict)
    
            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        if self.task == 'segmentation':
            confusion_matrix = torch.zeros(
                (self.num_predictands, self.num_predictands), device=self.device
            )

        if self.num_monte_carlo is not None:
            output_pred_unc = []
            output_model_unc = []
            
        loss = []
        for batch_idx, data in enumerate(self.dataloader):
            inputs, target, no_data = data["inputs"], data["target"], data["no_data"]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            target = target.to(self.device)
            no_data = no_data.to(self.device)

            if self.num_monte_carlo is not None:        
                output_mc = []
                logits_mc = []
                end_time = time.time()
                self.logger.info(f"Starting monte carlo inference for batch {batch_idx} / {len(self.dataloader)} {model_name} for {self.num_monte_carlo} runs")
                for _ in range(self.num_monte_carlo):
                    logits = self._inference(model, inputs)
                    logits_mc.append(logits)
                    output_mc.append(self.activation(logits))
                self.logger.info(f"Completed monte carlo inference for batch {batch_idx} / {len(self.dataloader)} {model_name} for {self.num_monte_carlo} runs in {time.time()-end_time:.3f}")
                output_mc = torch.stack(output_mc)
                pred_unc, model_unc = mutual_information(output_mc.flatten(-2).data.cpu().numpy())
                output_pred_unc.append(pred_unc)
                output_model_unc.append(model_unc)
                
                logits = torch.stack(logits_mc).mean(dim=0)
                output = output_mc.mean(dim=0)
            else:
                logits = self._inference(model, inputs)
                output = self.activation(logits) 
            
            if self.save_output:
                self._save_output(output, data, model_name)
            
            batch_loss = self.criterion(logits, target)[no_data].mean()
            loss.append(batch_loss)
            
            if self.task == 'segmentation':
                if output.shape[1] == 1:
                    pred = (output > 0.5).type(torch.int64)
                else:
                    pred = torch.argmax(output, dim=1, keepdim=True)
                pred = pred[no_data].view(pred.shape[0], self.num_predictands, -1)
                count = torch.bincount(
                    (pred * self.num_predictands + target.long()).flatten(), minlength=self.num_predictands ** 2
                )
                confusion_matrix += count.view(self.num_predictands, self.num_predictands)

        for i in range(len(self.dataloader)):
            if self.num_monte_carlo is not None:
                np.save(self.mc_pred_unc_path / f"{model_name}_{i}.npy", output_pred_unc[i])
                np.save(self.mc_model_unc_path / f"{model_name}_{i}.npy", output_model_unc[i])
        
        if self.distributed and self.task == 'segmentation':
            torch.distributed.all_reduce(
                confusion_matrix, op=torch.distributed.ReduceOp.SUM
            )
        metrics = self.compute_log_loss(loss)
        if self.task == 'segmentation':
            metrics.update(self.compute_log_seg_metrics(confusion_matrix.cpu()))
        elif self.task == 'regression':
            B, _, _, _ = output.shape
            metrics.update(self.compute_log_reg_metrics(output[no_data].view(B,self.num_predictands,-1), target[no_data].view(B,self.num_predictands,-1)))
        else:
            raise ValueError

        used_time = time.time() - t
        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)
    
    def _inference(self, model, inputs):
        if self.inference_mode == "sliding":
            logits = self.sliding_inference(model, inputs, self.input_size, max_batch=self.inference_batch, stride=self.sliding_stride)
        elif self.inference_mode == "random":
            logits = self.random_inference(model, inputs, self.input_size, max_batch=self.inference_batch)
        elif self.inference_mode == "whole":
            logits = model(inputs)
        else:
            raise NotImplementedError((f"Inference mode {self.inference_mode} is not implemented."))
        return logits
    
    def _save_output(self, output, data, model_name):
        output = output.cpu().numpy()
        indxes = data["index"]
        tif_list = [self.dataloader.dataset.tif_list[i] for i in indxes]
        out_dir = self.output_dir / f"{model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for tif, prob in zip(tif_list, output):
            with rasterio.open(tif) as src:
                profile = src.profile.copy()
                transform = from_bounds(
                    *src.bounds,
                    width=prob.shape[-1],
                    height=prob.shape[-2]
                )
            profile.update(
                dtype=rasterio.float32,
                count=prob.shape[0],
                compress="lzw",
                width=prob.shape[-1],
                height=prob.shape[-2],
                transform=transform
            )
            out_path = out_dir / Path(tif).name
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(prob)
    
    def compute_log_loss(self, loss):
        loss = torch.stack(loss).mean().item()
        loss_str = f"[{self.split}] ------- Loss --------\n"+"\t{:>7}".format("%.3f" % loss)
        self.logger.info(loss_str)
        if self.use_wandb and self.rank == 0:
            self.wandb.log({f"{self.split}_loss": loss})
        return {'loss': loss}

    def compute_log_seg_metrics(self, confusion_matrix, loss):
        metrics = {}

        # iou
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6)) * 100
        metrics['IoU'] = [m.item() for m in iou]
        metrics['mIoU'] = iou.mean().item()
        iou_str = self.format_metric("IoU", metrics["IoU"], metrics["mIoU"])
        self.logger.info(iou_str)

        # precision
        precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
        metrics['Precision'] = [m.item() for m in precision]
        precision_mean = sum(metrics["Precision"]) / len(metrics["Precision"]) if metrics["Precision"] else 0.0
        precision_str = self.format_metric("Precision", metrics["Precision"], precision_mean)
        self.logger.info(precision_str)

        # recall
        recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100
        metrics['Recall'] = [m.item() for m in recall]
        recall_mean = sum(metrics["Recall"]) / len(metrics["Recall"]) if metrics["Recall"] else 0.0
        recall_str = self.format_metric("Recall", metrics["Recall"], recall_mean)
        self.logger.info(recall_str)

        # f1
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        metrics['F1'] = [m.item() for m in f1]
        metrics['mF1'] = mf1 = f1.mean().item()
        f1_str = self.format_metric("F1-score", metrics["F1"], metrics["mF1"])
        self.logger.info(f1_str)

        # accuracy
        metrics['mAcc'] = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100
        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"
        self.logger.info(macc_str)
        
        if self.use_wandb and self.rank == 0:
            for key in metrics.keys():
                if key[0] == 'm':
                    wandb_log[f"{self.split}_{key}"] = metrics[key]
                else:
                    wandb_log.update({
                        f"{self.split}_{key}_{c}": v
                        for c, v in zip(self.predictands, metrics[key])
                    }),
            self.wandb.log(wandb_log)
        return metrics

    def compute_log_reg_metrics(self, output, target):
        metrics = {}
        error = target - output
        
        # mae
        abs_error = torch.abs(error)
        metrics['MAE'] = abs_error.mean(dim=(0,2))
        metrics['mMAE'] = abs_error.mean().item()
        mae_str = self.format_metric("MAE", metrics["MAE"], metrics["mMAE"])
        self.logger.info(mae_str)

        # mse
        sq_error = error ** 2
        metrics['MSE'] = sq_error.mean(dim=(0,2))
        metrics['mMSE'] = sq_error.mean().item()
        mse_str = self.format_metric("MSE", metrics["MSE"], metrics["mMSE"])
        self.logger.info(mse_str)

        # rmse
        rmse = torch.sqrt(sq_error)
        metrics['RMSE'] = sq_error.mean(dim=(0,2))
        metrics['mRMSE'] = rmse.mean().item()
        rmse_str = self.format_metric("RMSE", metrics["RMSE"], metrics["mRMSE"])
        self.logger.info(rmse_str)

        # r2
        mean_y = target.mean(dim=(0,2)) 
        ss_res = sq_error.sum(dim=(0,2))
        ss_tot = ((target - mean_y[None,:,None]) ** 2).sum(dim=(0,2)) 
        r2 = 1 - ss_res / (ss_tot + 1e-6)
        metrics['R2'] = r2
        metrics['mR2'] = r2.mean().item()
        r2_str = self.format_metric("R2", metrics["R2"], metrics["mR2"])
        self.logger.info(r2_str)

        # mape
        mape = torch.abs(error / (target + 1e-6)) * 100
        metrics['MAPE'] = mape.mean(dim=(0,2))
        metrics['mMAPE'] = mape.mean().item()
        mape_str = self.format_metric("MAPE", metrics["MAPE"], metrics["mMAPE"])
        self.logger.info(mape_str)

        if self.use_wandb and self.rank == 0:
            wandb_log = {}
            for key in metrics.keys():
                if key[0] == 'm':
                    wandb_log[f"{self.split}_{key}"] = metrics[key]
                else:
                    wandb_log.update({
                        f"{self.split}_{key}_{i}": v
                        for i, v in enumerate(metrics[key])
                    })
            self.wandb.log(wandb_log)

        return metrics

    def format_metric(self, name, values, mean_value):
        header = f"[{self.split}] ------- {name} --------\n"
        metric_str = (
                "\n".join(
                    c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                    for c, num in zip(self.predictands, values)
                )
                + "\n"
        )
        mean_str = (
                f"[{self.split}]-------------------\n"
                + f"[{self.split}] Mean".ljust(self.max_name_len, " ")
                + "\t{:>7}".format("%.3f" % mean_value)
        )
        return header + metric_str + mean_str
    
    @staticmethod
    def sliding_inference(model, inputs, input_size, max_batch=None, stride=None):
        b, _, _, height, width = inputs[list(inputs.keys())[0]].shape

        if stride is None:
            h = int(math.ceil(height / input_size))
            w = int(math.ceil(width / input_size))
        else:
            h = math.ceil((height - input_size) / stride) + 1
            w = math.ceil((width - input_size) / stride) + 1

        # NOTE: this feels silly but it works I think
        h_grid = torch.linspace(0, height - input_size, h).round().long()
        w_grid = torch.linspace(0, width - input_size, w).round().long()

        num_crops_per_input = h * w

        input_cropped = {}
        for k, v in inputs.items():
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
    
    @staticmethod
    def random_inference(model, inputs, input_size, max_batch=None):
        b, _, _, height, width = inputs[list(inputs.keys())[0]].shape
        
        stride = np.random.randint(input_size // 2, input_size)
        _h_start = np.random.randint(input_size // 2, input_size)
        _w_start = np.random.randint(input_size // 2, input_size)

        # NOTE: much better than the sliding inference...
        h_grid = torch.arange(-_h_start, height, step=stride)
        w_grid = torch.arange(-_w_start, width, step=stride)

        num_crops_per_input = len(h_grid) * len(w_grid)

        input_cropped = {}
        for k, v in inputs.items():
            input_crops = []
            for hi in h_grid:
                for wj in w_grid:
                    h_start, w_start = hi.item(), wj.item()
                    h_end, w_end = h_start + input_size, w_start + input_size

                    h0, h1 = max(h_start, 0), min(h_end, height)
                    w0, w1 = max(w_start, 0), min(w_end, width)
                    
                    crop = v[:, :, :, h0:h1, w0:w1]
                    pad_top = 0 if h_start >= 0 else -h_start
                    pad_left = 0 if w_start >= 0 else -w_start
                    pad_bottom = max(h_end - height, 0)
                    pad_right = max(w_end - width, 0)
                    if pad_top or pad_bottom or pad_left or pad_right:
                        # NOTE: for  the love of everthing F.pad from function is different from torchvision
                        crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom))
                    input_crops.append(crop)
                    
            input_cropped[k] = torch.cat(input_crops, dim=0)

        pred = []
        max_batch = max_batch if max_batch is not None else b * num_crops_per_input
        batch_num = int(math.ceil(b * num_crops_per_input / max_batch))

        for i in range(batch_num):
            input_ = {k: v[max_batch * i: min(max_batch * (i + 1), b * num_crops_per_input)]
                    for k, v in input_cropped.items()}
            pred_ = model.forward(input_)
            pred.append(pred_)
        pred = torch.cat(pred, dim=0)
        pred = pred.view(num_crops_per_input, b, -1, input_size, input_size).transpose(0, 1)

        merged_pred = torch.zeros((b, pred.shape[2], height, width), device=pred.device)
        pred_count = torch.zeros((b, height, width), dtype=torch.long, device=pred.device)

        idx = 0
        for hi in h_grid:
            for wj in w_grid:
                h_start, w_start = hi.item(), wj.item()
                h_end, w_end = h_start + input_size, w_start + input_size

                h0, h1 = max(h_start, 0), min(h_end, height)
                w0, w1 = max(w_start, 0), min(w_end, width)

                ph0 = h0 - h_start
                pw0 = w0 - w_start
                
                ph1 = input_size - (h_end - h1)
                pw1 = input_size - (w_end - w1)

                merged_pred[:, :, h0:h1, w0:w1] += pred[:, idx, :, ph0:ph1, pw0:pw1]
                pred_count[:, h0:h1, w0:w1] += 1
                idx += 1

        merged_pred = merged_pred / pred_count.unsqueeze(1)

        return merged_pred