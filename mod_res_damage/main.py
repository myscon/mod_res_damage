import os as os
import pathlib
import time
import hydra
import torch
import torch.nn as nn

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from mod_res_damage.logger import init_logger
from mod_res_damage.trainer import Trainer
from mod_res_damage.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)

REPO_DIR = pathlib.Path(__file__).parent.parent
WORK_DIR = REPO_DIR / "data"
EXPERIMENT_DIR = REPO_DIR / "experiments"
CONFIG_DIR = REPO_DIR / "configs"


@hydra.main(config_name=None, version_base=None, config_path=str(CONFIG_DIR))
def main(cfg: DictConfig) -> None:
    if "seed" not in cfg:
        cfg.seed = int(time.time())
    if cfg.fix_seed:
        fix_seed(cfg.seed)

    if cfg.distributed:
        torch.distributed.init_process_group(backend="nccl")
    else:
        os.environ["RANK"] = '0'
        # os.environ["LOCAL_RANK"] = f"{int(os.environ['SLURM_ARRAY_TASK_ID']) % 2}"
        os.environ["LOCAL_RANK"] = '0'
        
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    
    hydra_conf = HydraConfig.get()
    exp_dir = Path(hydra_conf.runtime.output_dir)
    exp_name = exp_dir.name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger_path = exp_dir / "out.log"
    
    if cfg.use_wandb and rank == 0:
        import wandb
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project="mod_res_damage_cv",
            name=exp_name,
            dir=exp_dir,
            config=wandb_cfg,
        )

    logger = init_logger(logger_path, rank=rank)
    logger.info("Experiment name: %s" % exp_name)
    logger.info("Device name: %s" % device)
    logger.info("The experiment is stored in %s" % exp_dir)
    
    model: nn.Module = instantiate(cfg.model)
    model_total_param_count = sum(p.numel() for p in model.parameters())
    model_trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    
    logger.info("Built {}.".format(model.__class__.__module__))
    logger.info("Model Total Parameter Count: {value:,}".format(value=model_total_param_count))
    logger.info("Model Trainable Parameter Count: {value:,}".format(value=model_trainable_param_count))

    train_dataset: Dataset = instantiate(cfg.dataset, split="train")
    val_dataset: Dataset = instantiate(cfg.dataset, split="val")

    logger.info("Built train dataset.".format(train_dataset.__class__.__module__))
    logger.info("Built val dataset.".format(val_dataset.__class__.__module__))
    logger.info(
        f"Total number of train samples: {len(train_dataset)}\n"
        f"Total number of validation samples: {len(val_dataset)}"
    )

    if cfg.distributed:
        logger.info("Preparing distributed data parallel.")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.find_unused_parameters,
        )
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    else:
        train_sampler, val_sampler = None, None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        worker_init_fn=seed_worker,
        generator=get_generator(cfg.seed),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.test_batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        worker_init_fn=seed_worker,
        drop_last=False,
    )
    if hasattr(train_dataset, 'class_weights'):
        weight = train_dataset.class_weights.to(device)
        criterion = instantiate(cfg.criterion, weight=weight)
    else:
        criterion = instantiate(cfg.criterion)
    activation = instantiate(cfg.activation)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = instantiate(
        cfg.lr_scheduler,
        optimizer=optimizer,
    )

    val_evaluator = instantiate(
        cfg.evaluator,
        dataloader=val_loader,
        exp_dir=exp_dir,
        device=device,
        distributed=cfg.distributed,
        use_wandb=cfg.use_wandb,
        activation=activation,
        criterion=criterion
    )
    trainer: Trainer = instantiate(
        cfg.trainer,
        model=model,
        train_loader=train_loader,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        criterion=criterion,
        evaluator=val_evaluator,
        ckpt_dir=ckpt_dir,
        device=device,
        distributed=cfg.distributed,
        cudnn_backend=cfg.cudnn_backend,
        use_wandb=cfg.use_wandb
    )

    trainer.train()
    if cfg.evaluator.save_output:
        holdout = None
    else:
        holdout = cfg.dataset.holdout

    test_dataset: Dataset = instantiate(cfg.dataset, holdout=holdout, split="test")
    test_sampler = DistributedSampler(test_dataset) if cfg.distributed else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=False,
    )
    test_evaluator = instantiate(
        cfg.evaluator,
        dataloader=test_loader,
        exp_dir=exp_dir,
        device=device,
        distributed=cfg.distributed,
        use_wandb=cfg.use_wandb,
        activation=activation,
        criterion=criterion
    )

    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(ckpt_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(ckpt_dir)
    test_evaluator.evaluate(model, model_ckpt_path)
    
    if cfg.distributed:
        torch.distributed.destroy_process_group()

    if cfg.use_wandb and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
