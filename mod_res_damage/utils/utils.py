import numpy as np
import random
import torch

from pathlib import Path
from typing import Optional


class RunningAverageMeter(object):
    def __init__(self, length=100):
        self.queue = []
        self.ptr = 0
        self.length = length

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.queue = []
        self.ptr = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if len(self.queue) < self.length:
            self.queue.append(val)
        else:
            self.queue[self.ptr] = val
            self.ptr = (self.ptr + 1) % self.length

        self.val = val
        self.sum = sum(self.queue)
        self.count = len(self.queue)
        self.avg = self.sum / self.count


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

  
def _find_ckpt(ckpt_dir: str | Path, suffix: str) -> Optional[str]:
    """Return the *first* file that ends with `suffix`; None if nothing found."""
    ckpt_dir = Path(ckpt_dir)
    for fname in ckpt_dir.iterdir():
        if fname.name.endswith(suffix):
            return str(fname)
    return None


def get_best_model_ckpt_path(ckpt_dir: str | Path) -> Optional[str]:
    return _find_ckpt(ckpt_dir, "_best.pth")


def get_final_model_ckpt_path(ckpt_dir: str | Path) -> Optional[str]:
    return _find_ckpt(ckpt_dir, "_final.pth")


def entropy(prob):
    return -np.sum(prob * np.log(prob + 1e-15), axis=1)


def mutual_information(mc_preds):
    # mc_preds shape: (MC, B, C, H, W)
    mean_probs = np.mean(mc_preds, axis=0)              # shape (B, C, H, W)
    entropy_mean = entropy(mean_probs)                  # shape (B, H, W)
    mean_entropy = np.mean(entropy(mc_preds), axis=0)   # shape (B, H, W)
    return entropy_mean, entropy_mean - mean_entropy    # shape (B, H, W)