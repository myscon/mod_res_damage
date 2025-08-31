import numpy as np
import pandas as pd
import random
import rasterio
import re
import torch

from glob import glob
from pathlib import Path
from rasterio.windows import Window, bounds, from_bounds
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


class ModifiedBright(Dataset):
    classes = ['background', 'intact', 'damaged', 'destroyed']
    num_classes = len(classes)

    def __init__(
        self,
        split: str,
        root_path: str,
        input_size: int,
        cache_arrays: bool = True,
        augment: int | None = None,
        holdout: list[str] | None = ["ukraine-conflict"],
        train_group_0_only: bool = False,
        threshold: bool = False,
    ):
        super().__init__()
        self.split = split
        self.root_path = root_path
        self.input_size = input_size
        self.cache_arrays = cache_arrays
        self.augment = augment
        self.holdout = holdout
        self.train_group_0_only = train_group_0_only
        self.threshold = threshold

        self.root_path = Path(root_path)
        self.radar_before_dir = self.root_path / "radar" / "before"
        self.radar_after_dir = self.root_path / "radar" / "after"
        self.mask_dir = self.root_path / "labels"
        
        self.radar_counts = pd.read_csv(self.root_path / "radar_counts.csv")
        self.label_counts = pd.read_csv(self.root_path / "label_counts.csv")
        
        self.tif_list = glob(str(self.mask_dir) + "/*.tif")
        self.re = re.compile(r'^(?P<event>.*?)_group-(?P<group>\d+)\.tif$')
        
        self._apply_holdout()
        self._init_stats()
        if self.split == "train":
            self._init_class_weights()
        if self.cache_arrays:
            self._cache_arrays()
    

    def __len__(self):
        if self.augment is not None and self.split == "train":
            return len(self.tif_list) * self.augment
        else:
            return len(self.tif_list)

    def __getitem__(self, index):
        if self.augment is not None and self.split == "train":
            index = index % len(self.tif_list)
        if self.cache_arrays:
            sar, target = self._read_cache(index)
        else:
            sar, target = self._read_tifs(index)
        
        if self.augment is not None and self.split == "train":
            if random.random() < 0.50:
                sar = F.hflip(sar)
                target = F.hflip(target)
            if random.random() < 0.50:
                sar = F.vflip(sar)
                target = F.vflip(target)
            if random.random() < 0.25:
                angle = random.choice([90, 180, 270])
                sar = F.rotate(sar, angle)
                target = F.rotate(target, angle)

        if self.input_size is not None:
            if sar.shape[-1] < self.input_size:
                    sdiff = self.input_size - sar.shape[-1]
                    if self.augment is not None and self.split == "train":
                        hdiff = np.random.randint(0, sdiff+1)
                    else:
                        hdiff = sdiff // 2
                    sar = F.pad(sar, padding=[hdiff, 0, sdiff - hdiff, 0])
                    target = F.pad(target, padding=[hdiff, 0, sdiff - hdiff, 0])
            if sar.shape[-2] < self.input_size:
                    sdiff = self.input_size - sar.shape[-2]
                    if self.augment is not None and self.split == "train":
                        hdiff = np.random.randint(0, sdiff+1)
                    else:
                        hdiff = sdiff // 2
                    sar = F.pad(sar, padding=[0, hdiff, 0, sdiff - hdiff])
                    target = F.pad(target, padding=[0, hdiff, 0, sdiff - hdiff])

        if self.threshold:
            before = (target > 0).long()
            after = (target == 1).long()
            return {
                "input": {"S1RTC": sar},
                "target": {
                    "target_before": before,
                    "target_after": after,
                "index": index
                }
            }
        else:
            return {
                "input": {"S1RTC": sar},
                "target": target,
                "index": index
            }
    
    def _apply_holdout(self):
        if self.holdout is not None:      
            if self.split == "train":
                self.tif_list = [l for l in self.tif_list if all([h not in l for h in self.holdout])]
            else:
                self.tif_list = [l for l in self.tif_list if all([h in l for h in self.holdout])]
            assert len(self.tif_list) > 0; "Tif list length must be greater than 0. Check holdout str."
            
            for h in self.holdout:
                self.radar_counts = self.radar_counts.loc[self.radar_counts["event"] != h]
                self.radar_counts = self.radar_counts.loc[self.radar_counts["group"] != h]
                self.label_counts = self.label_counts.loc[self.label_counts["event"] != h]
                self.label_counts = self.label_counts.loc[self.label_counts["group"] != h]
    
    def _init_stats(self):
        self.data_means = []
        self.data_stdvs = []

        for p in ["VV", "VH"]:
            subset = self.radar_counts[self.radar_counts["p"] == p]
            S = subset["S"].sum()
            Q = subset["Q"].sum()
            c = subset["count"].sum()
            mean = S / c
            variance = (Q - (S**2 / c)) / (c - 1)
            self.data_means.append(mean)
            self.data_stdvs.append(variance**0.5)

        self.data_means = torch.tensor(self.data_means, dtype=torch.float).view(1, 2, 1, 1)
        self.data_stdvs = torch.tensor(self.data_stdvs, dtype=torch.float).view(1, 2, 1, 1)
        
    def _init_class_weights(self):
        counts = torch.tensor(list(self.label_counts[self.classes].sum()), dtype=torch.float)
        self.class_weights = 1.0 / (counts + 1e-6)
        self.class_weights = self.class_weights / self.class_weights.sum()
        
    def _cache_arrays(self):
        self.target_list = []
        self.input_list = []
        for tif in self.tif_list:
            name = Path(tif).name
            m = self.re.match(name)
            event = m['event']
            group = int(m['group'])
            with rasterio.open(tif) as tsrc:
                target = torch.tensor(tsrc.read()).long()
                target = torch.nn.functional.one_hot(target.squeeze(0), num_classes=len(self.classes))
                target = target.permute(2, 0, 1).float()
                
                self.target_list.append(target)
                bounds = tsrc.bounds
                rh, rw = tsrc.height, tsrc.width
            dirs = [self.radar_before_dir, self.radar_after_dir]
            
            inputs = []
            for dir in dirs:
                paths = [dir.joinpath(f"{event}_group-{group}_VV.tif"),
                        dir.joinpath(f"{event}_group-{group}_VH.tif")]
                sars = []
                for path in paths:
                    with rasterio.open(path) as psrc:
                        window = from_bounds(*bounds, transform=psrc.transform)
                        input = psrc.read(window=window, out_shape=(psrc.count, rh, rw))
                        assert input.shape[-2:] == target.shape[-2:]; f"Mismatched input and target shapes {input.shape} {target.shape}"
                        sars.append(input)
                sars = np.concatenate(sars)
                inputs.append(sars)
            inputs = np.stack(inputs, axis=1)
            self.input_list.append(torch.tensor(inputs).float())
    
    def _read_cache(self, index):
        # NOTE: who knows if this even saves time...
        target = self.target_list[index]
        input = self.input_list[index]
        _, H, W = target.shape
        if self.split == "train":
            if H > self.input_size:
                h = np.random.randint(0, H - self.input_size)
                he = h + self.input_size
            else:
                h, he = 0, H
            if W > self.input_size:
                w = np.random.randint(0, W - self.input_size)
                we = w + self.input_size
            else:
                w, we = 0, W
            target = target[:, h: he, w: we]
            input = input[:, :, h: he, w: we]
        input = (input - self.data_means) / self.data_stdvs


        return input, target
    
    def _read_tifs(self, index):
        target_path = Path(self.tif_list[index])
        name = target_path.name
        m = self.re.match(name)
        event = m['event']
        group = int(m['group'])

        with rasterio.open(target_path) as src:
            if src.height <= self.input_size or self.split != "train":
                rh = src.height
                row_off = 0
            else:
                rh = self.input_size
                row_off = np.random.randint(0, src.height - rh)
            if src.width <= self.input_size or self.split != "train":
                rw = src.width
                col_off = 0
            else:
                rw = self.input_size
                col_off = np.random.randint(0, src.width - rw)
            window = Window(col_off=col_off, row_off=row_off, height=rh, width=rw)
            window_bounds = bounds(window, src.transform)
            target = src.read(window=window,out_shape=(src.count, rh, rw))
        sar = self._open_sar(event, group, window_bounds, rh, rw)
        
        target = torch.tensor(target).long()
        target = torch.nn.functional.one_hot(target.squeeze(0), num_classes=len(self.classes))
        target = target.permute(2, 0, 1).float()
        sar = (sar - self.data_means) / self.data_stdvs

        return sar, target
                        
    def _open_sar(self, event, group, window_bounds, rh, rw):
        dirs = [self.radar_before_dir, self.radar_after_dir]
        sars = []
        for dir in dirs:
            sar_arrs = []
            paths = [dir.joinpath(f"{event}_group-{group}_VV.tif"),
                    dir.joinpath(f"{event}_group-{group}_VH.tif")]
            for path in paths:
                with rasterio.open(path) as src:
                    window = from_bounds(*window_bounds, transform=src.transform)
                    ras = torch.tensor(src.read(window=window, out_shape=(src.count, rh, rw))).float()
                    sar_arrs.append(ras)
            sars.append(torch.cat(sar_arrs))
        return torch.stack(sars, dim=1).float()
