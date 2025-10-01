import ast
import geopandas as gpd
import numpy as np
import pandas as pd
import random
import rasterio
import re
import torch

from glob import glob
from pathlib import Path
from rasterio.features import rasterize
from rasterio.windows import Window, bounds, from_bounds, transform
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


class ModifiedBright(Dataset):
    predictands = ['damage']
    
    def __init__(
        self,
        split: str,
        root_path: str,
        input_size: int,
        cache_arrays: bool = True,
        average_time: bool = True,
        augment: int | None = None,
        holdout: list[str] | None = ["ukraine-conflict"],
    ):
        super().__init__()
        self.split = split
        self.root_path = root_path
        self.input_size = input_size
        self.cache_arrays = cache_arrays
        self.average_time = average_time
        self.augment = augment
        self.holdout = holdout

        self.root_path = Path(root_path)
        self.radar_dir = self.root_path / "gee_s1_exports"
        self.mask_dir = self.root_path / "labels"
        
        self.label_stats = pd.read_csv(self.root_path / "label_stats.csv")
        self.radar_stats = pd.read_csv(self.root_path / "radar_stats.csv")
        
        self.tif_list = glob(str(self.mask_dir) + "/*.tif")
        self.re = re.compile(r'^(?P<event>.*?)_group-(?P<group>\d{1})\.tif$')
        
        self._apply_holdout()
        self._init_stats()
        self._init_event_polygons()
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
            sar, target, no_data = self._read_cache(index)
        else:
            sar, target, no_data = self._read_tifs(index)
        
        if self.augment is not None and self.split == "train":
            no_data = no_data.unsqueeze(0)
            if random.random() < 0.50:
                sar = F.hflip(sar)
                target = F.hflip(target)
                no_data = F.hflip(no_data)
            if random.random() < 0.50:
                sar = F.vflip(sar)
                target = F.vflip(target)
                no_data = F.vflip(no_data)
            if random.random() < 0.25:
                angle = random.choice([90, 180, 270])
                sar = F.rotate(sar, angle)
                target = F.rotate(target, angle)
                no_data = F.rotate(no_data, angle)
            no_data = no_data.squeeze(0)
                
        if self.input_size is not None:
            if sar.shape[-1] < self.input_size:
                    sdiff = self.input_size - sar.shape[-1]
                    if self.augment is not None and self.split == "train":
                        hdiff = np.random.randint(0, sdiff+1)
                    else:
                        hdiff = sdiff // 2
                    sar = F.pad(sar, padding=[hdiff, 0, sdiff - hdiff, 0])
                    target = F.pad(target, padding=[hdiff, 0, sdiff - hdiff, 0])
                    no_data = F.pad(no_data, padding=[hdiff, 0, sdiff - hdiff, 0])
            if sar.shape[-2] < self.input_size:
                    sdiff = self.input_size - sar.shape[-2]
                    if self.augment is not None and self.split == "train":
                        hdiff = np.random.randint(0, sdiff+1)
                    else:
                        hdiff = sdiff // 2
                    sar = F.pad(sar, padding=[0, hdiff, 0, sdiff - hdiff])
                    target = F.pad(target, padding=[0, hdiff, 0, sdiff - hdiff])
                    no_data = F.pad(no_data, padding=[0, hdiff, 0, sdiff - hdiff])

        return {
                "inputs": {"S1GRD": sar},
                "target": target,
                "no_data": no_data,
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
                self.radar_stats = self.radar_stats.loc[self.radar_stats["event"] != h]

    def _init_stats(self):            
        self.radar_stats['mean'] = self.radar_stats['mean'].apply(lambda x: np.array(ast.literal_eval(x)))
        self.radar_stats['stdv'] = self.radar_stats['stdv'].apply(lambda x: np.array(ast.literal_eval(x)))
        self.radar_stats['count'] = self.radar_stats['count'].astype(float)

        total_count = self.radar_stats['count'].sum()
        global_mean = sum(self.radar_stats['count'].values[:, None] * np.stack(self.radar_stats['mean'].values)) / total_count

        means = np.stack(self.radar_stats['mean'].values)
        stds = np.stack(self.radar_stats['stdv'].values)
        counts = self.radar_stats['count'].values[:, None]

        within_var = (counts - 1) * (stds ** 2)
        between_var = counts * ((means - global_mean) ** 2)

        global_var = (within_var + between_var).sum(axis=0) / (counts.sum() - 1)
        global_stdv = np.sqrt(global_var)

        self.data_means = torch.tensor(global_mean, dtype=torch.float).view(2, 1, 1, 1)
        self.data_stdvs = torch.tensor(global_stdv, dtype=torch.float).view(2, 1, 1, 1)

    def _cache_arrays(self):
        self.target_list = []
        self.input_list = []
        self.no_data_list = []
        self.non_zero_coords = []
        
        for tif in self.tif_list:
            name = Path(tif).name
            m = self.re.match(name)
            event = m['event']
            group = int(m['group'])
            
            label_stats = self.label_stats.loc[(self.label_stats["group"] == group) & 
                                               (self.label_stats["event"] == event)]
            label_mean = np.array(label_stats['mean'])
            label_stdv = np.array(label_stats['stdv'])
            
            before_paths = sorted(self.radar_dir.glob(f"*{event}_group-{group}_1*.tif"))
            after_paths = sorted(self.radar_dir.glob(f"*{event}_group-{group}_2*.tif"))
            t_file = rasterio.open(tif)
            
            before_files = [rasterio.open(p) for p in before_paths]
            after_files =  [rasterio.open(p) for p in after_paths]
            all_files = [t_file] + before_files + after_files
            
            left = max([f.bounds.left for f in all_files])
            bottom = max([f.bounds.bottom for f in all_files])
            right = min([f.bounds.right for f in all_files])
            top = min([f.bounds.top for f in all_files])
            bounds = [left, bottom, right, top]
            
            t_win = from_bounds(*bounds, t_file.transform)
            t_arr = t_file.read(window=t_win)
            t_arr = (t_arr - label_mean) / label_stdv
            
            h, w = t_arr.shape[1:]
            before_arrs = [f.read(window=from_bounds(*bounds, f.transform)) for f in before_files]
            after_arrs = [f.read(window=from_bounds(*bounds, f.transform)) for f in after_files]
            before_sar = torch.tensor(np.stack(before_arrs, axis=1))
            after_sar = torch.tensor(np.stack(after_arrs, axis=1))
            if self.average_time:
                before_sar = (before_sar - self.data_means) / self.data_stdvs
                before_sar = before_sar.mean(dim=1, keepdims=True)
                after_sar = (after_sar - self.data_means) / self.data_stdvs
                after_sar = after_sar.mean(dim=1, keepdims=True)
                sar = torch.concat([before_sar, after_sar], dim=1)
            else:
                sar = torch.concat([before_sar, after_sar], dim=1)
                sar = (sar - self.data_means) / self.data_stdvs
            
            self.input_list.append(sar.to(torch.float))
            self.target_list.append(torch.tensor(t_arr, dtype=torch.float))
            
            # i could spend time to clean the data but no, let's do it the silly way
            win_transform = rasterio.windows.transform(t_win, t_file.transform)
            subset = self._original_chips.loc[(self._original_chips["group"] == group) &
                                              (self._original_chips['event'] == event)]
            subset = subset.to_crs(t_file.crs)
            no_data = rasterize(
                subset.geometry,
                out_shape=(h, w),
                transform=win_transform,
                dtype=np.uint8
            )   
            self.no_data_list.append(no_data)
            
            [f.close for f in all_files]
    
    def _init_event_polygons(self):
        original_chips = gpd.read_file(self.root_path / "original_chips.geojson")
        groups = list(original_chips["group"].unique())
        events = list(original_chips["event"].unique())
        
        self._original_chips = []
        for event in events:
            for group in groups:
                subset = original_chips.loc[(original_chips["group"] == group) &
                                            (original_chips['event'] == event)]
                if len(subset) > 0:
                    original_polygon = unary_union(subset.geometry)
                    self._original_chips.append({"event": event, "group": group, "geometry": original_polygon})
        self._original_chips = gpd.GeoDataFrame(self._original_chips, geometry="geometry").set_crs(original_chips.crs)
        
    
    def _read_cache(self, index):
        # NOTE: who knows if this even saves time...
        target = self.target_list[index]
        inputs = self.input_list[index]
        no_data = self.no_data_list[index]
        _, H, W = target.shape

        if self.split == "train":
            non_zero_coords = torch.argwhere(no_data)
            idx = np.random.randint(non_zero_coords.shape[0])
            y, x = non_zero_coords[idx]

            h = max(0, y.item() - self.input_size // 2)
            he = h + self.input_size
            if he > H:
                he = H

            w = max(0, x.item() - self.input_size // 2)
            we = w + self.input_size
            if we > W:
                we = W

            target = target[:, h:he, w:we]
            inputs = inputs[:, :, h:he, w:we]
            no_data = no_data[:, h:he, w:we]
        
        return inputs, target, no_data
