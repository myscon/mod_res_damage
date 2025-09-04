import geopandas as gpd
import numpy as np
import pandas as pd
import random
import rasterio
import re
import torch

from glob import glob
from pathlib import Path
from rasterio.windows import Window, bounds, from_bounds
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


GSD = 10


class ModifiedBright(Dataset):
    classes = ['background', 'intact', 'damaged', 'destroyed']
    collapsed_classes = ['background', 'intact', 'damaged']

    def __init__(
        self,
        split: str,
        root_path: str,
        input_size: int,
        collapse_classes: True,
        cache_arrays: bool = True,
        augment: int | None = None,
        holdout: list[str] | None = ["ukraine-conflict"],
        threshold: bool = False,
    ):
        super().__init__()
        self.split = split
        self.root_path = root_path
        self.input_size = input_size
        self.collapse_classes = collapse_classes
        self.cache_arrays = cache_arrays
        self.augment = augment
        self.holdout = holdout
        self.threshold = threshold

        self.root_path = Path(root_path)
        self.radar_before_dir = self.root_path / "radar" / "before"
        self.radar_after_dir = self.root_path / "radar" / "after"
        self.mask_dir = self.root_path / "labels"
        
        self.radar_counts = pd.read_csv(self.root_path / "radar_counts.csv")
        self.label_counts = pd.read_csv(self.root_path / "label_counts.csv")
        
        self.tif_list = glob(str(self.mask_dir) + "/*.tif")
        self.re = re.compile(r'^(?P<event>.*?)_group-(?P<group>\d+)\.tif$')
        if self.collapse_classes:
            self.classes = self.collapsed_classes
        self.num_classes = len(self.classes)
        
        self._apply_holdout()
        self._init_stats()
        if self.split == "train":
            self._init_class_weights()
        if self.cache_arrays:
            self._cache_arrays()
        else:
            self._init_event_polygons()

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

        if self.threshold:
            before = (target > 0).long()
            after = (target == 1).long()
            return {
                "input": {"S1RTC": sar},
                "target": {
                    "target_before": before,
                    "target_after": after,
                "no_data": no_data,
                "index": index
                }
            }
        else:
            return {
                "input": {"S1RTC": sar},
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
        if self.collapse_classes:
            counts = torch.tensor([counts[0],
                                   counts[1],
                                   counts[-2:].sum()])

        self.class_weights = 1.0 / (counts + 1e-6)
        self.class_weights = self.class_weights / self.class_weights.sum()
        self.class_weights = self.class_weights / self.class_weights.max()

        
    def _cache_arrays(self):
        self.target_list = []
        self.input_list = []
        self.no_data_list = []
        
        for tif in self.tif_list:
            name = Path(tif).name
            m = self.re.match(name)
            event = m['event']
            group = int(m['group'])
            with rasterio.open(tif) as tsrc:
                arr = tsrc.read()
                if self.collapse_classes:
                    arr = np.where(arr >= 2, 2, arr)

                no_data = torch.tensor(np.where(arr == -1, 0.0, 1.0))
                self.no_data_list.append(no_data)
                
                target = torch.tensor(arr).long().contiguous()
                target[target == -1] = 0
                target = torch.nn.functional.one_hot(target.squeeze(0), num_classes=self.num_classes)
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
    
    def _init_event_polygons(self):
        original_polygons = gpd.read_file(self.root_path / "original_polygons.geojson")
        groups = list(original_polygons["group"].unique())
        events = list(original_polygons["events"].unique())
        
        self._original_polygons = {}
        for event, group in zip(events, groups):
            subset = original_polygons.loc[(self.original_chips["group"] == group) &
                                            (self.original_chips['event'] == event)]
            original_polygon = unary_union(subset.geometry)
            self._original_polygons.append({"event": event, "group": group, "geometry": original_polygon})
        self._original_polygons = gpd.GeoDataFrame(self._original_polygons, geometry="geometry")
        
    
    def _read_cache(self, index):
        # NOTE: who knows if this even saves time...
        target = self.target_list[index]
        input = self.input_list[index]
        no_data = self.no_data_list[index]
        _, H, W = target.shape

        if self.split == "train":
            non_zero_coords = torch.argwhere(no_data[0] == 1)
            idx = np.random.randint(non_zero_coords.shape[0])
            y, x = non_zero_coords[idx]

            h = max(0, y.item() - self.input_size // 2)
            he = h + self.input_size
            if he > H:
                he = H
                h = H - self.input_size

            w = max(0, x.item() - self.input_size // 2)
            we = w + self.input_size
            if we > W:
                we = W
                w = W - self.input_size

            target = target[:, h:he, w:we]
            input = input[:, :, h:he, w:we]
            no_data = no_data[:, h:he, w:we]
        input = (input - self.data_means) / self.data_stdvs
        no_data = no_data.squeeze(0)
        
        return input, target, no_data
    
    def _read_tifs(self, index):
        target_path = Path(self.tif_list[index])
        name = target_path.name
        m = self.re.match(name)
        event = m['event']
        group = int(m['group'])

        with rasterio.open(target_path) as src:
            if self.split == "train":
                geom = self.original_chips.loc[(self.original_chips["group"] == group) &
                                               (self.original_chips['event'] == event)][0].geometry
                rand_point = self.random_point_in_polygon(geom, self.input_size)
                window, rh, rw = self.point_to_window(rand_point, self.input_size, self.input_size, src)
            else:
                rh, rw = src.height, src.width
                window = Window(col_off=0, row_off=0, height=rh, width=rw)
            window_bounds = bounds(window, src.transform)
            target = src.read(window=window,out_shape=(src.count, rh, rw))
            no_data = np.where(target == -1, 0.0, 1.0)            
        sar = self._open_sar(event, group, window_bounds, rh, rw)
        
        target = torch.tensor(target).long()
        target[target == -1] = 0
        target = torch.nn.functional.one_hot(target.squeeze(0), num_classes=self.num_classes)
        target = target.permute(2, 0, 1).float()

        no_data = torch.tensor(no_data).float()
        
        sar = (sar - self.data_means) / self.data_stdvs

        return sar, target, no_data
                        
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

    @staticmethod
    def random_point_in_polygon(polygon: Polygon, input_size) -> Point:
        polygon = polygon.buffer(-(input_size // 2)*GSD)
        minx, miny, maxx, maxy = polygon.bounds
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        while not polygon.contains(Point(x, y)):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
        return Point(x, y)

    @staticmethod
    def point_to_window(point: Point, size_h: int, size_w: int, src: rasterio.io.DatasetReader):
        col, row = ~src.transform * (point.x, point.y)
        row, col = int(row), int(col)

        row_off = max(0, row - size_h // 2)
        col_off = max(0, col - size_w // 2)
        
        crop_h = min(size_h, src.height-row_off)
        crop_w = min(size_w, src.width-col_off)
        
        window = Window(col_off=col_off, row_off=row_off, height=crop_h, width=crop_w)
        return window, crop_h, crop_w