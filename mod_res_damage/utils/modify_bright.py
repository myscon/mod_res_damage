import argparse
import asf_search as search
import datetime
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import shapely
import time
import yaml

from asf_search import constants
from concurrent.futures import ThreadPoolExecutor, as_completed
from hyp3_sdk import HyP3
from hyp3_sdk.util import extract_zipped_product
from pathlib import Path
from pyproj import Transformer
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from shapely.ops import transform
from sklearn.cluster import DBSCAN
from typing import Tuple

# NOTE: annoying timeout errors when calling geo_search
search.constants.INTERNAL.CMR_TIMEOUT = 60

CLUSTER_CRS = "EPSG:3857"
DEFAULT_CRS = "EPSG:4326"
SEARCH_CRS = "EPSG:4326"


def get_centroid_meters(src: rasterio.io.DatasetReader) -> tuple:
    transformer = Transformer.from_crs(src.crs, CLUSTER_CRS, always_xy=True)
    lon = (src.bounds.left + src.bounds.right) / 2
    lat = (src.bounds.top + src.bounds.bottom) / 2
    x, y = transformer.transform(lon, lat)
    return x, y

  
def get_geotiff_bounds_as_gdf(files: list[Path], group: int, event: str, dst_crs: str) -> gpd.GeoDataFrame:
    records = []
    for file in files:
        with rasterio.open(file) as src:
            bounds = src.bounds
            crs = src.crs
            if crs is None:
                raise ValueError(f"File {file} has no CRS defined.")
            geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            if src.crs.to_string() != dst_crs:
                transformer = Transformer.from_crs(src.crs, dst_crs, always_xy=True)
                geom = transform(transformer.transform, geom)

            records.append({
                "file": file.name,
                "geometry": geom,
                "group": group,
                "event": event
            })
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=dst_crs)
    return gdf


def cluster_rasters_by_proximity(files: list[Path], distance_m: int = 8192) -> list:
    centroids = []
    valid_files = []

    for f in files:
        try:
            with rasterio.open(f) as src:
                centroid = get_centroid_meters(src)
                centroids.append(centroid)
                valid_files.append(f)
        except Exception as e:
            raise ValueError(f"Error processing file {f}: {e}")

    if not centroids:
        return []
    coords = np.array(centroids)
    
    # NOTE: turns out all the groups are about 4km apart and really only affects the turkiye earthquake 
    db = DBSCAN(eps=distance_m, min_samples=2, metric='euclidean')
    labels = db.fit_predict(coords)

    groups = {}
    for event, file in zip(labels, valid_files):
        groups.setdefault(event, []).append(Path(file))
    return list(groups.values())
  

def reproject_in_mem(src_arr: np.ndarray,
                     bounds: Tuple[float, float, float, float],
                     profile: dict,
                     src_transform: rasterio.Affine,
                     src_crs: rasterio.crs.CRS,
                     dst_crs: rasterio.crs.CRS,
                     method: Resampling = Resampling.nearest) -> rasterio.io.MemoryFile:
    dst_transform, height, width = calculate_default_transform(src_crs,
                                                               DEFAULT_CRS,
                                                               src_arr.shape[1],
                                                               src_arr.shape[2],
                                                               *bounds)
    count, height, width = src_arr.shape
    profile.update({
        'count': count,
        'crs': dst_crs,
        'transform': dst_transform,
        'width': width,
        'height': height
    })
    dst_array = np.empty((count, height, width), dtype=src_arr.dtype)

    for i in range(count):
        reproject(
            source=src_arr[i],
            destination=dst_array[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=method
        )

    return dst_array, profile

  
def mosaic_to_default_crs(input_files: list[Path], output_path: Path):
    files = []
    crses = []
    for f in input_files:
        fsrc = rasterio.open(f)
        files.append(fsrc)
        crses.append(fsrc.crs)
    
    assert all(x == crses[0] for x in crses)

    mosaic, src_transform = merge(files)
    bounds = array_bounds(mosaic.shape[1],
                          mosaic.shape[2],
                          src_transform)
    mosaic, profile = reproject_in_mem(mosaic,
                                        bounds,
                                        files[0].meta.copy(),
                                        src_transform,
                                        files[0].crs,
                                        DEFAULT_CRS)
    with rasterio.open(output_path, "w", **profile) as dest:
        dest.write(mosaic)
        rprojected_bounds = dest.bounds
    for f in files:
        f.close()

    rprojected_bounds = box(*rprojected_bounds)
    return rprojected_bounds


def reproject_clip_raster(input_tif: Path, output_tif: Path, roi: shapely.geometry) -> None:
    output_tif.parent.exists()
    if not output_tif.parent.exists():
        output_tif.parent.mkdir(exist_ok=True)
        
    with rasterio.open(input_tif) as src:
        if src.crs.to_string() != DEFAULT_CRS:
            reprojected, profile = reproject_in_mem(src.read(),
                                           src.bounds,
                                           src.meta.copy(),
                                           src.transform,
                                           src.crs,
                                           DEFAULT_CRS,
                                           Resampling.cubic)
        else:
            reprojected, profile = src.read(), src.meta
        
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=reprojected.shape[1],
                width=reprojected.shape[2],
                count=reprojected.shape[0],
                dtype=reprojected.dtype,
                transform=profile['transform'],
                crs=DEFAULT_CRS
            ) as dataset:
                dataset.write(reprojected)
                out_image, out_transform = mask(
                    dataset=dataset,
                    shapes=[roi],
                    crop=True,
                    filled=False,
                )
        profile.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform,
            'crs': DEFAULT_CRS,
        })

        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(out_image)


def sort_products(product: search.ASFProduct, roi: shapely.geometry.Polygon) -> tuple:
    footprint = shapely.geometry.shape(product.geometry)
    intersection = int(100 * roi.intersection(footprint).area / roi.area) * -1
    date = product.properties['startTime']
    return intersection, date
  

def filter_products(scene_name: str, hyp3: HyP3) -> list[search.ASFProduct]:
    jobs = [j for j in hyp3.find_jobs(job_type='RTC_GAMMA') if not j.failed() and not j.expired()]
    jobs = [j for j in jobs if j.job_parameters['granules'] == [scene_name]]
    jobs = [j for j in jobs if j.job_parameters['radiometry'] == 'gamma0']
    jobs = [j for j in jobs if j.job_parameters['resolution'] == 10]
    
    if len(jobs) == 1:
        job = jobs[0]
        return job
    elif len(jobs) == 0:
        return None
    else:
        raise ValueError(f"Multiple existing jobs found for {scene_name} {jobs}.")


def get_hyp3_rtc(scene_name: str, scratch_dir: Path) -> tuple[Path, Path]:
    hyp3 = HyP3()
    job = filter_products(scene_name, hyp3)
    if job is None:
        print(f"\nNo existing job found for {scene_name}. Submitting new job.")
        job = hyp3.submit_rtc_job(scene_name, radiometry='gamma0', resolution=10)
        
    if not job.succeeded():
        print(f"\nWaiting for job\n{job}\n{scene_name} to complete...")
        hyp3.watch(job, timeout=14400)
        time.sleep(5)

    output_path = scratch_dir / job.to_dict()['files'][0]['filename']
    output_dir = output_path.with_suffix('')
    output_zip = output_path.with_suffix('.zip')

    if not output_dir.exists():
        print(f"\nDownloading files at {output_dir}")
        job.download_files(location=scratch_dir)
        extract_zipped_product(output_zip)
    else:
        print(f"\n{output_dir} already exists. Skipping") 
    vv_file = list(output_dir.glob('*_VV.tif'))[0]
    vh_file = list(output_dir.glob('*_VH.tif'))[0]
    return vv_file, vh_file


def process_group_radar(roi: shapely.geometry.Polygon, 
                        event: str, 
                        group: int, 
                        start_date: datetime, 
                        end_date: datetime, 
                        output_dir: Path, 
                        scratch_dir: Path,
                        days_before_after: int =  12):
    date_ranges = {"before": (start_date - datetime.timedelta(days=days_before_after), start_date), 
                   "after": (end_date, end_date + datetime.timedelta(days=days_before_after))}
    transformer = Transformer.from_crs(DEFAULT_CRS, SEARCH_CRS, always_xy=True)
    search_roi = transform(transformer.transform, roi)
    
    for key, (date_start, date_end) in date_ranges.items():
        vv_path = output_dir / key / f"{event}_group-{group}_VV.tif"
        vh_path = output_dir / key / f"{event}_group-{group}_VH.tif"
        
        # NOTE: Job submission would probably work much better async but eh
        if not vv_path.exists() or not vh_path.exists():
            search_results = search.geo_search(
                intersectsWith=search_roi.wkt,
                start=date_start,
                end=date_end,
                beamMode=constants.BEAMMODE.IW,
                polarization=constants.POLARIZATION.VV_VH,
                platform=constants.PLATFORM.SENTINEL1,
                processingLevel=constants.PRODUCT_TYPE.SLC,
            )
            if len(search_results) == 0:
                raise ValueError(f'No products found for {event} chip {roi} ({search_roi}) between {date_start} {date_end}')
            
            product = sorted(list(search_results), key=lambda x: sort_products(x, search_roi))[0]
            scene_name = product.properties['sceneName']
            
            vv_file, vh_file = get_hyp3_rtc(scene_name, scratch_dir)
            date_end = date_end.strftime("%Y-%m-%d")
            date_start = date_start.strftime("%Y-%m-%d")
            
            reproject_clip_raster(vv_file, vv_path, roi)
            reproject_clip_raster(vh_file, vh_path, roi)



def resample_to_mod_res(out_dir: Path, hr_label_path: Path, mod_res_path: Path, geometries: gpd.GeoSeries):
    no_data = -1
    with rasterio.open(mod_res_path) as mod_res:
        mod_transform = mod_res.transform
        mod_crs = mod_res.crs
        mod_width = mod_res.width
        mod_height = mod_res.height

    with rasterio.open(hr_label_path) as high_res:
        high_res_data = high_res.read()
        high_res_data, high_res_transform = mask(
                    dataset=high_res,
                    shapes=geometries,
                    nodata=no_data,
                    crop=False,
                )
        dst_array = np.full((high_res.count, mod_height, mod_width), fill_value=no_data, dtype=np.int8)

        # NOTE: two reprojections took place to get aoi then to get to moderate resolution this will introduce some subtle errors
        reproject(
            source=high_res_data.astype(np.int8),
            destination=dst_array,
            src_transform=high_res_transform,
            src_crs=high_res.crs,
            src_nodata=no_data,
            dst_transform=mod_transform,
            dst_crs=mod_crs,
            dst_nodata=no_data,
            resampling=Resampling.nearest
        )

        profile = high_res.profile
        profile.update({
            'height': mod_height,
            'width': mod_width,
            'transform': mod_transform,
            'crs': mod_crs,
            'dtype': np.int8
        })

    with rasterio.open(out_dir / hr_label_path.name, 'w', **profile) as dst:
        dst.write(dst_array)


def process_group(event: str,
                  dates: datetime,
                  group_num: int,
                  group: list[Path],
                  hr_label_dir: Path,
                  label_dir: Path,
                  radar_dir: Path,
                  scratch_dir: Path):
    print(f"\nhanding {event} group {group_num} with {len(group)} items")
    dates = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    
    # NOTE: might be useful to rename the group based on location but I am lazy.
    gdf = get_geotiff_bounds_as_gdf(group, group_num, event, DEFAULT_CRS)

    vv_before_path = radar_dir / "before" /f"{event}_group-{group_num}_VV.tif"
    hr_label_path = hr_label_dir / f"{event}_group-{group_num}.tif"
    roi = mosaic_to_default_crs(group, hr_label_path)

    if len(dates) == 1:
        start_date = dates[0]
        end_date = start_date + datetime.timedelta(days=1)
    elif len(dates) == 2:
        start_date, end_date = dates
    else:
        raise ValueError(f"Invalid date format for event {event}: {dates}")
    process_group_radar(roi, event, group_num, start_date, end_date, radar_dir, scratch_dir=scratch_dir)
    resample_to_mod_res(label_dir, hr_label_path, vv_before_path, gdf.geometry)
        
    return gdf




def main(search_dir, output_dir, date_path, scratch_dir, max_workers):
    event_dates = yaml.safe_load(open(date_path, 'r'))
    hr_label_dir = output_dir / "labels_high_res_merged"
    label_dir = output_dir / "labels"
    radar_dir = output_dir / "radar"
    
    hr_label_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    radar_dir.mkdir(exist_ok=True)
    
    tasks = []
    gdfs = []

    if max_workers > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for event, dates in event_dates.items():
                matched_files = list(search_dir.glob(f"*{event}*"))
                if matched_files:
                    groups = cluster_rasters_by_proximity(matched_files)
                    for i, group in enumerate(groups):
                        tasks.append(
                            executor.submit(
                                process_group,
                                event, dates, i, group, hr_label_dir, label_dir, radar_dir, scratch_dir
                            )
                        )

            for future in as_completed(tasks):
                try:
                    result = future.result()
                    if result is not None:
                        gdfs.append(result)
                except Exception as e:
                    print(f"\nError in task: {e}")
    else:
        for event, dates in event_dates.items():
            matched_files = list(search_dir.glob(f"*{event}*"))
            if matched_files:
                groups = cluster_rasters_by_proximity(matched_files)
                for i, group in enumerate(groups):
                    gdfs.append(process_group(event, dates, i, group, hr_label_dir, label_dir, radar_dir, scratch_dir))

    if gdfs:
        combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry")
        combined.to_file(output_dir / "original_chips.geojson", driver="GeoJSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process labeled events in a directory.")
    parser.add_argument(
        "--search_dir",
        type=str,
        default="./data/mbright/original_targets",
        help="Directory to search for matching files (default: ./data/mbright/original_targets)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/mbright",
        help="Directory to search for matching files (default: ./data/mbright/)"
    )
    parser.add_argument(
        "--scratch_dir",
        type=str,
        default="./data/mbright/raw_rtc",
        help="Directory to search for matching files (default: ./data/mbright/raw_rtc)"
    )
    parser.add_argument(
        "--date_path",
        type=str,
        default="./data/mbright/bright_dates.yaml",
        help="Path to the yaml file containing event dates (default: ./data/mbright/bright_dates.yaml)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=-1,
        help="Max workers to do parallel processings. Defaults to 4 workers but should be noted that merging of the many tifs opens many files which may cause some i/o issues."
    )
    args = parser.parse_args()
    main(Path(args.search_dir), Path(args.output_dir), args.date_path, Path(args.scratch_dir), args.max_workers)