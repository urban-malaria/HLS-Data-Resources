import os
import time
import numpy as np
import dask.distributed
import pystac_client
import geopandas as gpd
import odc.stac
import xarray as xr
import rasterio as rio
import rioxarray as rxr
import earthaccess
import pandas as pd

def create_quality_mask(quality_data, bit_nums: list = [1, 2, 3, 4, 5]):
    mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]))
    quality_data = np.nan_to_num(quality_data.copy(), nan=255).astype(np.int8)
    for bit in bit_nums:
        mask_temp = np.array(quality_data) & 1 << bit > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array

def main():
    start_time = time.time()

    # Authenticate with NASA Earthdata
    earthaccess.login(persist=True)

    # Start Dask client with memory limits
    client = dask.distributed.Client(n_workers=2, threads_per_worker=2, memory_limit='2GB')
    print(client)

    # Configure cloud access for STAC items
    odc.stac.configure_rio(cloud_defaults=True,
                           verbose=True,
                           client=client,
                           GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                           GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                           GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'))

    # Load boundary shapefile
    roi_all = gpd.read_file("../../data/Boundary_VaccStates_Export/Boundary_VaccStates_Export.shp")
    index = 0
    roi = roi_all.iloc[[index]]
    print("State Processing:", roi.iloc[0]['StateName'])

    # Search STAC catalog
    catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
    search_params = {
        "collections": ["HLSS30_2.0", "HLSL30_2.0"],
        "bbox": tuple(list(roi.total_bounds)),
        "datetime": "2015-08-01/2015-08-31",
        "limit": 100,
    }
    query = catalog.search(**search_params)
    items = list(query.items())
    print(f"Found {len(items)} items")

    # Harmonize band names
    for item in items:
        if "HLS.L30" in item.id:
            item.assets["NIR"] = item.assets.pop("B05")
        if "HLS.S30" in item.id:
            item.assets["NIR"] = item.assets.pop("B8A")

    # Load datasets lazily with chunks
    ds = odc.stac.stac_load(
        items,
        bands=("B02", "B04", "NIR", "Fmask"),
        crs="utm",
        resolution=30,
        chunks={"x": 512, "y": 512}
    )

    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326", inplace=False)
    ds = ds.chunk({"x": 512, "y": 512})
    ds = ds.rio.clip(roi.geometry.values, roi.crs, all_touched=True)

    # Reflectance scaling (Dask-safe)
    ds["NIR"] = ds.NIR * 0.0001
    ds["B04"] = ds.B04 * 0.0001
    ds["B02"] = ds.B02 * 0.0001

    # Calculate EVI lazily
    evi_ds = 2.5 * ((ds.NIR - ds.B04) / (ds.NIR + 6.0 * ds.B04 - 7.5 * ds.B02 + 1.0))

    #Quality mask updated to single chunk - added 
    if 'x' in ds.dims and 'y' in ds.dims:
        try:
            ds = ds.rechunk({'x': -1, 'y': -1})
            print("Rechunked ds to a single chunk across x and y.")
        except Exception as e:
            print("Rechunking failed:", e)
            return
    
    # Apply cloud/shadow mask -
    quality_mask = xr.apply_ufunc(
        create_quality_mask,
        ds.Fmask,
        kwargs={"bit_nums": [1, 2, 3, 4, 5]},
        input_core_dims=[["x", "y"]],
        output_core_dims=[["x", "y"]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.bool_],
    )

  

    evi_ds = evi_ds.where(~quality_mask)

    # Persist result to avoid recomputation
    try:
        evi_ds = evi_ds.persist()
    except Exception as e:
        print("Dask persist failed:", e)
        return

    # Monthly aggregation
    monthly_evi = evi_ds.groupby("time.month").mean(dim="time", skipna=True)
    monthly_evi.name = "EVI"

    monthly_evi.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    monthly_evi.rio.write_crs("EPSG:4326", inplace=True)

    output_dir = "2015_evi_rasters_NGA_new"
    os.makedirs(output_dir, exist_ok=True)

    for month in monthly_evi.month.values:
        evi_month = monthly_evi.sel(month=month)
        output_path = os.path.join(output_dir, f"EVI_month_{month:02d}.tif")
        evi_month.rio.to_raster(output_path)
        print(f" Exported: {output_path}")

    end_time = time.time()
    print(f" Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
