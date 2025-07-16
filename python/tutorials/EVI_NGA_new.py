#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
start_time = time.time()


# In[ ]:


# Load all the libraries
import os
import numpy as np
import dask.distributed
import pystac_client
import geopandas as gpd
import odc.stac
import xarray as xr
import rasterio as rio
import rioxarray as rxr
import earthaccess
import hvplot.xarray


# In[ ]:


# Log into earthaccess - ensures creation of .netrc file
earthaccess.login(persist=True)


# In[ ]:


# Initialize Dask Client
client = dask.distributed.Client()
print(client)


# In[ ]:


# Configure odc.stac rio env - requires a .netrc file, sends info to dask client
odc.stac.configure_rio(cloud_defaults=True,
                       verbose=True,
                       client=client,
                       GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                       GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                       GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'))


# In[ ]:


# Open ROI polygon
roi = gpd.read_file("../../data/Boundary_VaccStates_Export/Boundary_VaccStates_Export.shp")


# In[ ]:


catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
# Define search parameters
search_params = {
    "collections": ["HLSS30_2.0","HLSL30_2.0"],
    "bbox": tuple(list(roi.total_bounds)),
    "datetime": "2015-08-01/2015-08-31", #for 1 month in 2015
    "limit": 100,
}
# Perform the search
query = catalog.search(**search_params)
items = query.items()


# In[ ]:


items = list(query.items())
print(f"Found: {len(items):d} granules")


# In[ ]:


items[0]


# In[ ]:


# Rename HLSS B8A and HLSL B05 to common band name
for item in items:
    if "HLS.L30" in item.id:
        item.assets["NIR"] = item.assets.pop("B05")
    if "HLS.S30" in item.id:
        item.assets["NIR"] = item.assets.pop("B8A")


# In[ ]:


# Confirm this changed the stac results
items[0]


# In[ ]:


# Set CRS and resolution, open lazily with odc.stac
crs = "utm"
ds = odc.stac.stac_load(
    items,
    bands=("B02", "B04","NIR", "Fmask"),
    crs=crs,
    resolution=30,
    chunks={"band":1,"x":512,"y":512},  # If empty, chunks along band dim, 
    #groupby="solar_day", # This limits to first obs per day
)
print(ds)


# In[ ]:


# Show Geobox
ds.odc.geobox


# In[ ]:


# Clip
ds = ds.rio.clip(roi.geometry.values, roi.crs, all_touched=True)


# In[ ]:


# Show Clipped Geobox
ds.odc.geobox


# In[ ]:


# Scale the data
ds.NIR.data = 0.0001 * ds.NIR.data
ds.B04.data = 0.0001 * ds.B04.data
ds.B02.data = 0.0001 * ds.B02.data


# In[ ]:


ds.load()


# In[ ]:


# Plot to ensure scaling worked
ds.NIR.hvplot.image(x="x", y="y", groupby="time", cmap="viridis", width=600, height=500, crs='EPSG:4326', tiles='ESRI', rasterize=True)

#changed EPSG from 32610 to 4326


# In[ ]:


# Calculate EVI
evi_ds = 2.5 * ((ds.NIR - ds.B04) / (ds.NIR + 6.0 * ds.B04 - 7.5 * ds.B02 + 1.0))


# In[ ]:


evi_ds = evi_ds.compute()


# In[ ]:


evi_ds.hvplot.image(x="x", y="y", groupby="time", cmap="YlGn", clim=(0, 1), crs='EPSG:4326', tiles='ESRI', rasterize=True)


# In[ ]:


def create_quality_mask(quality_data, bit_nums: list = [1, 2, 3, 4, 5]):
    """
    Uses the Fmask layer and bit numbers to create a binary mask of good pixels.
    By default, bits 1-5 are used.
    """
    mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]))
    # Remove/Mask Fill Values and Convert to Integer
    quality_data = np.nan_to_num(quality_data.copy(), nan=255).astype(np.int8)
    for bit in bit_nums:
        # Create a Single Binary Mask Layer
        mask_temp = np.array(quality_data) & 1 << bit > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array


# In[ ]:


quality_mask = xr.apply_ufunc(
    create_quality_mask,
    ds.Fmask,
    kwargs={"bit_nums": [1,2,3,4,5]},
    input_core_dims=[["x", "y"]],
    output_core_dims=[["x", "y"]],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[np.bool],
)


# In[ ]:


evi_ds.where(~quality_mask).hvplot.image(x="x", y="y", groupby="time", cmap="YlGn", clim=(0, 1), crs='EPSG:4326', tiles='ESRI', rasterize=True)


# In[ ]:


#Convert to Raster

import xarray as xr
import pandas as pd
import os

# Step 1: Group by month and compute monthly mean
monthly_evi = evi_ds.groupby("time.month").mean(dim="time", skipna=True)

# Optional: Give the DataArray a name for export
monthly_evi.name = "EVI"

# Set spatial dimensions and CRS if not already set
monthly_evi.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
monthly_evi.rio.write_crs("EPSG:4326", inplace=True)  # adjust if your CRS is different

# Step 2: Export each month as a separate raster
output_dir = "2015_evi_rasters_NGA_new"
os.makedirs(output_dir, exist_ok=True)

for month in monthly_evi.month.values:
    evi_month = monthly_evi.sel(month=month)
    output_path = os.path.join(output_dir, f"EVI_month_{month:02d}.tif")
    evi_month.rio.to_raster(output_path)
    print(f"Exported: {output_path}")


# In[ ]:


# In[ ]:


#  Timing

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

