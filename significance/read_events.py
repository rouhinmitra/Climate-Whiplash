import os
import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import geopandas as gpd
from shapely.geometry import box
import rioxarray as rio
from shapely.geometry import box


def create_time_component(ds):
    # 1. Extract the start date string from the metadata
    start_date_str = ds['time'].attrs['units'].split(' since ')[1]
    
    # Convert the string to a proper datetime object** (This is the fix)
    start_date_obj = pd.to_datetime(start_date_str)
    
    # add the DateOffset to the datetime object
    dates = [start_date_obj + pd.DateOffset(months=int(m)) for m in ds['time'].values]
    
    # Assign the new datetime objects back to the coordinate
    ds['time'] = dates

    return ds

def load_and_process_data(dir_path1, dir_path2):
    df1 = xr.open_dataset(dir_path1, decode_times = False)
    df2 = xr.open_dataset(dir_path2, decode_times = False)
    df = df1 + df2
    df = create_time_component(df)
    return df

def mask_to_western_us(xarray_data, boundary_geom):
    """
    Mask xarray data to the western US boundary geometry.
    
    Parameters:
    -----------
    xarray_data : xarray.Dataset or xarray.DataArray
        The xarray data to be masked
    boundary_geom : geopandas.GeoDataFrame or shapely geometry
        The boundary geometry to mask to
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        The masked xarray data
    """
    xarray_data = xarray_data.rio.write_crs("epsg:4326")

    if hasattr(boundary_geom, 'geometry'):
        # If it's a GeoDataFrame, extract the geometry values
        geometry_values = boundary_geom.geometry.values
    else:
        # If it's already a geometry, use it directly
        geometry_values = boundary_geom
    
    # Set spatial dimensions if they're not already set
    # Check if spatial dimensions are set, if not, set them
    try:
        # Try to access x_dim to see if spatial dimensions are set
        _ = xarray_data.rio.x_dim
        # If we get here, spatial dimensions are already set
    except:
        # Spatial dimensions are not set, so set them based on common coordinate names
        if 'lon' in xarray_data.dims and 'lat' in xarray_data.dims:
            xarray_data = xarray_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
        elif 'longitude' in xarray_data.dims and 'latitude' in xarray_data.dims:
            xarray_data = xarray_data.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
    
    # Check if longitude coordinates need to be converted from 0-360 to -180-180
    if xarray_data.lon.max().values > 180:
        # Convert longitude coordinates from 0-360 to -180-180
        xarray_data = xarray_data.assign_coords(lon=((xarray_data.lon + 180) % 360) - 180)
        # Sort the data by longitude to fix plotting issues
        xarray_data = xarray_data.sortby('lon')
        # Re-set spatial dimensions after coordinate conversion
        if 'lon' in xarray_data.dims and 'lat' in xarray_data.dims:
            xarray_data = xarray_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    
    # Try to clip the data to the boundary
    try:
        masked_data = xarray_data.rio.clip(
            geometry_values, 
            crs=xarray_data.rio.crs, 
            all_touched=True, 
            drop=True
        )
        return masked_data
    except Exception as e:
        print(f"Warning: Could not clip data to boundary ({e}). Returning original data.")
        return xarray_data

# df = create_time_component(df)
# df = load_and_process_data(dir_path)
# print(df.isel(time=10).whiplash_event_frequency.plot())
# plt.show()

if __name__ == "__main__":

    dir_path1 = "/Users/rouhinmitra/Projects/Whiplash/Data/SPEI12_wet-dry_ERA5_whiplash_event_frequency.nc"
    dir_path2 = "/Users/rouhinmitra/Projects/Whiplash/Data/SPEI12_dry-wet_ERA5_whiplash_event_frequency.nc"
    us= gpd.read_file('/Users/rouhinmitra/Projects/Whiplash/Data/cb_2018_us_nation_5m/cb_2018_us_nation_5m.shp')
    print(us.crs)
    western_bounds = box(-180, 24, -100, 50)
    # Clip the US shape to get only the western portion
    western_us_geom = gpd.clip(us, western_bounds)
    western_us_geom = western_us_geom.to_crs(epsg=4326)
    df = load_and_process_data(dir_path1, dir_path2)
    df = mask_to_western_us(df, western_us_geom)
    print(df.isel(time=10).whiplash_event_frequency.plot())
    plt.show()
    print(plt.plot(df.time.values, df.mean(dim = ['lat', 'lon']).whiplash_event_frequency.values))
    plt.show()
    df.isel(time=10).whiplash_event_frequency.plot()
    plt.show()
    print(df.time.values)