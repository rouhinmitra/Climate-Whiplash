from read_events import load_and_process_data, mask_to_western_us
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import geopandas as gpd
from shapely.geometry import box
import rioxarray as rio
from shapely.geometry import box
from scipy import stats
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')


def sens_slope(data):
    """
    Calculate Sen's slope for trend magnitude estimation.
    Traditional method using median of all slopes.
    
    Parameters:
    -----------
    data : array-like
        Time series data
        
    Returns:
    --------
    float
        Sen's slope estimate
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 2:
        return np.nan
    
    n = len(clean_data)
    slopes = []
    
    # Calculate slopes between all pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            slope = (clean_data[j] - clean_data[i]) / (j - i)
            slopes.append(slope)
    
    # Return the median slope (traditional Sen's slope method)
    return np.median(slopes)


def compute_pairwise_slopes(data):
    """
    Compute all pairwise slopes (x_j - x_i) / (j - i) used by Sen's slope.
    NaNs are removed before computation.
    """
    clean_data = data[~np.isnan(data)]
    n = len(clean_data)
    if n < 2:
        return np.array([])
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((clean_data[j] - clean_data[i]) / (j - i))
    return np.array(slopes)


def mann_kendall_test(data, alpha=0.05):
    """
    Perform Mann-Kendall test for trend detection.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - tau: Kendall's tau statistic
        - p_value: p-value of the test
        - trend: 'increasing', 'decreasing', or 'no trend'
        - significant: boolean indicating if trend is significant
        - sens_slope: Sen's slope estimate
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 3:
        return {
            'tau': np.nan,
            'p_value': np.nan,
            'trend': 'insufficient_data',
            'significant': False,
            'sens_slope': np.nan
        }
    
    # Calculate Kendall's tau and p-value
    tau, p_value = kendalltau(range(len(clean_data)), clean_data)
    
    # Calculate Sen's slope
    slope = sens_slope(clean_data)
    
    # Determine trend direction and significance
    if p_value < alpha:
        if tau > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        significant = True
    else:
        trend = 'no_trend'
        significant = False
    
    return {
        'tau': tau,
        'p_value': p_value,
        'trend': trend,
        'significant': significant,
        'sens_slope': slope
    }


def grid_wise_mann_kendall(data_array, alpha=0.05):
    """
    Apply Mann-Kendall test to each grid cell in an xarray DataArray.
    
    Parameters:
    -----------
    data_array : xarray.DataArray
        DataArray with time dimension for trend analysis
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    xarray.Dataset
        Dataset containing:
        - tau: Kendall's tau for each grid cell
        - p_value: p-values for each grid cell
        - trend: trend direction for each grid cell
        - significant: significance mask
        - sens_slope: Sen's slope for each grid cell
    """
    # Get dimensions
    spatial_dims = [dim for dim in data_array.dims if dim != 'time']
    
    # Initialize result arrays
    tau_array = np.full([data_array.sizes[dim] for dim in spatial_dims], np.nan)
    p_value_array = np.full([data_array.sizes[dim] for dim in spatial_dims], np.nan)
    trend_array = np.full([data_array.sizes[dim] for dim in spatial_dims], 'no_trend', dtype=object)
    significant_array = np.full([data_array.sizes[dim] for dim in spatial_dims], False)
    sens_slope_array = np.full([data_array.sizes[dim] for dim in spatial_dims], np.nan)
    
    # Apply Mann-Kendall test to each grid cell
    for i in range(data_array.sizes[spatial_dims[0]]):
        for j in range(data_array.sizes[spatial_dims[1]]):
            # Extract time series for this grid cell
            if len(spatial_dims) == 2:
                time_series = data_array.isel({spatial_dims[0]: i, spatial_dims[1]: j}).values
            else:
                time_series = data_array.isel({spatial_dims[0]: i}).values
            
            # Perform Mann-Kendall test
            result = mann_kendall_test(time_series, alpha)
            
            # Store results
            tau_array[i, j] = result['tau']
            p_value_array[i, j] = result['p_value']
            trend_array[i, j] = result['trend']
            significant_array[i, j] = result['significant']
            sens_slope_array[i, j] = result['sens_slope']
            
            # Debug: Print first few calculations to see what's happening
            if i < 3 and j < 3:
                print(f"Grid cell [{i},{j}]: time_series length={len(time_series)}, "
                      f"tau={result['tau']:.6f}, slope={result['sens_slope']:.8f}")
    
    # Create xarray Dataset with results
    coords = {dim: data_array[dim] for dim in spatial_dims}
    
    results = xr.Dataset({
        'tau': (spatial_dims, tau_array),
        'p_value': (spatial_dims, p_value_array),
        'trend': (spatial_dims, trend_array),
        'significant': (spatial_dims, significant_array),
        'sens_slope': (spatial_dims, sens_slope_array)
    }, coords=coords)
    
    return results


dir_path1 = "/Users/rouhinmitra/Projects/Whiplash/Data/SPEI12_wet-dry_ERA5_whiplash_event_frequency.nc"
dir_path2 = "/Users/rouhinmitra/Projects/Whiplash/Data/SPEI12_dry-wet_ERA5_whiplash_event_frequency.nc"
us= gpd.read_file('/Users/rouhinmitra/Projects/Whiplash/Data/cb_2018_us_nation_5m/cb_2018_us_nation_5m.shp')
print(us.crs)
us = us.to_crs(epsg=4326)
print(us)
western_bounds = box(-180, 24, -100, 50)
# Clip the US shape to get only the western portion
western_us_geom = gpd.clip(us, western_bounds)
western_us_geom = western_us_geom.to_crs(epsg=4326)
# Read the nc file
df = load_and_process_data(dir_path1, dir_path2)
df = mask_to_western_us(df, western_us_geom)

# Perform grid-wise Mann-Kendall test
print("Performing grid-wise Mann-Kendall test...")
mk_results = grid_wise_mann_kendall(df.whiplash_event_frequency, alpha=0.05)

# Use original Sen's slope values (no per-decade conversion)
print(f"Sen's slope range (original units): {mk_results.sens_slope.min().values:.8f} to {mk_results.sens_slope.max().values:.8f}")

# Check for significant trends only
significant_slopes = mk_results.sens_slope.where(mk_results.significant)
sig_min = significant_slopes.min().values
sig_max = significant_slopes.max().values
print(f"Significant slopes range: {sig_min:.8f} to {sig_max:.8f}")

# Calculate dynamic color scaling for Sen's slope
sens_slope_range = max(abs(mk_results.sens_slope.min().values), abs(mk_results.sens_slope.max().values))

# Use appropriate scaling for the actual data range
if sens_slope_range < 0.001:
    print("Using fixed range for visualization")
    sens_slope_vmin = -0.01
    sens_slope_vmax = 0.01
else:
    sens_slope_vmin = -sens_slope_range * 1.2
    sens_slope_vmax = sens_slope_range * 1.2

print(f"Using color scale: {sens_slope_vmin:.6f} to {sens_slope_vmax:.6f}")

# Data inspection
print(f"\nData inspection:")
print(f"Number of non-NaN values: {(~np.isnan(mk_results.sens_slope)).sum().values}")
print(f"Number of significant trends: {mk_results.significant.sum().values}")

# Calculate valid land cells (exclude ocean/NaN areas)
valid_land_cells = (~np.isnan(mk_results.sens_slope)).sum().values
print(f"Valid land cells (excluding ocean/NaN): {valid_land_cells}")
print(f"Percentage of significant trends: {(mk_results.significant.sum() / valid_land_cells * 100).values:.2f}%")

# Print summary statistics
print(f"\nMann-Kendall Test Results Summary:")
print(f"Total grid cells: {mk_results.sizes['lat'] * mk_results.sizes['lon']}")
print(f"Valid land cells: {valid_land_cells}")
print(f"Significant trends: {mk_results.significant.sum().values}")
print(f"Percentage with significant trends: {(mk_results.significant.sum() / valid_land_cells * 100).values:.2f}%")

# Count trend directions
increasing_trends = (mk_results.trend == 'increasing').sum().values
decreasing_trends = (mk_results.trend == 'decreasing').sum().values
no_trends = (mk_results.trend == 'no_trend').sum().values

print(f"Increasing trends: {increasing_trends}")
print(f"Decreasing trends: {decreasing_trends}")
print(f"No significant trends: {no_trends}")

# Create focused visualization with 5 key plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Whiplash Event Frequency Analysis - Key Results', fontsize=16, fontweight='bold')

# Plot 1: Total number of whiplash events for every grid cell
df.sum(dim='time').whiplash_event_frequency.plot(ax=axes[0,0], cmap='viridis', cbar_kwargs={'label': 'Total Events', 'shrink': 0.6})
axes[0,0].set_title('Total Whiplash Events\n(All Years)')
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('')

# Plot 2: Trend strength (Kendall's tau)
mk_results.tau.plot(ax=axes[0,1], cmap='RdBu_r', center=0, cbar_kwargs={'label': "Kendall's τ", 'shrink': 0.6})
axes[0,1].set_title("Trend Strength\n(Kendall's Tau)")
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('')

# Plot 3: P-values with viridis colormap and triangle tip
mk_results.p_value.plot(ax=axes[0,2], cmap='viridis', cbar_kwargs={'label': 'P-value', 'shrink': 0.6})

# Add triangle at bottom tip to highlight significant range
import matplotlib.patches as patches
triangle = patches.Polygon([(0, 0), (0.05, 0), (0.025, 0.1)], 
                          facecolor='red', edgecolor='black', linewidth=1, alpha=0.7)
axes[0,2].add_patch(triangle)
axes[0,2].set_title('Statistical Significance')
axes[0,2].set_xlabel('')
axes[0,2].set_ylabel('')

# Plot 4: Significant trends with categorical labels (using Kendall's tau for direction)
trend_categorical_sig = xr.where(mk_results.significant, 
                               xr.where(mk_results.tau > 0, 1, -1),  # 1 for increasing, -1 for decreasing
                               np.nan)  # Not significant
trend_categorical_sig.plot(ax=axes[1,0], cmap='RdBu_r', center=0, cbar_kwargs={'label': 'Trend Direction', 'shrink': 0.6})
axes[1,0].set_title("Significant Trends")
axes[1,0].set_xlabel('')
axes[1,0].set_ylabel('')

# Plot 5: Sen's slope (without decade conversion)
mk_results.sens_slope.plot(ax=axes[1,1], cmap='RdBu_r', center=0, cbar_kwargs={'label': "Sen's Slope (events/year)", 'shrink': 0.6})
axes[1,1].set_title("Sen's Slope\n(Events per Year)")
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('')

# Plot 6: Sen's slope for significant trends only
significant_sens_slope = xr.where(mk_results.significant, mk_results.sens_slope, np.nan)
significant_sens_slope.plot(ax=axes[1,2], cmap='RdBu_r', center=0, cbar_kwargs={'label': "Sen's Slope (events/year)", 'shrink': 0.6})
axes[1,2].set_title("Sen's Slope\n(Significant Trends Only)")
axes[1,2].set_xlabel('')
axes[1,2].set_ylabel('')

# Add boundary overlay to all plots
for i, ax in enumerate([axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]):
    western_us_geom.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

plt.tight_layout(pad=2.0)
plt.show()

# Plot histogram of pairwise slopes for a specific California coordinate
# Set a default California coordinate; adjust as needed
target_lat = 36.5
target_lon = -119.5

# Find nearest grid coordinate and extract the time series
nearest_lat = float(df.lat.sel(lat=target_lat, method='nearest').values)
nearest_lon = float(df.lon.sel(lon=target_lon, method='nearest').values)
series_at_point = df.whiplash_event_frequency.sel(lat=nearest_lat, lon=nearest_lon).values

pairwise_slopes = compute_pairwise_slopes(series_at_point)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
fig.suptitle("Distribution of Pairwise Slopes (x_j - x_i) / (j - i)", fontsize=16, fontweight='bold')

ax.hist(pairwise_slopes, bins=60, alpha=0.75, color='steelblue', edgecolor='black', linewidth=0.5)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero slope')
median_pairwise = np.median(pairwise_slopes) if pairwise_slopes.size > 0 else np.nan
ax.axvline(median_pairwise, color='darkorange', linestyle='-', linewidth=2, label=f'Median = {median_pairwise:.6f}')

ax.set_xlabel("Slope value")
ax.set_ylabel("Count of pairs")
ax.set_title(f"Pairwise Slopes at California Point (lat={nearest_lat:.3f}, lon={nearest_lon:.3f})")
ax.legend()
ax.grid(True, alpha=0.3)

# Annotation with basic stats
if pairwise_slopes.size > 0:
    textstr = (
        f'N pairs: {pairwise_slopes.size}\n'
        f'Mean: {np.mean(pairwise_slopes):.6f}\n'
        f'Std: {np.std(pairwise_slopes):.6f}\n'
        f'Median: {median_pairwise:.6f}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Create histogram of Sen's slope values
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
fig.suptitle('Distribution of Sen\'s Slope Values', fontsize=16, fontweight='bold')

# Flatten the Sen's slope data and remove NaN values
sens_slope_flat = mk_results.sens_slope.values.flatten()
sens_slope_clean = sens_slope_flat[~np.isnan(sens_slope_flat)]

# Create histogram
ax.hist(sens_slope_clean, bins=50, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero slope')
ax.set_xlabel("Sen's Slope (events/year)")
ax.set_ylabel('Number of Grid Cells')
ax.set_title('Distribution of Sen\'s Slope Values Across All Grid Cells')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics text
mean_slope = np.mean(sens_slope_clean)
std_slope = np.std(sens_slope_clean)
median_slope = np.median(sens_slope_clean)

textstr = f'Mean: {mean_slope:.6f}\nStd: {std_slope:.6f}\nMedian: {median_slope:.6f}\nTotal cells: {len(sens_slope_clean)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Create histogram of p-values
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
fig.suptitle('Distribution of P-values', fontsize=16, fontweight='bold')

# Flatten the p-value data and remove NaN values
p_value_flat = mk_results.p_value.values.flatten()
p_value_clean = p_value_flat[~np.isnan(p_value_flat)]

# Print diagnostic information
print(f"\nP-value data diagnostic:")
print(f"Total grid cells (lat x lon): {mk_results.sizes['lat'] * mk_results.sizes['lon']}")
print(f"Total p-value values: {len(p_value_flat)}")
print(f"Non-NaN p-values: {len(p_value_clean)}")
print(f"NaN p-values: {len(p_value_flat) - len(p_value_clean)}")
print(f"Percentage with valid p-values: {(len(p_value_clean) / len(p_value_flat)) * 100:.1f}%")

# Create histogram
ax.hist(p_value_clean, bins=50, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Significance threshold (p=0.05)')
ax.set_xlabel('P-value')
ax.set_ylabel('Number of Grid Cells')
ax.set_title('Distribution of P-values Across All Grid Cells')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics text
mean_p = np.mean(p_value_clean)
std_p = np.std(p_value_clean)
median_p = np.median(p_value_clean)
significant_count = np.sum(p_value_clean < 0.05)
total_count = len(p_value_clean)
significant_percent = (significant_count / total_count) * 100

textstr = f'Mean: {mean_p:.4f}\nStd: {std_p:.4f}\nMedian: {median_p:.4f}\n'
textstr += f'Significant (p<0.05): {significant_count}\n'
textstr += f'Percentage significant: {significant_percent:.1f}%\n'
textstr += f'Valid land cells: {total_count}\n'
textstr += f'Total grid cells: {mk_results.sizes["lat"] * mk_results.sizes["lon"]}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Create a final summary plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Use the categorical approach for significant trends (using Kendall's tau for direction)
trend_categorical_final = xr.where(mk_results.significant, 
                                 xr.where(mk_results.tau > 0, 1, -1),  # 1 for increasing, -1 for decreasing
                                 np.nan)  # Not significant
trend_categorical_final.plot(ax=ax, cmap='RdBu_r', center=0,
                            cbar_kwargs={'label': 'Trend Direction', 'shrink': 0.6})
western_us_geom.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
ax.set_title('Significant Trends in Whiplash Event Frequency\n(Mann-Kendall Test + Kendall\'s Tau Direction)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')

# Add text box with interpretation
valid_land_cells = (~np.isnan(mk_results.sens_slope)).sum().values
textstr = f'Valid Land Cells: {valid_land_cells}\n'
textstr += f'Significant Trends: {mk_results.significant.sum().values}\n'
textstr += f'Increasing Trends: {(trend_categorical_final == 1).sum().values}\n'
textstr += f'Decreasing Trends: {(trend_categorical_final == -1).sum().values}\n'
textstr += f'Percentage Significant: {(mk_results.significant.sum() / valid_land_cells * 100).values:.1f}%'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Print detailed statistics including Sen's slope
print(f"\nDetailed Statistics:")
print(f"Mean Kendall's tau: {mk_results.tau.mean().values:.4f}")
print(f"Std Kendall's tau: {mk_results.tau.std().values:.4f}")
print(f"Min p-value: {mk_results.p_value.min().values:.6f}")
print(f"Max p-value: {mk_results.p_value.max().values:.6f}")
print(f"Mean p-value: {mk_results.p_value.mean().values:.4f}")

print(f"\nSen's Slope Statistics (Original Units):")
print(f"Mean Sen's slope: {mk_results.sens_slope.mean().values:.6f} events/year")
print(f"Std Sen's slope: {mk_results.sens_slope.std().values:.6f} events/year")
print(f"Min Sen's slope: {mk_results.sens_slope.min().values:.6f} events/year")
print(f"Max Sen's slope: {mk_results.sens_slope.max().values:.6f} events/year")

# Statistics for significant trends only
significant_slopes_original = mk_results.sens_slope.where(mk_results.significant)
print(f"\nSen's Slope Statistics (Significant Trends Only):")
print(f"Mean: {significant_slopes_original.mean().values:.6f} events/year")
print(f"Std: {significant_slopes_original.std().values:.6f} events/year")
print(f"Min: {significant_slopes_original.min().values:.6f} events/year")
print(f"Max: {significant_slopes_original.max().values:.6f} events/year")

# Count positive vs negative significant slopes
positive_slopes = (significant_slopes_original > 0).sum().values
negative_slopes = (significant_slopes_original < 0).sum().values
print(f"Positive significant slopes: {positive_slopes}")
print(f"Negative significant slopes: {negative_slopes}")

# Interpretation for discrete events
print(f"\nInterpretation (Original Units):")
print(f"- Sen's slope represents the rate of change in whiplash event frequency per year")
print(f"- Values close to 0 indicate little to no change in event frequency")
print(f"- Positive values indicate increasing event frequency over time")
print(f"- Negative values indicate decreasing event frequency over time")
print(f"- For discrete events, slopes are typically very small (e.g., 0.001-0.01 events/year)")

# Calculate expected change over the time period
time_span = len(df.time)
if time_span > 0:
    print(f"\nTime Series Context:")
    print(f"Total time period: {time_span} time steps")
    print(f"Expected change over full period for mean slope: {mk_results.sens_slope.mean().values * time_span:.4f} events")
    print(f"Expected change for significant trends: {significant_slopes_original.mean().values * time_span:.4f} events")

# Save results to NetCDF file
output_file = "/Users/rouhinmitra/Projects/Whiplash/Data/mann_kendall_results.nc"
mk_results.to_netcdf(output_file)
print(f"\nResults saved to: {output_file}")
print(df.whiplash_event_frequency.shape)