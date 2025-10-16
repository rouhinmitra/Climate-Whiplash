import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

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
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 3:
        return {
            'tau': np.nan,
            'p_value': np.nan,
            'trend': 'insufficient_data',
            'significant': False
        }
    
    # Calculate Kendall's tau and p-value
    tau, p_value = kendalltau(range(len(clean_data)), clean_data)
    
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
        'significant': significant
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
    """
    # Get dimensions
    spatial_dims = [dim for dim in data_array.dims if dim != 'time']
    
    # Initialize result arrays
    tau_array = np.full([data_array.sizes[dim] for dim in spatial_dims], np.nan)
    p_value_array = np.full([data_array.sizes[dim] for dim in spatial_dims], np.nan)
    trend_array = np.full([data_array.sizes[dim] for dim in spatial_dims], 'no_trend', dtype=object)
    significant_array = np.full([data_array.sizes[dim] for dim in spatial_dims], False)
    
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
    
    # Create xarray Dataset with results
    coords = {dim: data_array[dim] for dim in spatial_dims}
    
    results = xr.Dataset({
        'tau': (spatial_dims, tau_array),
        'p_value': (spatial_dims, p_value_array),
        'trend': (spatial_dims, trend_array),
        'significant': (spatial_dims, significant_array)
    }, coords=coords)
    
    return results

def monte_carlo_field_significance(data_array, n_permutations=1000, alpha=0.05, random_seed=142):
    """
    Perform Monte Carlo simulation for field significance testing.
    
    Parameters:
    -----------
    data_array : xarray.DataArray
        DataArray with time dimension for trend analysis
    n_permutations : int
        Number of Monte Carlo permutations (default: 1000)
    alpha : float
        Significance level (default: 0.05)
    random_seed : int
        Random seed for reproducibility (default: 142)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - observed_significant: number of significant grid cells in original data
        - permutation_counts: list of significant grid cell counts for each permutation
        - p_value: field significance p-value
        - field_significant: boolean indicating if field is significant
        - confidence_intervals: 95% confidence intervals
    """
    print(f"Starting Monte Carlo field significance test with {n_permutations} permutations...")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get original results
    print("Computing original Mann-Kendall results...")
    original_results = grid_wise_mann_kendall(data_array, alpha)
    observed_significant = original_results.significant.sum().values
    print(f"Observed significant grid cells: {observed_significant}")
    
    # Get valid land cells (exclude NaN areas)
    valid_land_cells = (~np.isnan(original_results.tau)).sum().values
    print(f"Valid land cells: {valid_land_cells}")
    
    # Store permutation results
    permutation_counts = []
    
    # Perform Monte Carlo permutations
    print("Performing Monte Carlo permutations...")
    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_permutations} permutations...")
        
        # Create permuted data by shuffling time dimension
        permuted_data = data_array.copy()
        time_indices = np.arange(len(data_array.time))
        np.random.shuffle(time_indices)
        permuted_data = permuted_data.isel(time=time_indices)
        
        # Compute Mann-Kendall test on permuted data
        permuted_results = grid_wise_mann_kendall(permuted_data, alpha)
        permuted_significant = permuted_results.significant.sum().values
        permutation_counts.append(permuted_significant)
    
    # Calculate field significance
    permutation_counts = np.array(permutation_counts)
    
    # Count how many permutations had >= observed significant cells
    p_value = np.sum(permutation_counts >= observed_significant) / n_permutations
    
    # Calculate confidence intervals
    confidence_intervals = np.percentile(permutation_counts, [2.5, 97.5])
    
    # Determine field significance
    field_significant = p_value < 0.05
    
    print(f"\nField Significance Results:")
    print(f"Observed significant cells: {observed_significant}")
    print(f"Mean permutation significant cells: {np.mean(permutation_counts):.2f}")
    print(f"Std permutation significant cells: {np.std(permutation_counts):.2f}")
    print(f"95% confidence interval: [{confidence_intervals[0]:.2f}, {confidence_intervals[1]:.2f}]")
    print(f"Field significance p-value: {p_value:.4f}")
    print(f"Field significant: {field_significant}")
    
    return {
        'observed_significant': observed_significant,
        'permutation_counts': permutation_counts,
        'p_value': p_value,
        'field_significant': field_significant,
        'confidence_intervals': confidence_intervals,
        'original_results': original_results
    }

def plot_field_significance_results(results, western_us_geom=None, save_path=None):
    """
    Create visualization of field significance test results.
    
    Parameters:
    -----------
    results : dict
        Results from monte_carlo_field_significance function
    western_us_geom : geopandas.GeoDataFrame, optional
        Western US boundary for overlay
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Field Significance Test Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Histogram of permutation results
    axes[0,0].hist(results['permutation_counts'], bins=50, alpha=0.7, color='skyblue', 
                   edgecolor='black', linewidth=0.5)
    axes[0,0].axvline(results['observed_significant'], color='red', linestyle='--', 
                     linewidth=2, label=f'Observed: {results["observed_significant"]}')
    axes[0,0].axvline(np.mean(results['permutation_counts']), color='green', linestyle='-', 
                     linewidth=2, label=f'Mean: {np.mean(results["permutation_counts"]):.1f}')
    axes[0,0].set_xlabel('Number of Significant Grid Cells')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Monte Carlo Permutation Results')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative distribution
    sorted_counts = np.sort(results['permutation_counts'])
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[0,1].plot(sorted_counts, cumulative, 'b-', linewidth=2)
    axes[0,1].axvline(results['observed_significant'], color='red', linestyle='--', 
                     linewidth=2, label=f'Observed: {results["observed_significant"]}')
    axes[0,1].set_xlabel('Number of Significant Grid Cells')
    axes[0,1].set_ylabel('Cumulative Probability')
    axes[0,1].set_title('Cumulative Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Significant trends map
    significant_trends = xr.where(results['original_results'].significant, 
                                xr.where(results['original_results'].tau > 0, 1, -1), 
                                np.nan)
    significant_trends.plot(ax=axes[1,0], cmap='RdBu_r', center=0, 
                           cbar_kwargs={'label': 'Trend Direction', 'shrink': 0.6})
    axes[1,0].set_title('Significant Trends\n(Red: Increasing, Blue: Decreasing)')
    axes[1,0].set_xlabel('')
    axes[1,0].set_ylabel('')
    
    # Add western US boundary to trend map
    if western_us_geom is not None:
        western_us_geom.plot(ax=axes[1,0], facecolor='none', edgecolor='black', linewidth=1)
    
    # Plot 4: P-values map
    results['original_results'].p_value.plot(ax=axes[1,1], cmap='viridis', 
                                            cbar_kwargs={'label': 'P-value', 'shrink': 0.6})
    axes[1,1].set_title('P-values')
    axes[1,1].set_xlabel('')
    axes[1,1].set_ylabel('')
    
    # Add western US boundary to p-value map
    if western_us_geom is not None:
        western_us_geom.plot(ax=axes[1,1], facecolor='none', edgecolor='black', linewidth=1)
    
    # Add text box with results
    textstr = f'Field Significance Test Results:\n'
    textstr += f'Observed significant: {results["observed_significant"]}\n'
    textstr += f'Mean permutation: {np.mean(results["permutation_counts"]):.1f}\n'
    textstr += f'Field p-value: {results["p_value"]:.4f}\n'
    textstr += f'Field significant: {results["field_significant"]}\n'
    textstr += f'95% CI: [{results["confidence_intervals"][0]:.1f}, {results["confidence_intervals"][1]:.1f}]'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[0,0].text(0.02, 0.98, textstr, transform=axes[0,0].transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    """
    Main function to run field significance test.
    """
    # Load your actual whiplash event data
    print("Loading whiplash event data...")
    from read_events import load_and_process_data, mask_to_western_us
    import geopandas as gpd
    from shapely.geometry import box
    
    # Data paths
    dir_path1 = "/Users/rouhinmitra/Projects/Whiplash/Data/SPEI12_wet-dry_ERA5_whiplash_event_frequency.nc"
    dir_path2 = "/Users/rouhinmitra/Projects/Whiplash/Data/SPEI12_dry-wet_ERA5_whiplash_event_frequency.nc"
    
    # Load US shapefile and create western US mask
    us = gpd.read_file('/Users/rouhinmitra/Projects/Whiplash/Data/cb_2018_us_nation_5m/cb_2018_us_nation_5m.shp')
    us = us.to_crs(epsg=4326)
    western_bounds = box(-180, 24, -100, 50)
    western_us_geom = gpd.clip(us, western_bounds)
    western_us_geom = western_us_geom.to_crs(epsg=4326)
    
    # Load and process whiplash event data
    df = load_and_process_data(dir_path1, dir_path2)
    df = mask_to_western_us(df, western_us_geom)
    
    print(f"Data shape: {df.whiplash_event_frequency.shape}")
    print(f"Time range: {df.time.min().values} - {df.time.max().values}")
    print(f"Latitude range: {df.lat.min().values:.2f} - {df.lat.max().values:.2f}")
    print(f"Longitude range: {df.lon.min().values:.2f} - {df.lon.max().values:.2f}")
    
    # Check for valid data
    valid_land_cells = (~np.isnan(df.whiplash_event_frequency)).sum().values
    total_cells = df.whiplash_event_frequency.size
    print(f"Valid land cells: {valid_land_cells}")
    print(f"Total grid cells: {total_cells}")
    print(f"Percentage valid: {(valid_land_cells/total_cells)*100:.1f}%")
    
    # Run field significance test
    results = monte_carlo_field_significance(
        df.whiplash_event_frequency, 
        n_permutations=1000,  # Increase for more robust results
        alpha=0.05,
        random_seed=142
    )
    
    # Create visualization
    plot_field_significance_results(results, western_us_geom=western_us_geom, 
                                   save_path="field_significance_results.png")
    
    # Print summary
    print(f"\n" + "="*50)
    print("FIELD SIGNIFICANCE TEST SUMMARY")
    print("="*50)
    print(f"Observed significant grid cells: {results['observed_significant']}")
    print(f"Field significance p-value: {results['p_value']:.4f}")
    print(f"Field significant: {results['field_significant']}")
    
    if results['field_significant']:
        print("\n✓ The spatial pattern of significant trends is FIELD SIGNIFICANT.")
        print("  This suggests the trends represent real climate signals, not random noise.")
    else:
        print("\n✗ The spatial pattern of significant trends is NOT field significant.")
        print("  This suggests the trends could be due to random chance.")
    
    print(f"\n95% Confidence Interval: [{results['confidence_intervals'][0]:.1f}, {results['confidence_intervals'][1]:.1f}]")
    print(f"Mean permutation significant cells: {np.mean(results['permutation_counts']):.1f} ± {np.std(results['permutation_counts']):.1f}")

if __name__ == "__main__":
    main()
