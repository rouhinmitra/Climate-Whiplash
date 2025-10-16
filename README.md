# Trend Analysis: Whiplash Event Frequency

This repository contains Python scripts for analyzing climate trend patterns in whiplash event frequency across the Western United States. The analysis includes both individual grid cell trend detection and field significance testing to determine if observed patterns represent real climate signals. Based on your shapefile input it can be done for any basin/region

## Overview

### What This Analysis Does

- **Trend Detection**: Identifies statistically significant trends in whiplash event frequency using Mann-Kendall tests and Sen's slope estimation
- **Field Significance Testing**: Uses Monte Carlo permutation testing to determine if spatial patterns of significant trends are field significant (not due to random chance)
- **Visualization**: Creates  plots showing trend patterns, statistical significance, and field significance results

### Scripts

- `read_events.py`: Data loading and preprocessing utilities for NetCDF climate data
- `trend_analysis.py`: Mann-Kendall trend analysis with Sen's slope estimation for each grid cell
- `field_significance_test.py`: Monte Carlo field significance testing to validate spatial patterns

## Installation


### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd significance
```

### Step 2: Install Dependencies

Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv climate_analysis_env

# Activate virtual environment
# On macOS/Linux:
source climate_analysis_env/bin/activate
# On Windows:
# climate_analysis_env\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import xarray, geopandas, scipy, matplotlib; print('All dependencies installed successfully!')"
```

## Data Requirements

Before running the analysis, ensure you have the following data files in the specified locations:

### Required Data Files

1. **NetCDF Climate Data**:
   - `SPEI12_wet-dry_ERA5_whiplash_event_frequency.nc`
   - `SPEI12_dry-wet_ERA5_whiplash_event_frequency.nc`
   
   **Data Source**: Whiplash event frequency data is available from [Swain et al. 2025](https://www.nature.com/articles/s43017-024-00624-z) - "Hydroclimate volatility on a warming Earth" published in Nature Reviews Earth & Environment.

2. **Shapefile Data**:
   - `cb_2018_us_nation_5m/cb_2018_us_nation_5m.shp` (US boundary shapefile)


## Usage

### Step 1: Run Trend Analysis

Start with the Mann-Kendall trend analysis to identify significant trends in whiplash event frequency:

```bash
python trend_analysis.py
```

### Step 2: Run Field Significance Test

After completing the trend analysis, run the field significance test:

```bash
python field_significance_test.py
```
=
