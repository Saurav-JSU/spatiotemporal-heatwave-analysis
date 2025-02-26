# ERA5 Climate Heatwave Analysis Pipeline

A comprehensive Python-based analysis pipeline for calculating heatwave indices and visualizing climate patterns using ERA5 reanalysis data. This repository contains the code used in the published paper on heatwave analysis in Mississippi counties.

## Overview

This pipeline enables the analysis of climate data to identify and characterize heatwaves across different geographical regions. It allows for flexible state selection, comprehensive heatwave metrics calculation, and generates visualizations of heatwave patterns.

## Features

- **Flexible geographical analysis**: Analyze data for Mississippi, specific states, or the entire CONUS (Continental United States)
- **Comprehensive heatwave indices**: Calculate standard heatwave metrics (HWN, HWTD, HWLD, HWMT, HDT)
- **Statistical analysis**: Calculate trends, perform significance testing
- **Visualizations**: Generate heatmaps, spatial maps, and temporal visualizations
- **Modular design**: Run the entire pipeline or specific components as needed

## Installation

### Prerequisites

- Python 3.8+
- Required Python packages:
  - pandas
  - numpy
  - geopandas
  - matplotlib
  - seaborn
  - scipy
  - xlsxwriter
  - openpyxl

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heatwave-analysis.git
cd heatwave-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Requirements

### Directory Structure

The pipeline expects the following directory structure:

```
project_root/
├── Shape-File/                # Contains CONUS shapefile data
│   ├── CONUS_county.shp       # Shapefile for CONUS counties
│   └── ...                    # Other shapefile-related files
│
├── Daily Data/                # Contains ERA5 daily data files
│   ├── ERA5_USA_YYYY_MM.csv   # ERA5 data files (by year and month)
│   └── ...
│
├── Output Directories (created automatically):
│   ├── Heatwave Indices/      # Calculated heatwave indices
│   ├── Mean and Max/          # Mean and max values of indices
│   ├── Figures/               # Generated visualization images
│   ├── Shifting Data/         # Processed hourly/monthly data
│   ├── General Indices/       # General temperature indices
│   └── Temporal/              # Temporal analysis results
```

### Input Data

1. **Shapefiles**: County-level shapefiles with a 'sequence' column that matches station IDs in ERA5 data
2. **ERA5 Data**: Daily temperature data files organized by year and month

## Usage

### Basic Usage

Run the full analysis pipeline for Mississippi:

```python
from heatwave_analysis import ClimateAnalysisPipeline

pipeline = ClimateAnalysisPipeline()
pipeline.run()
```

### Customized Analysis

Analyze specific states:

```python
# For a single state
pipeline = ClimateAnalysisPipeline(states=["Alabama"])

# For multiple states
pipeline = ClimateAnalysisPipeline(states=["Mississippi", "Alabama", "Georgia"])

# For all CONUS states
pipeline = ClimateAnalysisPipeline(states="ALL")
```

### Running Specific Steps

```python
# Run from a specific step to the end
pipeline.run_from_step('calculate_general_indices')

# Run a range of steps
pipeline.run_range('calculate_mean_max_values', 'create_heatwave_maps')

# Run only specific steps
pipeline.run_selected_steps(['calculate_trends', 'create_trend_maps'])
```

## Pipeline Components

The pipeline consists of the following main components:

1. **Data Loading**: Loads geographic data and station information
2. **Temperature Metrics Calculation**: Processes raw ERA5 data into temperature metrics
3. **Heatwave Analysis**: Calculates heatwave indices based on percentile thresholds
4. **Statistical Analysis**: Performs trend analysis and significance testing
5. **Visualization**: Creates maps, heatmaps, and other visual representations of the data

## Heatwave Indices

The pipeline calculates the following standard heatwave indices:

- **HWN** (Heatwave Number): Count of discrete heatwave events
- **HWTD** (Heatwave Total Days): Total number of days in heatwaves
- **HWLD** (Heatwave Longest Duration): Length of the longest heatwave in days
- **HWMT** (Heatwave Mean Temperature): Average temperature during heatwaves
- **HDT** (Hottest Day Temperature): Maximum temperature recorded during heatwaves

A heatwave is defined as 3+ consecutive days with temperature exceeding the 95th percentile threshold (configurable).

## Configuration Options

The pipeline can be configured through the `Config` class:

- **Date range**: Modify `start_year` and `end_year` (default: 1980-2022)
- **States**: Specify which states to analyze
- **Heatwave threshold**: Adjust `heatwave_percentile` (default: 0.95)
- **Significance threshold**: Adjust `significance_threshold` for trend analysis (default: 0.10)

## Output Files

The pipeline generates various output files, including:

- Excel files with heatwave indices by year and station
- CSV files with statistical analysis results
- PNG images with visualizations of heatwave patterns
- Summary files with temporal analysis

## Citation

If you use this code in your research, please cite the original paper:

```
Bhattarai, S., Bista, S., Sharma, S. et al. Spatiotemporal characterization of heatwave exposure across historically vulnerable communities. Sci Rep 14, 20882 (2024). https://doi.org/10.1038/s41598-024-71704-9
```

## Acknowledgments

- ERA5 data provided by the Copernicus Climate Change Service
- HICORPS Project (Hydrological Impacts Computing, Outreach, Resiliency and Partnership Project) in collaboration with ERDC and WOOLPERT. 
- Jackson State University
