"""
ERA5 Climate Analysis Pipeline
------------------------------
This script provides a comprehensive analysis of ERA5 climate data to calculate 
heatwave indices and visualize climate patterns in Mississippi counties.

Author: [Saurav Bhattarai]
Date: February 2025
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import calendar
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Config:
    """Configuration parameters for the analysis pipeline."""
    
    def __init__(self, base_dir=".", states=None):
        """Initialize configuration with default values.
        
        Args:
            base_dir (str): Base directory for all data and outputs
            states (list/str, optional): States to include in the analysis. 
                - List of state names for specific states
                - "ALL" for all CONUS states
                - None for default (Mississippi)
        """
        self.start_year = 1980
        self.end_year = 2022
        self.base_dir = base_dir
        
        # Set states to analyze
        self.states = states if states is not None else ["Mississippi"]
        self.include_all_states = (self.states == "ALL")
        
        # Input directories
        self.shape_dir = os.path.join(base_dir, "Shape-File")
        self.daily_dir = os.path.join(base_dir, "Daily Data")
        
        # Output directories
        self.heatwave_dir = os.path.join(base_dir, "Heatwave Indices")
        self.mean_max_dir = os.path.join(base_dir, "Mean and Max")
        self.figures_dir = os.path.join(base_dir, "Figures")
        self.shifting_dir = os.path.join(base_dir, "Shifting Data")
        self.general_dir = os.path.join(base_dir, "General Indices")
        self.temporal_dir = os.path.join(base_dir, "Temporal")
        
        # Create output directories if they don't exist
        for directory in [self.heatwave_dir, self.mean_max_dir, 
                         self.figures_dir, self.shifting_dir, self.general_dir,
                         self.temporal_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Analysis parameters
        self.heatwave_percentile = 0.95
        self.significance_threshold = 0.10


class DataLoader:
    """Handles loading of geographic and climate data."""
    
    def __init__(self, config):
        """Initialize data loader with configuration.
        
        Args:
            config (Config): Configuration instance
        """
        self.config = config
        self.station_numbers = []
        self.gdf = None
        self.gdf_file = None
        
    def load_shapefiles(self):
        """Load shapefiles and prepare geographic data."""
        print("Loading shapefiles...")
        try:
            # Load the county shapefile for CONUS
            self.gdf = gpd.read_file(os.path.join(self.config.shape_dir, 'CONUS_county.shp'))
            
            # Add sequence numbers if they don't exist
            if 'sequence' not in self.gdf.columns:
                self.gdf.loc[:, 'sequence'] = range(1, len(self.gdf) + 1)
                # Save with sequence numbers
                self.gdf.to_file(os.path.join(self.config.shape_dir, 'data_with_sequence.shp'))
            
            # Load the shapefile with sequence
            self.gdf_file = gpd.read_file(os.path.join(self.config.shape_dir, 'data_with_sequence.shp'))
            
            # Filter for selected states
            if self.config.include_all_states:
                print("Including all states in CONUS")
                # No filtering needed for all states
                filtered_gdf = self.gdf_file
                state_names = sorted(self.gdf_file["NAME_1"].unique())
                print(f"Total of {len(state_names)} states included: {', '.join(state_names)}")
            else:
                # Filter for specified states
                filtered_gdf = self.gdf_file[self.gdf_file["NAME_1"].isin(self.config.states)]
                print(f"Filtering for states: {', '.join(self.config.states)}")
                
                # Check if any states were not found
                found_states = filtered_gdf["NAME_1"].unique()
                missing_states = [state for state in self.config.states if state not in found_states]
                if missing_states:
                    print(f"Warning: The following states were not found in the shapefile: {', '.join(missing_states)}")
                    
                if filtered_gdf.empty:
                    print("Error: No counties found for the specified states.")
                    return False
            
            # Update the filtered GeoDataFrame
            self.gdf_file = filtered_gdf
            
            # Extract station numbers
            sequence_data = self.gdf_file['sequence']
            self.station_numbers = [str(station_number) for station_number in sequence_data.tolist()]
            
            print(f"Loaded {len(self.station_numbers)} station numbers for {len(self.gdf_file['NAME_1'].unique())} states")
            return True
        except Exception as e:
            print(f"Error loading shapefiles: {e}")
            return False


class DataProcessor:
    """Processes ERA5 data for climate analysis."""
    
    def __init__(self, config, data_loader):
        """Initialize data processor with configuration and data loader.
        
        Args:
            config (Config): Configuration instance
            data_loader (DataLoader): Data loader instance
        """
        self.config = config
        self.data_loader = data_loader
            
    def calculate_temperature_metrics(self):
        """Calculate daily max, min, and average temperatures from the Daily Data folder."""
        print("Calculating temperature metrics...")
        try:
            # Initialize lists to store DataFrames
            max_temperatures_list = []
            min_temperatures_list = []
            avg_temperatures_list = []
            
            # Get list of CSV files in the Daily Data folder
            files = [f for f in os.listdir(self.config.daily_dir) 
                   if f.endswith('.csv') and f.startswith('ERA5_USA_')]
            
            if not files:
                print("No ERA5 CSV files found in the Daily Data folder")
                return False
                
            for filename in sorted(files):
                year = int(filename.split('_')[2])
                month = int(filename.split('_')[3].split('.')[0])
                
                if year < self.config.start_year or year > self.config.end_year:
                    continue
                    
                print(f'Calculating temperature metrics for {filename}')
                input_file_path = os.path.join(self.config.daily_dir, filename)
                
                # Read data
                df = pd.read_csv(input_file_path)
                
                # Calculate daily temperature metrics
                station_cols = [col for col in df.columns if col not in ['Year', 'Month', 'Day', 'Hour', 'DateTime']]
                daily_max_temps = df.groupby(['Year', 'Month', 'Day'])[station_cols].max()
                daily_min_temps = df.groupby(['Year', 'Month', 'Day'])[station_cols].min()
                daily_avg_temps = (daily_max_temps + daily_min_temps) / 2
                
                # Append to lists
                max_temperatures_list.append(daily_max_temps)
                min_temperatures_list.append(daily_min_temps)
                avg_temperatures_list.append(daily_avg_temps)
            
            # Concatenate data
            if max_temperatures_list:
                all_max_temps = pd.concat(max_temperatures_list)
                all_min_temps = pd.concat(min_temperatures_list)
                all_avg_temps = pd.concat(avg_temperatures_list)
                
                # Save to files
                all_max_temps.to_csv(os.path.join(self.config.daily_dir, 'Tmax.csv'))
                all_min_temps.to_csv(os.path.join(self.config.daily_dir, 'Tmin.csv'))
                all_avg_temps.to_csv(os.path.join(self.config.daily_dir, 'Tavg.csv'))
                
                print("Temperature metrics calculation completed.")
                return True
            else:
                print("No temperature data was processed. Check input files.")
                return False
        except Exception as e:
            print(f"Error calculating temperature metrics: {e}")
            return False
            
    def combine_and_average_data(self):
        """Combine and average data by hour and month."""
        print("Combining and averaging data...")
        try:
            # Check if temperature files exist first
            temp_files = ['Tmax.csv', 'Tmin.csv', 'Tavg.csv']
            missing_files = [f for f in temp_files if not os.path.exists(os.path.join(self.config.daily_dir, f))]
            
            if missing_files:
                print(f"Missing temperature files: {missing_files}")
                print("Please run calculate_temperature_metrics first.")
                return False
            
            # Combine and average by hour
            self._combine_and_average_by_column('Hour', 'Hourly')
            
            # Combine and average by month
            self._combine_and_average_by_column('Month', 'Monthly')
            
            print("Data combination and averaging completed.")
            return True
        except Exception as e:
            print(f"Error combining and averaging data: {e}")
            return False
            
    def _combine_and_average_by_column(self, group_column, output_prefix):
        """Helper method to combine and average data by a specific column.
        
        Args:
            group_column (str): Column to group by (e.g., 'Hour', 'Month')
            output_prefix (str): Prefix for output files
        """
        # List CSV files
        excluded_files = ['Tmax.csv', 'Tmin.csv', 'Tavg.csv']
        csv_files = [f for f in os.listdir(self.config.daily_dir) 
                   if f.endswith('.csv') and f.startswith('ERA5_USA_') and f not in excluded_files]
        
        # Sort files by year and month
        csv_files.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3].split('.')[0])))
        
        # Initialize DataFrames
        max_data = pd.DataFrame()
        mean_data = pd.DataFrame()
        min_data = pd.DataFrame()
        
        # Process each file
        for csv_file in csv_files:
            print(f'Combining and averaging {csv_file}')
            csv_path = os.path.join(self.config.daily_dir, csv_file)
            df = pd.read_csv(csv_path)
            
            # Group and calculate
            grouped_max = df.groupby(group_column).max().reset_index()
            grouped_mean = df.groupby(group_column).mean().reset_index()
            grouped_min = df.groupby(group_column).min().reset_index()
            
            # Append to DataFrames
            max_data = pd.concat([max_data, grouped_max], ignore_index=True)
            mean_data = pd.concat([mean_data, grouped_mean], ignore_index=True)
            min_data = pd.concat([min_data, grouped_min], ignore_index=True)
            
        # Save results
        max_data.to_csv(os.path.join(self.config.shifting_dir, f'{output_prefix}_max.csv'), index=False)
        mean_data.to_csv(os.path.join(self.config.shifting_dir, f'{output_prefix}_mean.csv'), index=False)
        min_data.to_csv(os.path.join(self.config.shifting_dir, f'{output_prefix}_min.csv'), index=False)


class HeatwaveAnalyzer:
    """Analyzes temperature data to identify and characterize heatwaves."""
    
    def __init__(self, config, data_loader):
        """Initialize heatwave analyzer.
        
        Args:
            config (Config): Configuration instance
            data_loader (DataLoader): Data loader with station numbers
        """
        self.config = config
        self.data_loader = data_loader
        
    def calculate_heatwave_indices(self):
        """Calculate heatwave indices for different temperature datasets."""
        print("Calculating heatwave indices...")
        try:
            # Check if temperature files exist first
            temp_files = ['Tmax.csv', 'Tmin.csv', 'Tavg.csv']
            missing_files = [f for f in temp_files if not os.path.exists(os.path.join(self.config.daily_dir, f))]
            
            if missing_files:
                print(f"Missing temperature files: {missing_files}")
                print("Please run calculate_temperature_metrics first.")
                return False
                
            # Process individual temperature files
            csv_files = ['Tmax.csv', 'Tmin.csv', 'Tavg.csv']
            output_files = ['Output_day.xlsx', 'Output_night.xlsx', 'Output_average.xlsx']
            
            for csv_file, output_file in zip(csv_files, output_files):
                print(f"Processing {csv_file} for heatwave indices...")
                self._calculate_indices_for_file(csv_file, output_file)
                
            # Calculate day & night combined indices
            print("Processing day & night combined heatwave indices...")
            self._calculate_day_night_indices()
            
            print("Heatwave indices calculation completed.")
            return True
        except Exception as e:
            print(f"Error calculating heatwave indices: {e}")
            return False
    
    def _calculate_indices_for_file(self, csv_file, output_file):
        """Calculate heatwave indices for a single temperature file.
        
        Args:
            csv_file (str): Input CSV file name
            output_file (str): Output Excel file name
        """
        # Read temperature data
        df = pd.read_csv(os.path.join(self.config.daily_dir, csv_file))
        
        # Calculate percentiles for threshold
        percentiles = df.iloc[:, 3:].quantile(self.config.heatwave_percentile)
        
        # Initialize DataFrames for results
        heatwave_data = pd.DataFrame(columns=["Year"])
        contributing_days_data = pd.DataFrame(columns=["Year"])
        longest_heatwave_length_data = pd.DataFrame(columns=["Year"])
        average_magnitude_data = pd.DataFrame(columns=["Year"])
        hottest_temperature_data = pd.DataFrame(columns=["Year"])
        
        # Process each year
        for year in df["Year"].unique():
            year_data = df[df["Year"] == year]
            
            # Initialize lists for results
            heatwave_events = []
            contributing_days = []
            longest_heatwave_length = []
            average_magnitude = []
            hottest_temperature = []
            
            # Process each station
            for station in df.columns[3:]:
                print(f"Processing Station {station} in year {year}")
                station_data = year_data[station]
                
                # Identify heatwave events
                is_heatwave = (station_data > percentiles[station]).astype(int)
                
                # Initialize counters
                heatwave_events_count = 0
                in_heatwave = False
                current_heatwave = 0
                contributing_days_count = 0
                max_heatwave_length = 0
                max_heatwave_temperature = float('-999')
                sum_temperature_at_heatwaves = 0
                
                # Analyze each day
                for value, temperature in zip(is_heatwave, station_data):
                    if value == 1 and not in_heatwave:
                        # Start of a new heatwave
                        in_heatwave = True
                        current_heatwave = 1
                        hottest_temp = temperature
                        total_magnitude = temperature
                    elif value == 1 and in_heatwave:
                        # Continuation of a heatwave
                        current_heatwave += 1
                        total_magnitude += temperature
                        hottest_temp = max(hottest_temp, temperature)
                    elif value == 0 and in_heatwave:
                        # End of a heatwave
                        if current_heatwave >= 3:
                            # Only count as heatwave if lasted at least 3 days
                            heatwave_events_count += 1
                            contributing_days_count += current_heatwave
                            max_heatwave_length = max(max_heatwave_length, current_heatwave)
                            max_heatwave_temperature = max(max_heatwave_temperature, hottest_temp)
                            sum_temperature_at_heatwaves += total_magnitude
                        
                        in_heatwave = False
                        current_heatwave = 0
                
                # Check if we ended the year in a heatwave
                if in_heatwave and current_heatwave >= 3:
                    heatwave_events_count += 1
                    contributing_days_count += current_heatwave
                    max_heatwave_length = max(max_heatwave_length, current_heatwave)
                    max_heatwave_temperature = max(max_heatwave_temperature, hottest_temp)
                    sum_temperature_at_heatwaves += total_magnitude
                
                # Calculate average temperature during heatwaves
                average_temperature_at_heatwaves = sum_temperature_at_heatwaves / contributing_days_count if contributing_days_count > 0 else 0
                
                # Append results
                heatwave_events.append(heatwave_events_count)
                contributing_days.append(contributing_days_count)
                longest_heatwave_length.append(max_heatwave_length)
                average_magnitude.append(average_temperature_at_heatwaves)
                hottest_temperature.append(max_heatwave_temperature)
            
            # Create DataFrames for the current year's data
            heatwave_data_year = pd.DataFrame({"Year": year, **{station: count for station, count in zip(df.columns[3:], heatwave_events)}}, index=[0])
            contributing_days_data_year = pd.DataFrame({"Year": year, **{station: count for station, count in zip(df.columns[3:], contributing_days)}}, index=[0])
            longest_heatwave_length_data_year = pd.DataFrame({"Year": year, **{station: length for station, length in zip(df.columns[3:], longest_heatwave_length)}}, index=[0])
            average_magnitude_data_year = pd.DataFrame({"Year": year, **{station: magnitude for station, magnitude in zip(df.columns[3:], average_magnitude)}}, index=[0])
            hottest_temperature_data_year = pd.DataFrame({"Year": year, **{station: temp for station, temp in zip(df.columns[3:], hottest_temperature)}}, index=[0])
            
            # Concatenate to main DataFrames
            heatwave_data = pd.concat([heatwave_data, heatwave_data_year], ignore_index=True)
            contributing_days_data = pd.concat([contributing_days_data, contributing_days_data_year], ignore_index=True)
            longest_heatwave_length_data = pd.concat([longest_heatwave_length_data, longest_heatwave_length_data_year], ignore_index=True)
            average_magnitude_data = pd.concat([average_magnitude_data, average_magnitude_data_year], ignore_index=True)
            hottest_temperature_data = pd.concat([hottest_temperature_data, hottest_temperature_data_year], ignore_index=True)
        
        # Save results to Excel
        output_path = os.path.join(self.config.heatwave_dir, output_file)
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            heatwave_data.to_excel(writer, sheet_name="HWN", index=False)
            contributing_days_data.to_excel(writer, sheet_name="HWTD", index=False)
            longest_heatwave_length_data.to_excel(writer, sheet_name="HWLD", index=False)
            average_magnitude_data.to_excel(writer, sheet_name="HWMT", index=False)
            hottest_temperature_data.to_excel(writer, sheet_name="HDT", index=False)
    
    def _calculate_day_night_indices(self):
        """Calculate combined day and night heatwave indices."""
        # Read temperature data
        tmax = pd.read_csv(os.path.join(self.config.daily_dir, 'Tmax.csv'))
        tmin = pd.read_csv(os.path.join(self.config.daily_dir, 'Tmin.csv'))
        
        # Calculate percentiles
        percentiles_max = tmax.iloc[:, 3:].quantile(self.config.heatwave_percentile)
        percentiles_min = tmin.iloc[:, 3:].quantile(self.config.heatwave_percentile)
        
        # Initialize DataFrames for results
        heatwave_data = pd.DataFrame(columns=["Year"])
        contributing_days_data = pd.DataFrame(columns=["Year"])
        longest_heatwave_length_data = pd.DataFrame(columns=["Year"])
        average_magnitude_data = pd.DataFrame(columns=["Year"])
        hottest_temperature_data = pd.DataFrame(columns=["Year"])
        
        # Process each year
        for year in tmax["Year"].unique():
            year_data_max = tmax[tmax["Year"] == year]
            year_data_min = tmin[tmin["Year"] == year]
            
            # Initialize lists for results
            heatwave_events = []
            contributing_days = []
            longest_heatwave_length = []
            average_magnitude = []
            hottest_temperature = []
            
            # Process each station
            for station in tmax.columns[3:]:
                print(f"Processing day & night for Station {station} in year {year}")
                station_data_max = year_data_max[station]
                station_data_min = year_data_min[station]
                
                # Identify heatwave events (both day and night exceed thresholds)
                is_heatwave = ((station_data_max > percentiles_max[station]) & 
                              (station_data_min > percentiles_min[station])).astype(int)
                
                # Initialize counters
                heatwave_events_count = 0
                in_heatwave = False
                current_heatwave = 0
                contributing_days_count = 0
                max_heatwave_length = 0
                max_heatwave_temperature = float('-999')
                sum_temperature_at_heatwaves = 0
                
                # Analyze each day
                for value, temperature in zip(is_heatwave, station_data_max):
                    if value == 1 and not in_heatwave:
                        # Start of a new heatwave
                        in_heatwave = True
                        current_heatwave = 1
                        hottest_temp = temperature
                        total_magnitude = temperature
                    elif value == 1 and in_heatwave:
                        # Continuation of a heatwave
                        current_heatwave += 1
                        total_magnitude += temperature
                        hottest_temp = max(hottest_temp, temperature)
                    elif value == 0 and in_heatwave:
                        # End of a heatwave
                        if current_heatwave >= 3:
                            # Only count as heatwave if lasted at least 3 days
                            heatwave_events_count += 1
                            contributing_days_count += current_heatwave
                            max_heatwave_length = max(max_heatwave_length, current_heatwave)
                            max_heatwave_temperature = max(max_heatwave_temperature, hottest_temp)
                            sum_temperature_at_heatwaves += total_magnitude
                        
                        in_heatwave = False
                        current_heatwave = 0
                
                # Check if we ended the year in a heatwave
                if in_heatwave and current_heatwave >= 3:
                    heatwave_events_count += 1
                    contributing_days_count += current_heatwave
                    max_heatwave_length = max(max_heatwave_length, current_heatwave)
                    max_heatwave_temperature = max(max_heatwave_temperature, hottest_temp)
                    sum_temperature_at_heatwaves += total_magnitude
                
                # Calculate average temperature during heatwaves
                average_temperature_at_heatwaves = sum_temperature_at_heatwaves / contributing_days_count if contributing_days_count > 0 else 0
                
                # Append results
                heatwave_events.append(heatwave_events_count)
                contributing_days.append(contributing_days_count)
                longest_heatwave_length.append(max_heatwave_length)
                average_magnitude.append(average_temperature_at_heatwaves)
                hottest_temperature.append(max_heatwave_temperature)
            
            # Create DataFrames for the current year's data
            heatwave_data_year = pd.DataFrame({"Year": year, **{station: count for station, count in zip(tmax.columns[3:], heatwave_events)}}, index=[0])
            contributing_days_data_year = pd.DataFrame({"Year": year, **{station: count for station, count in zip(tmax.columns[3:], contributing_days)}}, index=[0])
            longest_heatwave_length_data_year = pd.DataFrame({"Year": year, **{station: length for station, length in zip(tmax.columns[3:], longest_heatwave_length)}}, index=[0])
            average_magnitude_data_year = pd.DataFrame({"Year": year, **{station: magnitude for station, magnitude in zip(tmax.columns[3:], average_magnitude)}}, index=[0])
            hottest_temperature_data_year = pd.DataFrame({"Year": year, **{station: temp for station, temp in zip(tmax.columns[3:], hottest_temperature)}}, index=[0])
            
            # Concatenate to main DataFrames
            heatwave_data = pd.concat([heatwave_data, heatwave_data_year], ignore_index=True)
            contributing_days_data = pd.concat([contributing_days_data, contributing_days_data_year], ignore_index=True)
            longest_heatwave_length_data = pd.concat([longest_heatwave_length_data, longest_heatwave_length_data_year], ignore_index=True)
            average_magnitude_data = pd.concat([average_magnitude_data, average_magnitude_data_year], ignore_index=True)
            hottest_temperature_data = pd.concat([hottest_temperature_data, hottest_temperature_data_year], ignore_index=True)
        
        # Save results to Excel
        output_file = os.path.join(self.config.heatwave_dir, 'Output_day&night.xlsx')
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            heatwave_data.to_excel(writer, sheet_name="HWN", index=False)
            contributing_days_data.to_excel(writer, sheet_name="HWTD", index=False)
            longest_heatwave_length_data.to_excel(writer, sheet_name="HWLD", index=False)
            average_magnitude_data.to_excel(writer, sheet_name="HWMT", index=False)
            hottest_temperature_data.to_excel(writer, sheet_name="HDT", index=False)
    
    def calculate_mean_max_values(self):
        """Calculate mean and maximum values for heatwave indices."""
        print("Calculating mean and maximum values for heatwave indices...")
        try:
            # Check if heatwave indices files exist
            excel_files = ['Output_night.xlsx', 'Output_day.xlsx', 'Output_average.xlsx', 'Output_day&night.xlsx']
            missing_files = [f for f in excel_files if not os.path.exists(os.path.join(self.config.heatwave_dir, f))]
            
            if missing_files:
                print(f"Missing heatwave indices files: {missing_files}")
                print("Please run calculate_heatwave_indices first.")
                return False
                
            # List Excel files and output files
            excel_files = ['Output_night.xlsx', 'Output_day.xlsx', 'Output_average.xlsx', 'Output_day&night.xlsx']
            output_mean_files = ['Mean_values_night.csv', 'Mean_values_day.csv', 'Mean_values_average.csv', 'Mean_values_day&night.csv']
            output_max_files = ['Max_values_night.csv', 'Max_values_day.csv', 'Max_values_average.csv', 'Max_values_day&night.csv']
            
            for excel_file, output_mean_file, output_max_file in zip(excel_files, output_mean_files, output_max_files):
                print(f"Processing {excel_file} for mean and max values...")
                
                # Load Excel file
                excel_path = os.path.join(self.config.heatwave_dir, excel_file)
                excel_data = pd.ExcelFile(excel_path)
                sheet_names = excel_data.sheet_names
                
                print(f"Found sheets: {sheet_names}")
                
                # Initialize dictionaries
                mean_dataframes = {}
                max_dataframes = {}
                
                for sheet_name in sheet_names:
                    df = excel_data.parse(sheet_name)
                    print(f"Processing sheet: {sheet_name}, shape: {df.shape}")
                    
                    # Exclude non-numeric columns
                    numeric_columns = df.columns[1:]  # Assuming first column is 'Year'
                    
                    # Replace -999 with NaN
                    df[numeric_columns] = df[numeric_columns].replace(-999, np.nan)
                    
                    # Calculate mean and max
                    means = df[numeric_columns].mean()
                    max_values = df[numeric_columns].max()
                    
                    # Store in dictionaries
                    mean_dataframes[sheet_name] = means
                    max_dataframes[sheet_name] = max_values
                
                # Transpose DataFrames
                transposed_mean_dfs = [df.transpose() for df in mean_dataframes.values()]
                transposed_max_dfs = [df.transpose() for df in max_dataframes.values()]
                
                # Get headers from the first sheet
                first_sheet = excel_data.parse(sheet_names[0])
                headers = list(first_sheet.columns[1:])  # Skip 'Year' column
                
                # Concatenate DataFrames along columns axis
                final_mean_df = pd.concat(transposed_mean_dfs, axis=1)
                final_max_df = pd.concat(transposed_max_dfs, axis=1)
                
                print(f"Shape after concatenation - mean: {final_mean_df.shape}, max: {final_max_df.shape}")
                
                # Reset index and add as Station column
                final_mean_df = final_mean_df.reset_index()
                final_max_df = final_max_df.reset_index()
                
                final_mean_df.rename(columns={'index': 'Station'}, inplace=True)
                final_max_df.rename(columns={'index': 'Station'}, inplace=True)
                
                # Set column names dynamically based on sheets
                column_names = ['Station'] + sheet_names
                
                # Make sure column count matches
                if final_mean_df.shape[1] != len(column_names):
                    print(f"Warning: Column count mismatch. DataFrame has {final_mean_df.shape[1]} columns but {len(column_names)} names.")
                    # Use generic column names if needed
                    if final_mean_df.shape[1] > len(column_names):
                        # Add generic names for extra columns
                        extra_cols = final_mean_df.shape[1] - len(column_names)
                        column_names += [f'Column_{i}' for i in range(len(column_names), len(column_names) + extra_cols)]
                    else:
                        # Truncate column names if too many
                        column_names = column_names[:final_mean_df.shape[1]]
                
                final_mean_df.columns = column_names
                final_max_df.columns = column_names
                
                # Save to CSV
                final_mean_df.to_csv(os.path.join(self.config.mean_max_dir, output_mean_file), index=False)
                final_max_df.to_csv(os.path.join(self.config.mean_max_dir, output_max_file), index=False)
            
            print("Mean and maximum values calculation completed.")
            return True
        except Exception as e:
            print(f"Error calculating mean and max values: {e}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_general_indices(self):
        """Calculate general temperature indices."""
        print("Calculating general temperature indices...")
        try:
            # Check if temperature files exist
            temp_files = ['Tmax.csv', 'Tmin.csv']
            missing_files = [f for f in temp_files if not os.path.exists(os.path.join(self.config.daily_dir, f))]
            
            if missing_files:
                print(f"Missing temperature files: {missing_files}")
                print("Please run calculate_temperature_metrics first.")
                return False
                
            # Read temperature data
            tmax = pd.read_csv(os.path.join(self.config.daily_dir, 'Tmax.csv'))
            tmin = pd.read_csv(os.path.join(self.config.daily_dir, 'Tmin.csv'))
            
            # Clean data
            tmax.drop(columns=['Month', 'Day'], inplace=True)
            tmin.drop(columns=['Month', 'Day'], inplace=True)
            tmax.replace(-9999, np.nan, inplace=True)
            tmin.replace(-9999, np.nan, inplace=True)
            
            # Get station columns
            station_cols = tmax.columns[1:]
            
            # Calculate indices
            txx = tmax.groupby(['Year']).max()
            tnn = tmin.groupby(['Year']).min()
            txn = tmax.groupby(['Year']).min()
            tnx = tmin.groupby(['Year']).max()
            
            # Save indices to Excel
            with pd.ExcelWriter(os.path.join(self.config.general_dir, 'General Indices.xlsx'), engine='xlsxwriter') as writer:
                txx.to_excel(writer, sheet_name='Txx')
                tnn.to_excel(writer, sheet_name='Tnn')
                txn.to_excel(writer, sheet_name='Txn')
                tnx.to_excel(writer, sheet_name='Tnx')
            
            print("General indices calculation completed.")
            return True
        except Exception as e:
            print(f"Error calculating general indices: {e}")
            return False


class StatisticalAnalyzer:
    """Performs statistical analysis on climate data."""
    
    def __init__(self, config):
        """Initialize statistical analyzer.
        
        Args:
            config (Config): Configuration instance
        """
        self.config = config
    
    def calculate_trends(self):
        """Calculate trends (slope and p-value) for general indices."""
        print("Calculating trends for general indices...")
        try:
            input_file_path = os.path.join(self.config.general_dir, 'General Indices.xlsx')
            
            if not os.path.exists(input_file_path):
                print(f"Missing general indices file: {input_file_path}")
                print("Please run calculate_general_indices first.")
                return False
                
            output_file_path = os.path.join(self.config.general_dir, 'Slope and P-value.xlsx')
            
            # Read the Excel file
            xls = pd.ExcelFile(input_file_path)
            
            with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
                # Process each sheet
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(input_file_path, sheet_name=sheet_name)
                    results = []
                    
                    # Calculate trends for each station
                    for station_column in df.columns[1:]:
                        year = df['Year']
                        temperature_data = df[station_column]
                        
                        # Calculate Sen's slope
                        sen_slope_value = self._sen_slope(temperature_data)
                        
                        # Calculate Mann-Kendall p-value
                        tau, p_value = kendalltau(year, temperature_data)
                        
                        # Append results
                        results.append([station_column, sen_slope_value, p_value])
                    
                    # Create and save results
                    result_df = pd.DataFrame(results, columns=["Station", "Sen's Slope", "Mann-Kendall p-value"])
                    result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print("Trend calculation completed.")
            return True
        except Exception as e:
            print(f"Error calculating trends: {e}")
            return False
    
    def _sen_slope(self, y):
        """Calculate Sen's slope.
        
        Args:
            y (array-like): Data values
            
        Returns:
            float: Sen's slope value
        """
        n = len(y)
        xi = np.arange(1, n + 1)
        slopes = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                slopes[i] += np.sign(y[j] - y[i])
        
        return np.median(slopes) / (np.median(xi) - xi[0])
    
    def calculate_mean_indices(self):
        """Calculate mean values for general indices."""
        print("Calculating mean values for general indices...")
        try:
            # Check if general indices file exists
            input_file_path = os.path.join(self.config.general_dir, 'General Indices.xlsx')
            if not os.path.exists(input_file_path):
                print(f"Missing general indices file: {input_file_path}")
                print("Please run calculate_general_indices first.")
                return False
                
            # Load Excel file
            file = pd.ExcelFile(input_file_path)
            
            # Process each sheet
            for sheet_name in file.sheet_names:
                # Read sheet, skipping Year column
                df = pd.read_excel(file, sheet_name=sheet_name, usecols=lambda x: x != 'Year')
                
                # Calculate mean for each column
                mean_series = df.mean()
                
                # Create result DataFrame
                result_df = pd.DataFrame({'Station': mean_series.index, f'{sheet_name}_mean': mean_series.values})
                
                # Save result
                result_df.to_csv(os.path.join(self.config.general_dir, f'{sheet_name}_mean.csv'), 
                                 header=True, index=False)
            
            print("Mean indices calculation completed.")
            return True
        except Exception as e:
            print(f"Error calculating mean indices: {e}")
            return False
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns of heatwave indices."""
        print("Analyzing temporal patterns...")
        try:
            # Define Excel files to process
            excel_files = [
                os.path.join(self.config.heatwave_dir, 'Output_average.xlsx'),
                os.path.join(self.config.heatwave_dir, 'Output_day.xlsx'),
                os.path.join(self.config.heatwave_dir, 'Output_day&night.xlsx'),
                os.path.join(self.config.heatwave_dir, 'Output_night.xlsx')
            ]
            
            # Check if files exist
            missing_files = [f for f in excel_files if not os.path.exists(f)]
            if missing_files:
                print(f"Missing heatwave indices files: {missing_files}")
                print("Please run calculate_heatwave_indices first.")
                return False
            
            for file_path in excel_files:
                # Process each Excel file
                self._process_temporal_data(file_path)
            
            # Combine and summarize results
            self._combine_temporal_results()
            
            print("Temporal pattern analysis completed.")
            return True
        except Exception as e:
            print(f"Error analyzing temporal patterns: {e}")
            return False
    
    def _process_temporal_data(self, file_path):
        """Process a single Excel file for temporal analysis.
        
        Args:
            file_path (str): Path to Excel file
        """
        # Get output file name
        output_file = os.path.join(self.config.temporal_dir, 
                                 f'summary_{os.path.basename(file_path)}.csv')
        
        # Load Excel file
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        
        # Parse sheets into DataFrames
        dataframes = {sheet_name: xls.parse(sheet_name) for sheet_name in sheet_names}
        
        # Process each sheet
        results = {}
        for sheet_name, df in dataframes.items():
            max_values = {}
            for col in df.columns:
                if col != 'Year':
                    # Find the most recent year with the maximum value
                    max_value = df[col].max()
                    max_indices = np.where(df[col] == max_value)[0]
                    max_years = df.iloc[max_indices]['Year']
                    most_recent_year = max_years.max()
                    max_values[col] = most_recent_year
            results[sheet_name] = max_values
        
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(results)
        summary_df['Station'] = summary_df.index
        summary_df.to_csv(output_file, index=False)
    
    def _combine_temporal_results(self):
        """Combine and summarize temporal analysis results."""
        # Read summary files
        file1 = pd.read_csv(os.path.join(self.config.temporal_dir, 'summary_Output_average.xlsx.csv'))
        file2 = pd.read_csv(os.path.join(self.config.temporal_dir, 'summary_Output_day.xlsx.csv'))
        file3 = pd.read_csv(os.path.join(self.config.temporal_dir, 'summary_Output_day&night.xlsx.csv'))
        file4 = pd.read_csv(os.path.join(self.config.temporal_dir, 'summary_Output_night.xlsx.csv'))
        
        # Rename columns to distinguish scenarios
        file2 = file2.rename(columns={'HWN': 'HWN-Day', 'HWTD': 'HWTD-Day', 'HWLD': 'HWLD-Day',
                                     'HWMT': 'HWMT-Day', 'HDT': 'HDT-Day'})
        file4 = file4.rename(columns={'HWN': 'HWN-Night', 'HWTD': 'HWTD-Night', 'HWLD': 'HWLD-Night',
                                     'HWMT': 'HWMT-Night', 'HDT': 'HDT-Night'})
        file3 = file3.rename(columns={'HWN': 'HWN-Day&Night', 'HWTD': 'HWTD-Day&Night', 'HWLD': 'HWLD-Day&Night',
                                     'HWMT': 'HWMT-Day&Night', 'HDT': 'HDT-Day&Night'})
        file1 = file1.rename(columns={'HWN': 'HWN-Avg', 'HWTD': 'HWTD-Avg', 'HWLD': 'HWLD-Avg',
                                     'HWMT': 'HWMT-Avg', 'HDT': 'HDT-Avg'})
        
        # Merge files
        merged_df = pd.merge(file1, file2, on='Station', how='inner')
        merged_df = pd.merge(merged_df, file3, on='Station', how='inner')
        merged_df = pd.merge(merged_df, file4, on='Station', how='inner')
        
        # Save combined results
        merged_df.to_csv(os.path.join(self.config.temporal_dir, 'Summary_All.csv'), index=False)


class Visualizer:
    """Creates visualizations from climate data and analysis results."""
    
    def __init__(self, config, data_loader):
        """Initialize visualizer.
        
        Args:
            config (Config): Configuration instance
            data_loader (DataLoader): Data loader with GeoDataFrames
        """
        self.config = config
        self.data_loader = data_loader
    
    def create_heatwave_maps(self):
        """Create maps of heatwave indices."""
        print("Creating heatwave index maps...")
        try:
            # Check if mean/max files exist
            csv_files_mean = ['Mean_values_night.csv', 'Mean_values_day.csv', 
                            'Mean_values_average.csv', 'Mean_values_day&night.csv']
            
            csv_files_max = ['Max_values_night.csv', 'Max_values_day.csv', 
                           'Max_values_average.csv', 'Max_values_day&night.csv']
            
            missing_files = [f for f in csv_files_mean + csv_files_max 
                          if not os.path.exists(os.path.join(self.config.mean_max_dir, f))]
            
            if missing_files:
                print(f"Missing mean/max values files: {missing_files}")
                print("Please run calculate_mean_max_values first.")
                return False
            
            # Columns to plot
            columns_to_plot = ['HWN', 'HWMT', 'HDT', 'HWLD', 'HWTD']
            
            # Mapping of CSV file names to subplot titles
            subplot_titles_mean = {
                'Mean_values_night.csv': 'A)  Night',
                'Mean_values_day.csv': 'B)  Day',
                'Mean_values_average.csv': 'C)  Average',
                'Mean_values_day&night.csv': 'D)  Day&Night'
            }
            
            # Create maps for mean values
            for column in columns_to_plot:
                print(f"Creating maps for {column} (Mean)")
                fig, axes = plt.subplots(4, 1, figsize=(7, 8))
                fig.subplots_adjust(hspace=0.5)
                axes = axes.flatten()
                fig.suptitle(f'{column} (Mean)', fontsize=16, x=0.5, y=1)
                
                for i, csv_file in enumerate(csv_files_mean):
                    # Read CSV data
                    csv_df = pd.read_csv(os.path.join(self.config.mean_max_dir, csv_file))
                    
                    # Merge with GeoDataFrame
                    merged_gdf = self.data_loader.gdf.merge(
                        csv_df, left_on='sequence', right_on='Station', how='inner')
                    
                    # Plot on map
                    ax = axes[i]
                    merged_gdf.plot(
                        column=column,
                        cmap='BrBG',
                        edgecolor='black',
                        linewidth=0.05,
                        ax=ax,
                        legend=True,
                        aspect=1
                    )
                    
                    ax.set_title(f'{subplot_titles_mean[csv_file]}')
                
                # Save figure
                plt.tight_layout()
                fig.savefig(os.path.join(self.config.figures_dir, f'{column}_mean.png'), 
                          dpi=1000, bbox_inches='tight')
                plt.close()
            
            # Mapping of CSV file names to subplot titles for max values
            subplot_titles_max = {
                'Max_values_night.csv': 'A)  Night',
                'Max_values_day.csv': 'B)  Day',
                'Max_values_average.csv': 'C)  Average',
                'Max_values_day&night.csv': 'D)  Day&Night'
            }
            
            # Create maps for max values
            for column in columns_to_plot:
                print(f"Creating maps for {column} (Max)")
                fig, axes = plt.subplots(4, 1, figsize=(7, 8))
                fig.subplots_adjust(hspace=0.5)
                axes = axes.flatten()
                fig.suptitle(f'{column} (Max)', fontsize=16, x=0.5, y=1)
                
                for i, csv_file in enumerate(csv_files_max):
                    # Read CSV data
                    csv_df = pd.read_csv(os.path.join(self.config.mean_max_dir, csv_file))
                    
                    # Merge with GeoDataFrame
                    merged_gdf = self.data_loader.gdf.merge(
                        csv_df, left_on='sequence', right_on='Station', how='inner')
                    
                    # Plot on map
                    ax = axes[i]
                    merged_gdf.plot(
                        column=column,
                        cmap='BrBG',
                        edgecolor='black',
                        linewidth=0.05,
                        ax=ax,
                        legend=True,
                        aspect=1
                    )
                    
                    ax.set_title(f'{subplot_titles_max[csv_file]}')
                
                # Save figure
                plt.tight_layout()
                fig.savefig(os.path.join(self.config.figures_dir, f'{column}_max.png'), 
                          dpi=1000, bbox_inches='tight')
                plt.close()
            
            print("Heatwave index maps created.")
            return True
        except Exception as e:
            print(f"Error creating heatwave maps: {e}")
            return False
    
    def create_hourly_plots(self):
        """Create hourly analysis plots."""
        print("Creating hourly analysis plots...")
        try:
            # Check if hourly files exist
            hourly_files = [f'Hourly_{dtype}.csv' for dtype in ['max', 'min', 'mean']]
            missing_files = [f for f in hourly_files 
                          if not os.path.exists(os.path.join(self.config.shifting_dir, f))]
            
            if missing_files:
                print(f"Missing hourly data files: {missing_files}")
                print("Please run combine_and_average_data first.")
                return False
                
            for data_type in ['max', 'min', 'mean']:
                print(f"Creating hourly plots for {data_type} data")
                self._process_and_plot_hourly_data(data_type)
            
            print("Hourly plots created.")
            return True
        except Exception as e:
            print(f"Error creating hourly plots: {e}")
            return False
    
    def _process_and_plot_hourly_data(self, data_type):
        """Process and plot hourly data.
        
        Args:
            data_type (str): Type of data ('max', 'min', or 'mean')
        """
        # Read data
        file_hourly = pd.read_csv(os.path.join(self.config.shifting_dir, f'Hourly_{data_type}.csv'))
        
        # Check if Day and Month columns exist
        if 'Day' in file_hourly.columns:
            del file_hourly['Day']
        if 'Month' in file_hourly.columns:
            del file_hourly['Month']
        
        df = file_hourly
        station_numbers = [col for col in df.columns if col not in ['Year', 'Hour', 'Cities']]
        
        # Create column list
        columns_to_keep = ['Year', 'Hour'] + station_numbers
        
        # Filter DataFrame
        filtered_df = df[columns_to_keep]
        
        # Calculate aggregation
        if data_type == 'max':
            filtered_df['Cities'] = filtered_df.iloc[:, 2:].max(axis=1)
        elif data_type == 'min':
            filtered_df['Cities'] = filtered_df.iloc[:, 2:].min(axis=1)
        elif data_type == 'mean':
            filtered_df['Cities'] = filtered_df.iloc[:, 2:].mean(axis=1)
        
        # Create result DataFrame
        result_df = filtered_df[['Year', 'Hour', 'Cities']]
        
        # Save result
        result_df.to_csv(os.path.join(self.config.shifting_dir, f'Hourly_{data_type}.csv'), index=False)
        
        # Read and prepare for plotting
        result_df = pd.read_csv(os.path.join(self.config.shifting_dir, f'Hourly_{data_type}.csv'))
        result_df = result_df.drop_duplicates(subset=['Hour', 'Year'])
        
        # Convert Hour to proper time format
        try:
            result_df['Hour'] = pd.to_datetime(result_df['Hour'], format='%H')
            result_df['Hour'] = result_df['Hour'] - pd.Timedelta(hours=6)
            result_df['Hour'] = result_df['Hour'].dt.strftime('%H:%M')
        except:
            # If Hour is already in string format, just use it as is
            pass
            
        result_df['Year'] = result_df['Year'].astype(int)
        
        # Create pivot table
        pivot_table = result_df.pivot(index='Hour', columns='Year', values='Cities')
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, cmap='coolwarm')
        
        # Customize plot
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('Hour', fontsize=16)
        plt.title(f'Heatmap of Cities by Year and Hour ({data_type})')
        
        # Save plot
        plt.savefig(os.path.join(self.config.figures_dir, f'heatmap_hourly_{data_type}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    def create_monthly_plots(self):
        """Create monthly analysis plots."""
        print("Creating monthly analysis plots...")
        try:
            # Check if monthly files exist
            monthly_files = [f'Monthly_{dtype}.csv' for dtype in ['max', 'min', 'mean']]
            missing_files = [f for f in monthly_files 
                          if not os.path.exists(os.path.join(self.config.shifting_dir, f))]
            
            if missing_files:
                print(f"Missing monthly data files: {missing_files}")
                print("Please run combine_and_average_data first.")
                return False
                
            for data_type in ['max', 'min', 'mean']:
                print(f"Creating monthly plots for {data_type} data")
                self._process_and_plot_monthly_data(data_type)
            
            print("Monthly plots created.")
            return True
        except Exception as e:
            print(f"Error creating monthly plots: {e}")
            return False
    
    def _process_and_plot_monthly_data(self, data_type):
        """Process and plot monthly data.
        
        Args:
            data_type (str): Type of data ('max', 'min', or 'mean')
        """
        # Read data
        file_monthly = pd.read_csv(os.path.join(self.config.shifting_dir, f'Monthly_{data_type}.csv'))
        
        # Check if Day and Hour columns exist
        if 'Day' in file_monthly.columns:
            del file_monthly['Day']
        if 'Hour' in file_monthly.columns:
            del file_monthly['Hour']
        
        df = file_monthly
        station_numbers = [col for col in df.columns if col not in ['Year', 'Month', 'Cities']]
        
        # Create column list
        columns_to_keep = ['Year', 'Month'] + station_numbers
        
        # Filter DataFrame
        filtered_df = df[columns_to_keep]
        
        # Calculate aggregation
        if data_type == 'max':
            filtered_df['Cities'] = filtered_df.iloc[:, 2:].max(axis=1)
        elif data_type == 'min':
            filtered_df['Cities'] = filtered_df.iloc[:, 2:].min(axis=1)
        elif data_type == 'mean':
            filtered_df['Cities'] = filtered_df.iloc[:, 2:].mean(axis=1)
        
        # Create result DataFrame
        result_df = filtered_df[['Year', 'Month', 'Cities']]
        
        # Save result
        result_df.to_csv(os.path.join(self.config.shifting_dir, f'Hourly_{data_type}.csv'), index=False)
        
        # Read and prepare for plotting
        result_df = pd.read_csv(os.path.join(self.config.shifting_dir, f'Hourly_{data_type}.csv'))
        
        # Convert Month to proper format
        try:
            result_df['Month'] = pd.to_datetime(result_df['Month'], format='%m')
            result_df['Month'] = result_df['Month'].dt.strftime('%b')
        except:
            # If Month is already in string format, just use it as is
            pass
            
        result_df = result_df.drop_duplicates(subset=['Month', 'Year'])
        result_df['Year'] = result_df['Year'].astype(int)
        
        # Create custom month order
        custom_order = list(calendar.month_abbr[1:])
        custom_order.reverse()
        
        # Order months
        result_df['Month'] = pd.Categorical(result_df['Month'], categories=custom_order, ordered=True)
        pivot_table = result_df.pivot_table(index='Month', columns='Year', values='Cities').iloc[::-1]
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_table, cmap='Reds')
        
        # Customize plot
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('Month', fontsize=16)
        plt.title(f'Heatmap of Cities by Year and Month ({data_type})')
        
        # Save plot
        plt.savefig(os.path.join(self.config.figures_dir, f'heatmap_monthly_{data_type}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    def create_general_indices_map(self):
        """Create map of general temperature indices."""
        print("Creating general indices map...")
        try:
            # Check if mean indices files exist
            sheet_names = ['Txx', 'Tnn', 'Txn', 'Tnx']
            missing_files = [f for f in [f'{name}_mean.csv' for name in sheet_names] 
                          if not os.path.exists(os.path.join(self.config.general_dir, f))]
            
            if missing_files:
                print(f"Missing mean indices files: {missing_files}")
                print("Please run calculate_mean_indices first.")
                return False
                
            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(12, 16))
            
            for ax in axes:
                ax.set_aspect('equal')
            
            plt.subplots_adjust(hspace=0.4)
            
            # Labels for subplots
            subplot_labels = ['A)', 'B)', 'C)', 'D)']
            
            # Plot each index
            for i, sheet_name in enumerate(sheet_names):
                # Read index data
                csv_path = os.path.join(self.config.general_dir, f'{sheet_name}_mean.csv')
                csv_data = pd.read_csv(csv_path)
                
                # Merge with GeoDataFrame
                merged_gdf = self.data_loader.gdf.merge(csv_data, left_on='sequence', right_on='Station')
                
                # Create plot
                ax = axes[i]
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=-0.4)
                merged_gdf.plot(column=f'{sheet_name}_mean', cmap='coolwarm', cax=cax, ax=ax, legend=True)
                ax.set_title(f'{subplot_labels[i]} {sheet_name}', fontsize=16)
                
                # Plot boundaries
                self.data_loader.gdf.boundary.plot(ax=ax, linewidth=0.3, color='brown')
            
            # Save figure
            plt.savefig(os.path.join(self.config.figures_dir, 'General_Indices.png'), 
                       bbox_inches='tight', dpi=600)
            plt.close()
            
            print("General indices map created.")
            return True
        except Exception as e:
            print(f"Error creating general indices map: {e}")
            return False
    
    def create_trend_maps(self):
        """Create maps of trend analysis results."""
        print("Creating trend analysis maps...")
        try:
            # Check if trend file exists
            output_file_path = os.path.join(self.config.general_dir, 'Slope and P-value.xlsx')
            if not os.path.exists(output_file_path):
                print(f"Missing trend file: {output_file_path}")
                print("Please run calculate_trends first.")
                return False
                
            # Load trend data
            xls = pd.ExcelFile(output_file_path)
            
            # Initialize vmin and vmax
            vmin = float('inf')
            vmax = float('-inf')
            
            # Set up subplots
            num_sheets = len(xls.sheet_names)
            num_rows = 4
            num_cols = 1
            fig_width = 8
            fig_height = 12
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), constrained_layout=False)
            plt.subplots_adjust(hspace=0.4)
            
            # Create maps for each sheet
            for i, sheet_name in enumerate(xls.sheet_names):
                # Read sheet data
                df = pd.read_excel(output_file_path, sheet_name=sheet_name)
                
                # Merge with GeoDataFrame
                merged_gdf = self.data_loader.gdf.merge(df, left_on="sequence", right_on="Station", how="left")
                
                # Select subplot
                ax = axes[i]
                
                # Set up colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                
                # Check for single value
                if len(merged_gdf["Sen's Slope"].unique()) == 1:
                    single_value = merged_gdf["Sen's Slope"].values[0]
                    vmin = single_value
                    vmax = single_value
                else:
                    vmin = min(vmin, merged_gdf["Sen's Slope"].min())
                    vmax = max(vmax, merged_gdf["Sen's Slope"].max())
                
                # Plot data
                merged_gdf.plot(column="Sen's Slope", ax=ax, legend=True, cax=cax, 
                               cmap="BrBG", vmin=vmin, vmax=vmax, linewidth=0.1)
                
                # Plot boundaries
                self.data_loader.gdf.boundary.plot(ax=ax, linewidth=0.3, color='brown')
                
                # Hatch significant areas
                merged_gdf[merged_gdf["Mann-Kendall p-value"] < 0.10].plot(
                    ax=ax, hatch="//", facecolor="none", edgecolor="k", aspect=1)
                
                # Set title
                ax.set_title(f'{chr(65 + i)}) Trend of {sheet_name}', fontsize=16)
            
            # Remove extra subplots
            for i in range(num_sheets, num_rows):
                fig.delaxes(axes[i])
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.figures_dir, 'Slope-General-Indices.png'), dpi=600)
            plt.close()
            
            print("Trend analysis maps created.")
            return True
        except Exception as e:
            print(f"Error creating trend maps: {e}")
            return False
    
    def create_temporal_visualizations(self):
        """Create visualizations of temporal patterns."""
        print("Creating temporal pattern visualizations...")
        try:
            # Check if summary file exists
            summary_file = os.path.join(self.config.temporal_dir, 'Summary_All.csv')
            if not os.path.exists(summary_file):
                print(f"Missing summary file: {summary_file}")
                print("Please run analyze_temporal_patterns first.")
                return False
                
            # Read data
            csv_file = pd.read_csv(summary_file)
            
            # Merge with GeoDataFrame
            merged_file = self.data_loader.gdf_file.merge(
                csv_file, left_on='sequence', right_on='Station', how='inner')
            
            # Set index
            merged_file.set_index('sequence', inplace=True)
            
            # Select columns for heatmap
            columns = ['HWN', 'HWTD']
            new_columns = ([column + '-Day' for column in columns] + 
                          [column + '-Night' for column in columns] + 
                          [column + '-Day&Night' for column in columns] + 
                          [column + '-Avg' for column in columns])
            
            # Check if all columns exist
            missing_columns = [col for col in new_columns if col not in merged_file.columns]
            if missing_columns:
                print(f"Missing columns in the merged file: {missing_columns}")
                return False
                
            # Modify labels
            y_tick_labels = [label.replace('-Day&Night', '-DN').replace('-Night', '-N')
                            .replace('-Day', '-D').replace('-Avg', '-A') 
                            for label in new_columns]
            
            # Create heatmap
            plt.figure(figsize=(10, 3))
            heatmap = sns.heatmap(merged_file[new_columns].T, cmap='tab20c', 
                                annot=False, vmin=1978, vmax=2022, fmt='d')
            
            # Remove x-ticks
            heatmap.set_xticks([])
            
            # Set y-tick labels
            heatmap.set_yticklabels(y_tick_labels)
            
            # Customize plot
            plt.title('Maximum Heatwave Indices and Cities by Year', fontsize=16)
            plt.xlabel('Cities', fontsize=16)
            
            # Set colorbar ticks
            cbar = heatmap.collections[0].colorbar
            cbar.set_ticks(range(1978, 2023, 11))
            cbar.set_ticklabels([str(year) for year in range(1978, 2023, 11)])
            
            # Save figure
            plt.savefig(os.path.join(self.config.figures_dir, 'Temporal_Heatmap.png'), dpi=1000)
            plt.close()
            
            # Create spatial plots
            columns_to_plot = new_columns
            title_labels = y_tick_labels
            
            # Set up grid
            num_rows, num_cols = 2, 4
            fig = plt.figure(figsize=(10, 4))
            fig.suptitle("Maximum Heatwave Indices by Year", fontsize=16)
            gs = gridspec.GridSpec(num_rows, num_cols, width_ratios=[1] * num_cols, 
                                 height_ratios=[1] * num_rows)
            
            # Create subplots
            ax = [plt.subplot(gs[i]) for i in range(num_rows * num_cols)]
            for axis in ax:
                axis.set_frame_on(True)
            
            # Plot each column
            for i, (column, title) in enumerate(zip(columns_to_plot, title_labels)):
                merged_file.plot(column=column, ax=ax[i], legend=False, cmap='tab20c', 
                               vmin=1978, vmax=2023)
                ax[i].set_title(title)
                ax[i].axis('off')
            
            # Adjust layout
            plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08, 
                              wspace=0.05, hspace=0.15)
            
            # Add colorbar
            cax = fig.add_axes([0.15, 0.06, 0.7, 0.02])
            cbar_values = [1978, 1989, 2000, 2011, 2022]
            sm = plt.cm.ScalarMappable(cmap='tab20c', norm=plt.Normalize(vmin=1978, vmax=2022))
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
            cbar.set_ticks(cbar_values)
            
            # Save figure
            plt.savefig(os.path.join(self.config.figures_dir, 'Temporal_Spatial.png'), dpi=1000)
            plt.close()
            
            print("Temporal visualizations created.")
            return True
        except Exception as e:
            print(f"Error creating temporal visualizations: {e}")
            return False


class ClimateAnalysisPipeline:
    """Main class to orchestrate the climate analysis pipeline."""
    
    def __init__(self, base_dir=".", states=None):
        """Initialize the pipeline.
        
        Args:
            base_dir (str): Base directory for all data and outputs
            states (list/str, optional): States to include in the analysis. 
                - List of state names for specific states
                - "ALL" for all CONUS states
                - None for default (Mississippi)
        """
        # Initialize configuration
        self.config = Config(base_dir, states)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.data_processor = DataProcessor(self.config, self.data_loader)
        self.heatwave_analyzer = None  # Will be initialized after data is loaded
        self.statistical_analyzer = None  # Will be initialized after data is loaded
        self.visualizer = None  # Will be initialized after data is loaded
        
        # Define pipeline steps in order
        self.pipeline_steps = [
            ('load_shapefiles', self._load_shapefiles),
            ('calculate_temperature_metrics', self._calculate_temperature_metrics),
            ('calculate_heatwave_indices', self._calculate_heatwave_indices),
            ('calculate_mean_max_values', self._calculate_mean_max_values),
            ('calculate_general_indices', self._calculate_general_indices),
            ('combine_and_average_data', self._combine_and_average_data),
            ('calculate_trends', self._calculate_trends),
            ('calculate_mean_indices', self._calculate_mean_indices),
            ('analyze_temporal_patterns', self._analyze_temporal_patterns),
            ('create_heatwave_maps', self._create_heatwave_maps),
            ('create_hourly_plots', self._create_hourly_plots),
            ('create_monthly_plots', self._create_monthly_plots),
            ('create_general_indices_map', self._create_general_indices_map),
            ('create_trend_maps', self._create_trend_maps),
            ('create_temporal_visualizations', self._create_temporal_visualizations)
        ]
    
    def _load_shapefiles(self):
        """Load shapefiles and initialize components."""
        if not self.data_loader.load_shapefiles():
            return False
        
        # Initialize remaining components
        self.heatwave_analyzer = HeatwaveAnalyzer(self.config, self.data_loader)
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.visualizer = Visualizer(self.config, self.data_loader)
        return True
        
    def _calculate_temperature_metrics(self):
        return self.data_processor.calculate_temperature_metrics()
        
    def _calculate_heatwave_indices(self):
        return self.heatwave_analyzer.calculate_heatwave_indices()
        
    def _calculate_mean_max_values(self):
        return self.heatwave_analyzer.calculate_mean_max_values()
        
    def _calculate_general_indices(self):
        return self.heatwave_analyzer.calculate_general_indices()
        
    def _combine_and_average_data(self):
        return self.data_processor.combine_and_average_data()
        
    def _calculate_trends(self):
        return self.statistical_analyzer.calculate_trends()
        
    def _calculate_mean_indices(self):
        return self.statistical_analyzer.calculate_mean_indices()
        
    def _analyze_temporal_patterns(self):
        return self.statistical_analyzer.analyze_temporal_patterns()
        
    def _create_heatwave_maps(self):
        return self.visualizer.create_heatwave_maps()
        
    def _create_hourly_plots(self):
        return self.visualizer.create_hourly_plots()
        
    def _create_monthly_plots(self):
        return self.visualizer.create_monthly_plots()
        
    def _create_general_indices_map(self):
        return self.visualizer.create_general_indices_map()
        
    def _create_trend_maps(self):
        return self.visualizer.create_trend_maps()
        
    def _create_temporal_visualizations(self):
        return self.visualizer.create_temporal_visualizations()
    
    def run(self):
        """Run the complete analysis pipeline."""
        print("\n" + "=" * 80)
        print("Starting ERA5 Climate Analysis Pipeline")
        print("=" * 80 + "\n")
        
        for step_name, step_func in self.pipeline_steps:
            print(f"Running step: {step_name}")
            if not step_func():
                print(f"Step {step_name} failed. Exiting.")
                return False
        
        print("\n" + "=" * 80)
        print("ERA5 Climate Analysis Pipeline completed successfully!")
        print("=" * 80 + "\n")
        
        return True
    
    def run_from_step(self, start_step):
        """Run the pipeline starting from a specific step.
        
        Args:
            start_step (str): Name of the step to start from
            
        Returns:
            bool: True if all steps completed successfully, False otherwise
        """
        print("\n" + "=" * 80)
        print(f"Starting ERA5 Climate Analysis Pipeline from step: {start_step}")
        print("=" * 80 + "\n")
        
        # Always run load_shapefiles first if not starting with it
        if start_step != 'load_shapefiles':
            print("Running prerequisite step: load_shapefiles")
            if not self._load_shapefiles():
                print("Failed to load shapefiles. Exiting.")
                return False
        
        # Find the starting index
        start_index = -1
        for i, (step_name, _) in enumerate(self.pipeline_steps):
            if step_name == start_step:
                start_index = i
                break
        
        if start_index == -1:
            print(f"Unknown step: {start_step}")
            print(f"Available steps: {[name for name, _ in self.pipeline_steps]}")
            return False
        
        # Run from the starting step onwards
        for step_name, step_func in self.pipeline_steps[start_index:]:
            print(f"Running step: {step_name}")
            if not step_func():
                print(f"Step {step_name} failed. Exiting.")
                return False
        
        print("\n" + "=" * 80)
        print("ERA5 Climate Analysis Pipeline completed successfully!")
        print("=" * 80 + "\n")
        
        return True
    
    def run_range(self, start_step, end_step=None):
        """Run a range of steps in the pipeline.
        
        Args:
            start_step (str): Name of the step to start from
            end_step (str, optional): Name of the step to end at (inclusive)
            
        Returns:
            bool: True if all steps completed successfully, False otherwise
        """
        print("\n" + "=" * 80)
        if end_step:
            print(f"Running ERA5 Climate Analysis Pipeline from {start_step} to {end_step}")
        else:
            print(f"Running ERA5 Climate Analysis Pipeline from {start_step} to end")
        print("=" * 80 + "\n")
        
        # Always run load_shapefiles first if not starting with it
        if start_step != 'load_shapefiles':
            print("Running prerequisite step: load_shapefiles")
            if not self._load_shapefiles():
                print("Failed to load shapefiles. Exiting.")
                return False
        
        # Find the starting and ending indices
        start_index = -1
        end_index = len(self.pipeline_steps) if end_step is None else -1
        
        for i, (step_name, _) in enumerate(self.pipeline_steps):
            if step_name == start_step:
                start_index = i
            if end_step and step_name == end_step:
                end_index = i + 1  # +1 to include the end step
        
        if start_index == -1:
            print(f"Unknown start step: {start_step}")
            print(f"Available steps: {[name for name, _ in self.pipeline_steps]}")
            return False
        
        if end_step and end_index == -1:
            print(f"Unknown end step: {end_step}")
            print(f"Available steps: {[name for name, _ in self.pipeline_steps]}")
            return False
        
        # Run the specified range of steps
        for step_name, step_func in self.pipeline_steps[start_index:end_index]:
            print(f"Running step: {step_name}")
            if not step_func():
                print(f"Step {step_name} failed. Exiting.")
                return False
        
        print("\n" + "=" * 80)
        print("ERA5 Climate Analysis Pipeline completed successfully!")
        print("=" * 80 + "\n")
        
        return True
    
    def run_selected_steps(self, steps):
        """Run only specified steps in the pipeline.
        
        Args:
            steps (list): List of step names to run
            
        Returns:
            bool: True if all steps completed successfully, False otherwise
        """
        print("\n" + "=" * 80)
        print(f"Running selected steps in ERA5 Climate Analysis Pipeline: {steps}")
        print("=" * 80 + "\n")
        
        # Always run load_shapefiles first
        if 'load_shapefiles' not in steps:
            print("Running prerequisite step: load_shapefiles")
            if not self._load_shapefiles():
                print("Failed to load shapefiles. Exiting.")
                return False
        
        # Create a map of step names to functions
        step_map = {name: func for name, func in self.pipeline_steps}
        
        # Validate all steps
        unknown_steps = [step for step in steps if step not in step_map]
        if unknown_steps:
            print(f"Unknown steps: {unknown_steps}")
            print(f"Available steps: {list(step_map.keys())}")
            return False
        
        # Run the specified steps
        for step in steps:
            print(f"Running step: {step}")
            func = step_map[step]
            if not func():
                print(f"Step {step} failed. Exiting.")
                return False
        
        print("\n" + "=" * 80)
        print("Selected steps completed successfully!")
        print("=" * 80 + "\n")
        
        return True

# Usage examples:
if __name__ == "__main__":
    # Example 1: Run for Mississippi (default)
    # pipeline = ClimateAnalysisPipeline()
    
    # Example 2: Run for specific states
    # pipeline = ClimateAnalysisPipeline(states=["Mississippi", "Alabama", "Georgia"])
    
    # Example 3: Run for all CONUS states
    # pipeline = ClimateAnalysisPipeline(states="ALL")
    
    # Choose how to run the pipeline
    # pipeline.run()  # Run full pipeline
    # pipeline.run_from_step('calculate_general_indices')  # Run from a specific step
    
    # Default: Mississippi only, starting from combine_and_average_data
    pipeline = ClimateAnalysisPipeline(states=["Mississippi", "Alabama"])
    pipeline.run()