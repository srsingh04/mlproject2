import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wormstats(csv_file_path):
    data = pd.read_csv(csv_file_path)
    
    def calculate_mapped_values(data):
        """
        Adds a new column to the DataFrame that maps the nth row to the formula:
        2 * n + floor((n - 1) / 900) * 5.5 * 3600.
    
        Args:
            data (pd.DataFrame): The input DataFrame.
    
        Returns:
            pd.DataFrame: Updated DataFrame with the new 'Mapped Value' column.
        """
        # Calculate row indices starting from 1
        n = data.index + 1  # Row index (n), starting from 1
        # Apply the formula (see iPad)
        data['Time elapsed (in sec)'] = 2 * n + np.floor((n - 1) / 900) * 5.5 * 3600
        data['Time elapsed (in hours)'] = data['Time elapsed (in sec)']/3600
        return data
    
    
    data['group'] = data.index.map(lambda x : np.floor((x)/900) )
    data = calculate_mapped_values(data)

    # Calculate differences in X, Y, and time
    data_dx = data['X'].diff()
    data_dy = data['Y'].diff()
    delta_time = data['Time elapsed (in sec)'].diff()
    
    distance = np.sqrt(data_dx**2 + data_dy**2)
    data['dist'] = distance
    # Calculate speed as distance divided by time
    data['EucSpeed'] = distance / delta_time

    condition = (data.index) % 900 == 0
    data['Changed Pixels'] = data['Changed Pixels'].shift(1)
    data.loc[condition, ['EucSpeed', 'Changed Pixels']] = None

    def find_death_row(data, change_col='Changed Pixels', threshold=5, window=10, persistence=0.9):
        """
        Find the first row where the worm is considered "dead."
        
        Args:
            data (pd.DataFrame): The worm's trajectory data.
            change_col (str): Column indicating changes in pixels.
            threshold (int): Threshold below which the worm is considered inactive.
            window (int): Number of subsequent rows to check for inactivity.
            persistence (float): Fraction of rows in the window that must satisfy the condition.
        
        Returns:
            int: Index of the death row or -1 if no such row is found.
        """
        for i in range(len(data) - window):
            # Check if the current row satisfies the condition
            if data.loc[i, change_col] < threshold:
                # Check the persistence condition in the window
                subsequent = data[change_col].iloc[i:i+window]
                if (subsequent < threshold).sum() >= persistence * window:
                    return i
        return -1  # Return -1 if no death row is found
    
    death_row = find_death_row(data, change_col='Changed Pixels', threshold=5, window=500, persistence=0.9)
    final_age = data['Time elapsed (in hours)'].iloc[death_row]
    
    data = data.dropna(subset=['X', 'Y', 'EucSpeed'])

    def add_derived_columns(data, speed_threshold=0.1):
        """
        Adds derived columns to the worm trajectory CSV.
    
        Args:
            csv_file (str): Path to the input CSV file.
            output_file (str): Path to save the updated CSV file (optional).
            speed_threshold (float): Threshold for stationary flag.
    
        Returns:
            pd.DataFrame: DataFrame with added columns.
        """
        # Add derived columns
        data['Change in Speed'] = data['EucSpeed'].diff().abs().fillna(0)
        data['Change in Pixels'] = data['Changed Pixels'].diff().fillna(0)
        data['Instantaneous Distance'] = np.sqrt(
            (data['X'].diff() ** 2) + (data['Y'].diff() ** 2)
        ).fillna(0)
        data['Total Distance'] = data['Instantaneous Distance'].cumsum()
        data['Angle'] = np.arctan2(data['Y'].diff(), data['X'].diff()).fillna(0)
        data['Angular Change'] = data['Angle'].diff().abs().fillna(0)
        data['Stationary'] = (data['EucSpeed'] < speed_threshold).astype(int)
        data['Cumulative Stationary Time'] = data['Stationary'].cumsum()
    
        return data
    
    #output_file = 'updated_worm_file.csv'  # Optional output file
    data = add_derived_columns(data)
    data.reset_index(inplace = True)

    THEC = data.EucSpeed.mean()*1.2
    w = 10
    data['Roaming2'] = data['EucSpeed'].rolling(window=w, center=True).mean()
    data['RoamingIndic'] = (data['Roaming2'] > THEC).astype(int)
    #data['Roaming Fraction2'] = data['RoamingIndic'].rolling(window=10, center=True, min_periods=1).mean()

    roaming_frac=data['RoamingIndic'].sum()*2
    total_time = data.shape[0]*2

    average_speed = data.EucSpeed.mean()
    average_distance_per_frame = data['dist'].mean()
    maximal_stpwsdistance_travalled = data['dist'].max()
    maximal_distance_travalled = data['Total Distance'].max()
    average_change_in_pixels = data['Change in Pixels'].mean()
    average_angular_speed= data['Angular Change'].mean()
    FRF = roaming_frac/total_time
    final_age

    return [average_speed, average_distance_per_frame, maximal_stpwsdistance_travalled ,  maximal_distance_travalled, average_change_in_pixels, average_angular_speed, FRF, final_age]