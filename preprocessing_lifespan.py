import pandas as pd
import numpy as np

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
        data = data.reset_index(drop=True)
        for i in range(len(data) - window):
            # Check if the current row satisfies the condition
            if data.loc[i, change_col] < threshold:
                # Check the persistence condition in the window
                subsequent = data[change_col].iloc[i:i+window]
                if (subsequent < threshold).sum() >= persistence * window:
                    return i
        return -1  # Return -1 if no death row is found


def wormstats(csv_file_path):
    # 1. Read CSV data
    data = pd.read_csv(csv_file_path)

    # 2. Add time-related columns (before dropping NaNs to preserve timestamp integrity)
    n = data.index + 1  # Row indices (1-based)
    data['Time elapsed (in sec)'] = 2 * n + np.floor((n - 1) / 900) * 5.5 * 3600
    data['Time elapsed (in hours)'] = data['Time elapsed (in sec)'] / 3600
    
    # 3. Drop rows where X or Y are NaN (clean data step)
    data = data.dropna(subset=['X', 'Y'])
    
    # 4. Calculate distance and speed
    data_dx = data['X'].diff()
    data_dy = data['Y'].diff()
    delta_time = data['Time elapsed (in sec)'].diff()
    distance = np.sqrt(data_dx**2 + data_dy**2)
    data['dist'] = distance
    data['EucSpeed'] = distance / delta_time
    
    # 5. Group rows into 900-row chunks (adjusted for dropped rows)
    data['group'] = np.floor(data.index / 900)
    
    # 6. Finding the death row
    death_row = find_death_row(data, change_col='Changed Pixels', threshold=5, window=500, persistence=0.9)
    if death_row is not None and 0 <= death_row < len(data):
        final_age = data['Time elapsed (in hours)'].iloc[death_row]
    else:
        final_age = data['Time elapsed (in hours)'].iloc[-1]
    # final_age = data['Time elapsed (in hours)'].iloc[death_row]

    # 7. Adding derived features
    data['Change in Speed'] = data['EucSpeed'].diff().abs().fillna(0)
    data['Change in Pixels'] = data['Changed Pixels'].diff().fillna(0)
    data['Instantaneous Distance'] = np.sqrt((data['X'].diff() ** 2) + (data['Y'].diff() ** 2)).fillna(0)
    data['Total Distance'] = data['Instantaneous Distance'].cumsum()
    data['Angle'] = np.arctan2(data['Y'].diff(), data['X'].diff()).fillna(0)
    data['Angular Change'] = data['Angle'].diff().abs().fillna(0)
    data['Stationary'] = (data['EucSpeed'] < 0.01).astype(int)  # Example threshold for stationary
    data['Cumulative Stationary Time'] = data['Stationary'].cumsum()

    # 8. Roaming indicator
    w = 10  # Window size for rolling mean
    THEC = 1.2 * data['EucSpeed'].mean()
    data['Roaming2'] = data['EucSpeed'].rolling(window=w, center=True).mean()
    data['RoamingIndic'] = (data['Roaming2'] > THEC).astype(int)

    # 9. Aggregating by group
    grouped_df = pd.DataFrame({
        'Mean Speed': data.groupby('group')['EucSpeed'].mean(),
        'Mean Distance': data.groupby('group')['dist'].mean(),
        'Max Total Distance': data.groupby('group')['Total Distance'].max(),
        'Mean Change Pixels': data.groupby('group')['Change in Pixels'].mean(),
        'Mean Change Speed': data.groupby('group')['Change in Speed'].mean(),
        'Mean Angular Speed': data.groupby('group')['Angular Change'].mean(),
        'Final Time': data.groupby('group')['Time elapsed (in hours)'].apply(lambda x: x.iloc[-1]),
        'Roaming Fraction': data.groupby('group')['RoamingIndic'].mean(),
        'Speed Std Dev': data.groupby('group')['EucSpeed'].std()
    }).reset_index()

    # 10. Add lifespan
    grouped_df['lifespan'] = final_age

    return grouped_df
