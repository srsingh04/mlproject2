import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_time_units(frames):
    """
    Calculate elapsed time in seconds, minutes, hours, and days for each frame.

    Args:
    frames (array-like): Continuous frame numbers.

    Returns:
    pandas.DataFrame: A DataFrame with columns for seconds, minutes, hours, and days.

    To run:
    time_columns = calculate_time_units(data['continuous_frames'])
    data = pd.concat([data, time_columns], axis=1)
    """
    frames = np.array(frames)  # Ensure input is a numpy array for vectorized computation
    
    # Calculate elapsed time in seconds
    elapsed_seconds = 2 * frames
    # Convert to other units
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60
    elapsed_days = elapsed_hours / 24

    # Return as a DataFrame
    return pd.DataFrame({
        'time_seconds': elapsed_seconds,
        'time_minutes': elapsed_minutes,
        'time_hours': elapsed_hours,
        'time_days': elapsed_days
    })


def calculate_time_units_atr(frames):
    """
    Calculate elapsed time in seconds, minutes, hours, and days for each frame.

    Args:
    frames (array-like): Continuous frame numbers.

    Returns:
    pandas.DataFrame: A DataFrame with columns for seconds, minutes, hours, and days.

    To run:
    time_columns = calculate_time_units(data['continuous_frames'])
    data = pd.concat([data, time_columns], axis=1)
    """
    frames = np.array(frames)  # Ensure input is a numpy array for vectorized computation

    # Calculate elapsed time in seconds
    elapsed_seconds = 2 * frames
    
    # Convert to other units
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60
    elapsed_days = elapsed_hours / 24

    # Return as a DataFrame
    return pd.DataFrame({
        'time_seconds': elapsed_seconds,
        'time_minutes': elapsed_minutes,
        'time_hours': elapsed_hours,
        'time_days': elapsed_days
    })


def plot_data_over_interval(x, y, interval, x_label, y_label):
    """
    Plot speed variability over time.

    Args:
    data (pd.DataFrame): DataFrame containing 'speed_variability' and 'time'.
    interval (int): Interval used for calculations (used for graphing notes).
    """
    plt.figure(figsize=(10, 6))
    # Plot speed variability against time
    plt.plot(x, y, label=y_label, color='blue', linewidth=1.5)
    plt.title(f"{y_label} Over Time (Interval = {interval} frames)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.legend()
    plt.grid()
    plt.show()

#speed variability
def calculate_speed_variability(data, interval, threshold_speed):
    """
    Calculate speed variability over specified intervals.

    Args:
    data (pd.DataFrame): DataFrame containing at least the 'euclidean_speed' column.
    interval (int): Number of rows (frames) per interval.

    Returns:
    pd.DataFrame: DataFrame with new columns:
                  - 'std_speed': Standard deviation of speed in each interval
                  - 'mean_speed': Mean speed in each interval
                  - 'speed_variability': Ratio of std to mean in each interval
    """
    # Validate input
    if 'euclidean_speed' not in data.columns:
        raise ValueError("The DataFrame must contain a column named 'euclidean_speed'.")

    data['RoamingIndic'] = (data['euclidean_speed'] > threshold_speed).astype(int)
    # Calculate rolling statistics based on intervals
    rolling_std = data['euclidean_speed'].rolling(window=interval, min_periods=1).std()
    rolling_mean = data['euclidean_speed'].rolling(window=interval, min_periods=1).mean()

    # Assign computed values to new columns
    data['std_speed'] = rolling_std
    data['mean_speed'] = rolling_mean
    data['speed_variability'] = data['std_speed'] / data['mean_speed']

    data['roaming_frac'] = data['RoamingIndic'].rolling(window=interval, min_periods=1).mean()

    # Propagate interval values to all rows within the interval
    for col in ['std_speed', 'mean_speed', 'speed_variability']:
        data[col] = data[col].shift(-interval + 1).fillna(method='bfill')
        
    return data