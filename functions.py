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
    recording_period = 900  # Frames in a recording session (30 minutes)
    recording_duration = 2 * recording_period  # Time in seconds for one session (30 minutes)
    break_duration = 5.5 * 3600  # 5.5 hours in seconds

    # Calculate elapsed time in seconds
    elapsed_seconds = (
        2 * (frames % recording_period) +  # Time within the current recording session
        (frames // recording_period) * (recording_duration + break_duration)  # Time from skipped intervals
    )
    
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