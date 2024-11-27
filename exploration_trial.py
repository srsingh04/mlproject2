import pandas as pd
import numpy as np
import os

def worm_death(filename, threshold=100):
    #read the file
    data = pd.read_csv(filename)
    #Correct Frame numbers --> now all frames are unique, use as ID column
    data["continuous_frames"] = data.index + 1 
    #Add Euclidean Speed
    data['euclidean_speed'] = np.sqrt((data['X'].diff())**2 + (data['Y'].diff())**2) * 2
    data['euclidean_speed'] = data['euclidean_speed'].shift(-1)
    #Compare euclidean speed and speed that is in dataset (for later debugging/cleaning)
    data['speed_diff'] = data['euclidean_speed'] - data['Speed']
    data['speed_diff'] = np.where(abs(data['speed_diff']) < 0.01, 0, data['speed_diff'])
    non_zero_count = (data['speed_diff'] != 0).sum()

    #Worm death
    # Step 1: Check for "no significant movement" with a tolerance of 0.5
    tolerance_movement = 0.5 #death based on x,y movement
    # data['no_movement'] = (data['X'].diff().fillna(0).abs() < tolerance_movement) & (data['Y'].diff().fillna(0).abs() < tolerance_movement)
    tolerance_speed = 0.1 #death based on speed (Explore what happens if you use euclidean speed)
    data['no_movement'] = data['Speed'].abs() < tolerance_speed

    # Step 2: Count consecutive no-movement frames
    # Use a cumulative counter that resets when movement is detected
    data['no_movement_count'] = data['no_movement'].cumsum() - data['no_movement'].cumsum().where(~data['no_movement']).ffill().fillna(0).astype(int)

    # Step 3: Identify when no movement reaches the threshold
    # threshold = 100
    data['dead'] = data['no_movement_count'] >= threshold

    # Step 4: Record the first row where the worm is declared dead
    dead_row = data[data['dead']].index.min()

    # if pd.notna(dead_row):
    #     print(f"The worm is declared dead at row index: {dead_row}")
    #     # Display relevant columns for verification
    #     print(data[['Frame', 'X', 'Y', 'Changed Pixels', 'Speed', 'no_movement', 'no_movement_count', 'dead']].iloc[max(dead_row - 3, 0):min(dead_row + 3, len(data) - 1)])
    # else:
    #     print("The worm never reaches the dead threshold.")

    return dead_row