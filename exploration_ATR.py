import numpy as np
import pandas as pd
import argparse
import functions


threshold = 500
tolerance_speed = 0.1
speed_difference_threshold = 100
INTERVAL = 900
THEC= 5 #threshold needs to be properly set for the data


def main():
    parser = argparse.ArgumentParser(description="Preprocessing CSV data.")
    parser.add_argument('csv_file', type=str, help="Path to the CSV file")
    args = parser.parse_args()

    # Load the data
    data = pd.read_csv(args.csv_file)
    
    # Insert your preprocessing logic here
    data_preprocess(data)



def data_preprocess(data):

    data = data.dropna(subset=['X', 'Y']) # Drop rows with missing X or Y values
    data = data.reset_index(drop=True)

    data["continuous_frames"] = data.index + 1


    data['Instantaneous Distance'] = np.sqrt(
        (data['X'].diff() ** 2) + (data['Y'].diff() ** 2)
    ).fillna(0)
    data['Total Distance'] = data['Instantaneous Distance'].cumsum()
    data['Angle'] = np.arctan2(data['Y'].diff(), data['X'].diff()).fillna(0)    
    data['Angular Change'] = data['Angle'].diff().abs().fillna(0)


    data['euclidean_speed'] = np.sqrt((data['X'].diff())**2 + (data['Y'].diff())**2) * 2
    data['euclidean_speed'] = data['euclidean_speed'].shift(-1)

    
    data['speed_diff'] = data['euclidean_speed'] - data['Speed']
    data['speed_diff'] = np.where(abs(data['speed_diff']) < 0.01, 0, data['speed_diff'])

    # Step 1: Check for "no significant movement" with a tolerance of 0.5
    #tolerance_movement = 0.5
    # data['no_movement'] = (data['X'].diff().fillna(0).abs() < tolerance_movement) & (data['Y'].diff().fillna(0).abs() < tolerance_movement)
    
    data['no_movement'] = data['Speed'].abs() < tolerance_speed

# Step 2: Count consecutive no-movement frames
# Use a cumulative counter that resets when movement is detected
    data['no_movement_count'] = data['no_movement'].cumsum() - data['no_movement'].cumsum().where(~data['no_movement']).ffill().fillna(0).astype(int)

# Step 3: Identify when no movement reaches the threshold
    #threshold = 153
    data['dead'] = data['no_movement_count'] >= threshold

# Step 4: Record the first row where the worm is declared dead
    dead_row = data[data['dead']].index.min()

    if pd.notna(dead_row):
        print(f"The worm is declared dead at row index: {dead_row}")
        final_age = data['time_hours'].iloc[dead_row]
        print(f"The worm lived: {final_age/24} days.")
    # Display relevant columns for verification
        print(data[['Frame', 'X', 'Y', 'Changed Pixels', 'Speed', 'no_movement', 'no_movement_count', 'dead']].iloc[max(dead_row - 3, 0):min(dead_row + 3, len(data) - 1)])
    else:
        print("The worm never reaches the dead threshold.")
       

    time_columns = functions.calculate_time_units(data['continuous_frames'])
    data = pd.concat([data, time_columns], axis=1)
    data = functions.calculate_speed_variability(data, interval=INTERVAL, threshold_speed=THEC)
    
    return data


if __name__ == "__main__":
    main()


