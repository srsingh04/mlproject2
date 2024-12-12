import numpy as np
import pandas as pd
import argparse
import functionsATR


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

    time_columns = functionsATR.calculate_time_units(data['continuous_frames'])
    data = pd.concat([data, time_columns], axis=1)
    data = functionsATR.calculate_speed_variability(data, interval=INTERVAL, threshold_speed=THEC)
    
    return data


if __name__ == "__main__":
    main()


