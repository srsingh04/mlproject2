import pandas as pd
from glob import glob
import exploration_ATR

# Define the folder containing all worm CSV files
#folder_path = "Optogenetics-20241202T112946Z-001/Optogenetics/ATR+"  # Update with the path to your csv files

folder_path = input("Please enter the path to the folder containing worm CSV files: ").strip()

# Ensure the path ends with a slash or backslash
if not folder_path.endswith(("/","\\")):
    folder_path += "/"



def combineFiles(folder_path):
    # Find all CSV files in the folder
    files = glob(folder_path + "*.csv")
    if not files:
        print("No CSV files found in the specified folder.")
        return

    # Initialize an empty list to store DataFrames
    all_worms_data = []

    # Loop through each file and add a worm_id
    for worm_id, file in enumerate(files):
        # Read the CSV file
        df = pd.read_csv(file)
        df = exploration_ATR.data_preprocess(df)
        # Add a worm_id column
        df['worm_id'] = worm_id
        # Append to the list
        all_worms_data.append(df)

    # Combine all DataFrames into one
    combined_data = pd.concat(all_worms_data, ignore_index=True)

    # Save the combined DataFrame to a CSV file (optional)
    combined_data.to_csv("merged_worm_data.csv", index=False)

    print("Combined data with worm_id column:")
    print(combined_data.head())
    
combineFiles(folder_path)