import pandas as pd
import numpy as np


def split_by_worm_id(df, test_size=0.2):
  """Splits a DataFrame based on 'worm_id' into training and testing sets.

  Args:
    df: The input DataFrame.
    test_size: The proportion of data to include in the test set.

  Returns:
    A tuple of two DataFrames: (train_df, test_df)
  """
  while True:
    # Get unique worm IDs
    unique_worms = df['worm_id'].unique()

    # Randomly select worm IDs for the test set
    np.random.shuffle(unique_worms)
    test_worm_ids = unique_worms[:int(len(unique_worms) * test_size)]

    # Create training and testing DataFrames
    train_df = df[~df['worm_id'].isin(test_worm_ids)]
    test_df = df[df['worm_id'].isin(test_worm_ids)]

    #drop worm id
    train_df = train_df.drop('worm_id', axis=1)
    test_df = test_df.drop('worm_id', axis=1)
    #drop distance
    train_df = train_df.drop('average_distance_per_frame', axis=1)
    test_df = test_df.drop('average_distance_per_frame', axis=1)
    #drop maximal distance
    train_df = train_df.drop('maximal_distance_traveled', axis=1)
    test_df = test_df.drop('maximal_distance_traveled', axis=1)
    #drop group
    train_df = train_df.drop('group', axis=1)
    test_df = test_df.drop('group', axis=1)
    #drop acceleration
    train_df = train_df.drop('average_acceleration', axis=1)
    test_df = test_df.drop('average_acceleration', axis=1)

    # Check if all classes are present in both sets
    if len(train_df['drugged'].unique()) == 3 and len(test_df['drugged'].unique()) == 3:
      return train_df, test_df


def train_test_x_and_y(train_df, test_df):
  """Splits a DataFrame based on the column id 'drugged' to return X and y training and testing sets

  Args:
    train_df: Training DataFrame
    test_df: Testing DataFram

  Returns:
    A tuple of four DataFrames: (X_train, y_train, X_test, y_test)
  """

  X_train = train_df.drop('drugged', axis=1)  # Features
  y_train = train_df['drugged']  # Target variable

  X_test = test_df.drop('drugged', axis=1)  # Features
  y_test = test_df['drugged']  # Target variable

  return X_train, y_train, X_test, y_test

