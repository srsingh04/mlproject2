import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

  
def group_kfold_validation(df, model_select="logistic", n_splits=5, random_state=42):
    """
    Perform GroupKFold cross-validation on the dataset based on worm_id.

    Args:
        df: DataFrame containing the dataset.
        n_splits: Number of folds for cross-validation.

    Returns:
        None (prints mean accuracy and classification report).
    """
    # Extract features (X), target (y), and groups (worm_id)
    X = df.drop(columns=['id', 'worm_id', 'drugged', 'average_distance_per_frame', 'maximal_distance_traveled', 'average_acceleration'])  # Drop 'worm_id' and target #dropping id removes all prediction strength
    y = df['drugged']  # Target variable
    groups = df['worm_id']  # Group variable for GroupKFold
    
    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    accuracies = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        # Create train and test sets
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


        # sample imputing (NEED TO IMPROVE) -- ONLY TEMPORARY
        if np.isnan(X_train).any().any() or np.isnan(X_test).any().any():
          # Impute missing values with the median
          imputer = SimpleImputer(strategy='median')
          X_train = imputer.fit_transform(X_train)
          X_test = imputer.fit_transform(X_test)

        # Initialize and train Random Forest Classifier
        if model_select == "logistic":
          model = LogisticRegression(random_state=random_state, penalty='l2', max_iter=1000, multi_class='multinomial', solver='saga', class_weight={0:0.5,1:0.25,2:0.25}) #
        elif model_select == "random_forest":
           model = RandomForestClassifier(random_state=random_state)
        elif model_select == "decision_tree":
           model = DecisionTreeClassifier(random_state=random_state)
        elif model_select == "svm":
           model = SVC(random_state=random_state, class_weight='balanced')
        # elif model_select == "linear":
        #    model = LinearRegression()
        elif model_select == "xgboost":
          model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, learning_rate=0.05, gamma=0.001) #converges regardless of n_estimators, learning_rate and gamma
        else:
           raise Exception("Invalid model selected, options are logistic, random_forest, decision_tree")


        #scale for linear models only
        if model_select == "logistic" or model_select == "svm":
            scaler_train = StandardScaler()
            scaler_train.fit(X_train)
            X_train = scaler_train.transform(X_train)

            scaler_test = StandardScaler()
            scaler_test.fit(X_test)
            X_test = scaler_test.transform(X_test)


        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Optionally print classification report for each fold
        print(classification_report(y_test, y_pred))

        # Feature importance for tree-based models
        if model_select in ["random_forest", "xgboost"]:
            importances = model.feature_importances_
            feature_importances = sorted(zip(X.columns, importances), key=lambda x: -x[1])
            print("\nFeature Importances:")
            for feature, importance in feature_importances:
                print(f"{feature}: {importance:.4f}")

    # Report overall results
    print(f"Mean Accuracy: {np.mean(accuracies):.2f}")
    print(f"Standard Deviation: {np.std(accuracies):.2f}")

    return np.mean(accuracies), np.std(accuracies)