#Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

def evaluate_model_kfold(data, model_select="linear", n_splits=5, random_state=42):
    df = data.copy()
    df["lifespan"] = df["lifespan"] / 24

    X = df.drop(columns=["lifespan", "worm_id", "id", "group"])
    y = df["lifespan"]
    groups = df["worm_id"]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    if model_select == "linear":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = LinearRegression()
    elif model_select == "random_forest":
        model = RandomForestRegressor(random_state=random_state)
    elif model_select == "xgboost":
        model = XGBRegressor(random_state=random_state, n_estimators=100, learning_rate=0.05)
    elif model_select == "elasticnet":
        model = ElasticNet(random_state=random_state, alpha=0.1, l1_ratio=0.5)
    else:
        raise ValueError("Invalid model selection.")

    # Perform GroupKFold CV and collect metrics
    gkf = GroupKFold(n_splits=n_splits)
    r2_scores = []
    mse_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # Calculate mean and std of metrics
    mse_mean, mse_std = np.mean(mse_scores), np.std(mse_scores)
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)

    return {"model": model_select, "mse_mean": mse_mean, "mse_std": mse_std, 
            "r2_mean": r2_mean, "r2_std": r2_std}


def predict_lifespan(data, model_select="linear", test_size=0.2, random_state=42):
    """
    Predict the lifespan of worms based on behavioral data.

    Args:
        df: DataFrame containing the dataset.
        model_select: Choice of regression model ('linear', 'random_forest', 'xgboost').
        test_size: Fraction of data to use as test set.
        random_state: Seed for reproducibility.

    Returns:
        None (prints evaluation metrics and feature importances for tree-based models).
    """
    df = data.copy()
    # csv = 'mergedworms.csv'# Read the CSV file
    # df = pd.read_csv(csv)


    # Preprocess lifespan (convert from hours to days)
    df['lifespan'] = df['lifespan'] / 24

    # Extract features (X), target (y), and groups (worm_id)
    # X = df.drop(columns=['id', 'worm_id', 'lifespan', 'average_distance_per_frame', 'maximal_distance_traveled', 'average_acceleration'])
    X = df.drop(columns = ['lifespan', 'worm_id']) #id
    if 'id' in df.columns:
        X = X.drop(columns = ['id'])
    print(X.columns)
    y = df['lifespan']
    groups = df['worm_id']
    

    # Split the dataset into training and testing based on worm_id grouping
    gkf = GroupKFold(n_splits=int(1 / test_size))
    train_idx, test_idx = next(gkf.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    #imputing
    if np.isnan(X_train).any().any() or np.isnan(X_test).any().any():
        # Impute missing values with the median
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.fit_transform(X_test)

    # Initialize the model
    if model_select == "linear":
        model = LinearRegression()
    elif model_select == "random_forest":
        model = RandomForestRegressor(random_state=random_state)
    elif model_select == "xgboost":
        model = XGBRegressor(random_state=random_state, n_estimators=100, learning_rate=0.05)
    else:
        raise Exception("Invalid model selected. Options are 'linear', 'random_forest', 'xgboost'.")

    # Scale features for linear models
    if model_select == "linear":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Feature importance for tree-based models
    if model_select in ["random_forest", "xgboost"]:
        importances = model.feature_importances_
        feature_importances = sorted(zip(X.columns, importances), key=lambda x: -x[1])
        print("\nFeature Importances:")
        for feature, importance in feature_importances:
            print(f"{feature}: {importance:.4f}")

    import matplotlib.pyplot as plt
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Lifespan (days)")
    plt.ylabel("Predicted Lifespan (days)")
    plt.title("Predicted vs. Actual Lifespan")
    plt.show()

    print("y_pred max min", y_pred.max(), y_pred.min())
    print("y_test max min", y_test.max(), y_test.min())
    print("size: ", y_pred.size, y_test.size)

    return model, mse, r2


def predict_lifespan2(data, model_select="linear", test_size=0.2, n_splits=5, n_repeats=3, random_state=42):
    """
    Predict the lifespan of worms using repeated GroupKFold cross-validation.

    Args:
        data: DataFrame containing the dataset.
        model_select: Regression model ('linear', 'random_forest', 'xgboost').
        test_size: Fraction for a single test fold size.
        n_splits: Number of splits for GroupKFold.
        n_repeats: Number of times to repeat K-Fold cross-validation.
        random_state: Seed for reproducibility.

    Returns:
        None (prints evaluation metrics, shows feature importances, and plots predictions).
    """
    df = data.copy()
    df["lifespan"] = df["lifespan"] / 24  # Convert lifespan to days

    # Extract features (X) and target (y)
    X = df.drop(columns=["id", "lifespan", "worm_id", "group"])  # Drop target and grouping columns #, "time_elapsed_(hours)", "group"
    if id in X.columns:
        X = X.drop(columns=["id"])
    y = df["lifespan"]
    groups = df["worm_id"]

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Scale features for linear regression
    if model_select == "linear":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Initialize model
    if model_select == "linear":
        model = LinearRegression()
    elif model_select == "random_forest":
        model = RandomForestRegressor(random_state=random_state)
    elif model_select == "xgboost":
        model = XGBRegressor(random_state=random_state, n_estimators=100, learning_rate=0.05)
    elif model_select == "elasticnet":
        model = ElasticNet()
    elif model_select == "svr":
        model = SVR()
    elif model_select == "knr":
        model = KNeighborsRegressor()
    else:
        raise ValueError("Invalid model selection. Choose 'linear', 'random_forest', or 'xgboost'.")

    # # Perform repeated GroupKFold cross-validation
    # gkf = GroupKFold(n_splits=n_splits)
    # y_pred = cross_val_predict(
    #     model, X, y, groups=groups, cv=gkf, method="predict"
    # )  # Generate predictions for all data points
    # Perform GroupKFold CV and collect predictions
    gkf = GroupKFold(n_splits=n_splits)
    preds_per_split = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y.iloc[train_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_per_split.append((test_idx, preds))

    # Aggregate predictions and calculate std deviation
    pred_means = np.zeros(len(y))
    pred_stds = np.zeros(len(y))
    for test_idx, preds in preds_per_split:
        pred_means[test_idx] = preds
        pred_stds[test_idx] = np.std(preds)

    mse = mean_squared_error(y, pred_means)
    r2 = r2_score(y, pred_means)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Feature importance for linear models
    if model_select in ["linear", "elasticnet"]:
        model.fit(X, y)  # Fit the model
        feature_importances = sorted(zip(df.drop(columns=["lifespan", "worm_id"]).columns, model.coef_), key=lambda x: -abs(x[1]))
        print("\nFeature Importances (Linear Regression Coefficients):")
        for feature, coef in feature_importances:
            print(f"{feature}: {coef:.4f}")

    # Feature importance for tree-based models
    if model_select in ["random_forest", "xgboost"]:
        model.fit(X, y)
        importances = model.feature_importances_
        feature_importances = sorted(zip(df.drop(columns=["lifespan", "worm_id"]).columns, importances), key=lambda x: -x[1])
        print("\nFeature Importances:")
        for feature, importance in feature_importances:
            print(f"{feature}: {importance:.4f}")

    # Plot predictions with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(y, pred_means, yerr=pred_stds, fmt="o", alpha=0.7, label="Predicted", ecolor="red", capsize=3)
    # plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect Fit")
    plt.xlabel("Actual Lifespan (days)")
    plt.ylabel("Predicted Lifespan (days)")
    plt.title("Predicted vs. Actual Lifespan with Uncertainty")
    plt.legend()
    plt.show()

    return model, mse, r2