import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    ## PREPARE DATA ##
    wine_df = pd.read_csv("../winequality.csv")

    # Fill missing data with either random data or a category corresponding to "Unknown"
    for column in wine_df.columns:
        if wine_df[column].isna().any() and pd.api.types.is_numeric_dtype(wine_df[column]):
            wine_df.loc[wine_df[column].isna(), column] = [i for i in np.random.choice(range(round(wine_df[column].min()), round(wine_df[column]. max())), wine_df[column].isna().sum())]
        elif wine_df[column].isna().any() and (pd.api.types.is_object_dtype(wine_df[column]) or pd.api.types.is_categorical_dtype(wine_df[column])):
            wine_df[column].fillna("Unknown")

    # One-hot encode wine type
    for column in wine_df.columns:
        if pd.api.types.is_categorical_dtype(wine_df[column]) or pd.api.types.is_object_dtype(wine_df[column]):
            one_hot = pd.get_dummies(wine_df[column], prefix=column)
            wine_df = wine_df.drop(column, axis = 1)
            wine_df = wine_df.join(one_hot)

    ## CLASSIFY WITHOUT REMOVING DATA FEATURES ##
    X = wine_df.drop("quality", axis=1)
    y = wine_df["quality"]
    y = y - 3 # remap labels from 3-9 to 0-6

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print()
    print("Accuracies without dropping features:")
    run_clf(X_train, X_test, y_train, y_test)
    
    ## FEATURE SELECTION ##
    # Correlation with target variable quality
    cor = wine_df.corr()
    cor_quality = abs(cor["quality"])

    threshold = 0.075

    # Selecting only features with correlation coefficient > threshold
    important_features = cor_quality[cor_quality>threshold].sort_values()

    # Checking for correlation between the important features
    # If features are highly intercorrelated, we should only keep one and drop the other
    # we should probably drop either red or white and maybe density since it is highly correlated with alcohol

    feature_cor = wine_df[list(important_features.iloc[:-1].index)].corr().abs()

    # Select upper triangle of correlation matrix
    feature_cor_upper = feature_cor.where(np.triu(np.ones(feature_cor.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    features_to_exclude = [column for column in feature_cor_upper.columns if any(feature_cor_upper[column] > 0.95)]
    
    # Find features to be kept
    features_to_be_kept = [feature for feature in important_features.index.to_list() if feature not in features_to_exclude]

    # Drop features: drop all features that show a low correlation with the target variable and that are highly intercorrelated
    for column in wine_df.columns:
        if column not in features_to_be_kept:
            wine_df.drop(column, axis=1, inplace=True)


    ## CLASSIFY AFTER FEATURE SELECTION ##
    X = wine_df.drop("quality", axis=1)
    y = wine_df["quality"]
    y = y - 3 # remap labels from 3-9 to 0-6

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Accuracies with feature subset:")
    run_clf(X_train, X_test, y_train, y_test)

def run_clf(X_train, X_test, y_train, y_test):
    ## XGBOOST ##
    # Initialize the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')

    # Train the classifier
    xgb_clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score

    # Predictions on the test set
    xgb_y_pred = xgb_clf.predict(X_test)

    # Calculate the accuracy
    xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
    print("XGBoost accuracy:", xgb_accuracy)


    ## RANDOM FOREST ##
    # Initialize the RF classifier
    rf_clf = RandomForestClassifier()

    # Train the classifier
    rf_clf.fit(X_train, y_train)

    # Predictions on the test set
    rf_y_pred = rf_clf.predict(X_test)

    # Calculate the accuracy
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    print("RF accuracy:", rf_accuracy)

    print()

if __name__ == "__main__":
    main()
