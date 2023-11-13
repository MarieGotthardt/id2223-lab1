import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

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

    ## SMOTE ##
    minority_class_size = y.value_counts().min()
    n_neighbors = min(minority_class_size - 1, 5)
    smote = SMOTE(k_neighbors=n_neighbors)

    # Split original data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Resample training data
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    ## CLASSIFY AFTER SMOTE ##
    print("Accuracies when trained on SMOTE data:")
    run_clf(X_resampled, X_test, y_resampled, y_test)


## RUN CLASSIFIERS
def run_clf(X_train, X_test, y_train, y_test):
    ## XGBOOST ##
    # Initialize the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')

    # Train the classifier
    xgb_clf.fit(X_train, y_train)

    

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
