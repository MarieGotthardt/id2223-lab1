import pandas as pd
import numpy as np
import xgboost as xgb
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

    # Check for duplicates and drop duplicates
    wine_df.drop_duplicates(inplace=True)
    wine_df.reset_index(drop=True, inplace=True)

    # Transform categorical variables into numerical variables
    for column in wine_df.columns:
        if pd.api.types.is_categorical_dtype(wine_df[column]) or pd.api.types.is_object_dtype(wine_df[column]):
            one_hot = pd.get_dummies(wine_df[column], prefix=column)
            wine_df = wine_df.drop(column, axis = 1)
            wine_df = wine_df.join(one_hot)

    # One-hot encode wine type
    for column in wine_df.columns:
        if pd.api.types.is_categorical_dtype(wine_df[column]) or pd.api.types.is_object_dtype(wine_df[column]):
            one_hot = pd.get_dummies(wine_df[column], prefix=column)
            wine_df = wine_df.drop(column, axis = 1)
            wine_df = wine_df.join(one_hot)

    # Get correlations within the features
    features = wine_df.loc[:, wine_df.columns != 'quality']
    cor = abs(features.corr())

    # Get only upper half of the symmetric correlation matrix
    feature_cor_upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))

    # Exclude features with a correlation coefficient that is higher than 0.7 to at least one other feature
    features_to_exclude = [column for column in feature_cor_upper.columns if any(feature_cor_upper[column] > 0.7)]

    # Find features to be kept
    features_to_be_kept = [feature for feature in wine_df.columns if feature not in features_to_exclude]

    # Drop features: drop all features that show a low correlation with the target variable and that are highly intercorrelated
    for column in wine_df.columns:
        if column not in features_to_be_kept:
            wine_df.drop(column, axis=1, inplace=True)


    # Initially, target values range from 3 to 9; we bin them into 5 categories and relabel them from 0 to 4
    # 0-3: Very Bad (0) = 1 stars, 4: Bad (1) = 2 stars, 5: Mediocre (2) = 3 stars, 6-7: Good (3) = 4 stars, 8-10: Very Good (4) = 5 stars
    wine_df['quality'] = [0 if x < 4  else 1 if x==4 else 2 if x==5 else 3 if x <8  else 4 for x in wine_df['quality']]

    # prepare data for training
    X = wine_df.drop("quality", axis=1)
    y = wine_df["quality"]

    # Split original data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ## CLASSIFY WITHOUT SMOTE ##
    print()
    print("Accuracy before SMOTE:")
    run_clf(X_train, X_test, y_train, y_test)

    ## SMOTE ##
    minority_class_size = y.value_counts().min()
    n_neighbors = min(minority_class_size - 1, 5)
    smote = SMOTE(k_neighbors=n_neighbors)
    
    # Resample training data
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    ## CLASSIFY AFTER SMOTE ##
    print("Accuracy when trained on SMOTE data:")
    run_clf(X_resampled, X_test, y_resampled, y_test)

    ## CLASSIFY WITH SMOTE AND ADJUSTED LABEL DISTRIBUTION ##
    wine_df = pd.DataFrame(X_resampled, columns=wine_df.columns)
    wine_df['quality'] = y_resampled

    wine_df = drop_samples(wine_df, class_label=0, num_samples_to_drop=2000)
    wine_df = drop_samples(wine_df, class_label=4, num_samples_to_drop=2000)
    wine_df = drop_samples(wine_df, class_label=1, num_samples_to_drop=1000)

    # prepare data for training
    X_adjusted = wine_df.drop("quality", axis=1)
    y_adjusted = wine_df["quality"]

    print("Accuracy when trained on adjusted SMOTE data:")
    run_clf(X_adjusted, X_test, y_adjusted, y_test)


# Define a function to randomly drop samples from a specified class
def drop_samples(dataframe, class_label, num_samples_to_drop):
    # Filter the class
    class_df = dataframe[dataframe['quality'] == class_label]
    
    # Randomly select samples to drop
    drop_indices = np.random.choice(class_df.index, num_samples_to_drop, replace=False)
    
    # Drop the samples
    return dataframe.drop(drop_indices)

## RUN CLASSIFIERS
def run_clf(X_train, X_test, y_train, y_test):
    ## XGBOOST ##
    # Initialize the XGBoost classifier
    xgb_clf = xgb.XGBClassifier()

    # Train the classifier
    xgb_clf.fit(X_train, y_train)

    # Predictions on the test set
    xgb_y_pred = xgb_clf.predict(X_test)

    # Calculate the accuracy
    xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
    print("XGBoost accuracy:", xgb_accuracy)
    print()

if __name__ == "__main__":
    main()
