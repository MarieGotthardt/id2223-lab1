import pandas as pd
import numpy as np
from sklearn import mixture
import random

def generate_wine(wine_type, df):
    if wine_type == 0:
        df = df.loc[df['type_red'] == 1]
        print("Red wine added")
    else:
        df = df.loc[df['type_white'] == 1]
        print("White wine added")
    
    gmm = mixture.GaussianMixture(n_components=8, covariance_type='full', random_state=0) # n_components chosen based on experiments performed
    gmm.fit(df)
    wine_new, _ = gmm.sample(1)

    # clip feature values so they do not go outside original range
    for i, column in enumerate(df.columns):
        min_val = df[column].min()
        max_val = df[column].max()
        original_value = wine_new[0, i]

        # Check if the value is outside the min-max range
        if original_value < min_val or original_value > max_val:
            print(f"Clipping value {original_value} in '{column}' to range [{min_val}, {max_val}]")
            wine_new[0, i] = np.clip(original_value, min_val, max_val)

    synthetic_df = pd.DataFrame(wine_new, columns=df.columns)

    # change type back to the appropriate integer
    if wine_type == 0:
        synthetic_df['type_red'] = 1
        synthetic_df['type_white'] = 0
    else:
        synthetic_df['type_red'] = 0
        synthetic_df['type_white'] = 1

    # round quality to an integer
    synthetic_df['quality'] = int(round(synthetic_df['quality']))

    return synthetic_df

def main():
    # Load an example wine dataset or replace with your own
    df = pd.read_csv("../winequality.csv")

    # Fill missing data with either random data or a category corresponding to "Unknown"
    for column in df.columns:
        if df[column].isna().any() and pd.api.types.is_numeric_dtype(df[column]):
            df.loc[df[column].isna(), column] = [i for i in np.random.choice(range(round(df[column].min()), round(df[column]. max())), df[column].isna().sum())]
        elif df[column].isna().any() and (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])):
            df[column].fillna("Unknown")

    # One-hot encode wine type
    for column in df.columns:
        if pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            one_hot = pd.get_dummies(df[column], prefix=column)
            df = df.drop(column, axis = 1)
            df = df.join(one_hot)

    wine_type = random.choice([0, 1])
    synthetic_df = generate_wine(wine_type, df)
    print(synthetic_df)


if __name__ == "__main__":
    main()
