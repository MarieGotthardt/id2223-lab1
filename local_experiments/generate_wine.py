from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

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

X = df.drop("quality", axis=1)
y = df["quality"]
y = y - 3 # remap labels from 3-9 to 0-6

smote = SMOTE(k_neighbors=4)

# Fit SMOTE to your data and generate synthetic samples
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a DataFrame of the resampled data
synthetic_df = pd.DataFrame(X_resampled, columns=df.columns[:-1])
synthetic_df['quality'] = y_resampled

# Print some of the synthetic data
print(synthetic_df.head())
