import random
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

rand_idx = random.randint(0, len(df) - 1)
row = df.iloc[rand_idx].copy()
for col in df.columns:
    if col != 'quality' and col != 'type_red' and col != 'type_white':
        perturbation = np.random.normal(0, 0.1 * df[col].std())
        row[col] += perturbation

print("Original wine")
print(df.iloc[rand_idx])
print()
print("New wine")
print(row)
