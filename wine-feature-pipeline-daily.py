import os
import random
import hopsworks
import pandas as pd
import numpy as np
from sklearn import mixture

def generate_wine(wine_type, df):
    """
    Returns a new wine as a single row in a DataFrame
    """
    if wine_type == 0:
        df = df.loc[df['type_red'] == 1]
        print("Red wine added")
    else:
        df = df.loc[df['type_red'] == 0]
        print("White wine added")
    
    gmm = mixture.GaussianMixture(n_components=8, covariance_type='full', random_state=0) # n_components chosen based on experiments performed
    gmm.fit(df)
    wine_new, _ = gmm.sample(1)

    # clip feature values so they do not go outside original range
    for i, column in enumerate(df.columns):
        min_val = df[column].min()
        max_val = df[column].max()
        wine_new[0, i] = np.clip(wine_new[0, i], min_val, max_val)

    synthetic_df = pd.DataFrame(wine_new, columns=df.columns)

    # round quality to an integer
    synthetic_df['quality'] = int(round(synthetic_df['quality']))

    return synthetic_df

def g():
    project = hopsworks.login()
    fs = project.get_feature_store()
    wine_fg = fs.get_feature_group(name="wine_enriched",version=1)
    query = wine_fg.select_all()
    df = query.read()

    wine_type = random.choice([0, 1])

    synthetic_df = generate_wine(wine_type, df)

    wine_fg.insert(synthetic_df)

if __name__ == "__main__":
   g()
