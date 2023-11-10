import os
import modal

def generate_wine():
    """
    Returns a new wine as a single row in a DataFrame
    """
    import pandas as pd
    import random
      # TODO

    #return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = generate_wine()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
   g()
