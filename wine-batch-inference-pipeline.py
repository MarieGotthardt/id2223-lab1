import os
import modal

LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    hopsworks_image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "sklearn==1.1.1", "dataframe-image"])


