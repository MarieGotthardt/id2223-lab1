import os
import modal

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=6)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

    feature_view = fs.get_feature_view(name="wine_enriched_balanced", version=1)
    batch_data = feature_view.get_batch_data()

    # Make predictions
    y_pred = model.predict(batch_data)
    # print(y_pred)

    # Get the latest wine quality
    offset = 1
    quality = y_pred[y_pred.size-offset]
    print("Predicted quality: " + str(quality))
    quality_url = "https://raw.githubusercontent.com/SamuelHarner/review-images/main/images/" + str(int(quality+1)) + "_stars.png"
    img = Image.open(requests.get(quality_url, stream=True).raw)            
    img.save("./latest_wine.png")
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)

    wine_fg = fs.get_feature_group(name="wine_enriched_balanced", version=1)
    df = wine_fg.read()
    # print(df)
    label = df.iloc[-offset]["quality"]
    print("Actual quality: " + str(label))
    label_url = "https://raw.githubusercontent.com/SamuelHarner/review-images/main/images/" + str(int(label+1)) + "_stars.png"
    img = Image.open(requests.get(label_url, stream=True).raw)
    img.save("./actual_wine.png")
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)

    monitor_fg = fs.get_or_create_feature_group(name="quality_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine quality Prediction/Outcome Monitoring"
                                                )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [quality],
        'label': [label],
        'datetime': [now],
    }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent_wine.png', table_conversion='matplotlib')
    dataset_api.upload("./df_recent_wine.png", "Resources/images", overwrite=True)

    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our quality_predictions feature group has examples of at least 5 qualities
    print("Number of different quality predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() >= 5:
        results = confusion_matrix(labels, predictions)

        true_labels = [f'True {int(i)}' for i in range(0, 5)]
        pred_labels = [f'Pred {int(i)}' for i in range(0, 5)]
        df_cm = pd.DataFrame(results, true_labels, pred_labels)
        cm = sns.heatmap(df_cm, annot=True, fmt='g')
        fig = cm.get_figure()

        fig.savefig("./confusion_matrix_wine.png")
        dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)
    else:
        print("You need at least 5 different wine quality predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get at least 5 different wine quality predictions")


if __name__ == "__main__":
    g()


