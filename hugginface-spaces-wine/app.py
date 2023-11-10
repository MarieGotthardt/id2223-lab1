import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")


def wine(fixed_acidity, citric_acid, type_white, chlorides, volatile_acidity, density, alcohol):
    print("Calling function")
    #     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]],
    df = pd.DataFrame([[fixed_acidity, citric_acid, type_white, chlorides, volatile_acidity, density, alcohol]],
                      columns=['fixed_acidity', 'citric_acid', 'type_white', 'chlorides', 'volatile_acidity', 'density', 'alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    #     print("Res: {0}").format(res)
    print(res)
    return str(res[0])

demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with fixed_acidity, citric_acid, type, chlorides, volatile_acidity, density, alcohol"
                "to predict of which quality the wine is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=7.2, label="fixed acidity"),
        gr.inputs.Number(default=0.34, label="volatile acidity"),
        gr.inputs.Number(default=0.32, label="citric acid"),
        gr.inputs.Textbox(default="red", label="type (red, white)"),
        gr.inputs.Number(default=10.5, label="alcohol"),
        gr.inputs.Number(default=0.99, label="density"),
        gr.inputs.Number(default=0.06, label="chlorides"),
    ],
    outputs="text")

demo.launch(debug=True)

