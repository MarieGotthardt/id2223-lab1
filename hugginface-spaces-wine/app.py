import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")


def wine(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                         free_sulfur_dioxide, density, ph, sulphates, alcohol, type_red):
    print("Calling function")
    #     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]],
    df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                         free_sulfur_dioxide, density, ph, sulphates, alcohol, type_red]],
                      columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                               'free_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol', 'type_red'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    #     print("Res: {0}").format(res)
    print(res)

    star_url = "https://raw.githubusercontent.com/SamuelHarner/review-images/main/images/" + str(res[0]) + "_stars.png"
    img = Image.open(requests.get(star_url, stream=True).raw)
    return img

demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with fixed_acidity, citric_acid, type, chlorides, volatile_acidity, density, alcohol"
                "to predict of which quality the wine is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=7.2, label="fixed acidity (3.8 ... 15.9)"),
        gr.inputs.Number(default=0.34, label="volatile acidity (0.00 ... 1.58)"),
        gr.inputs.Number(default=0.32, label="citric acid (0.00 ... 1.66)"),
        gr.inputs.Number(default=0, label="type (0...red, 1...white)"),
        gr.inputs.Number(default=10.5, label="alcohol (8.0 ... 14.9"),
        gr.inputs.Number(default=0.99, label="density (0.99 ... 1.04)"),
        gr.inputs.Number(default=0.06, label="chlorides (0.00 ...0.61)"),
        gr.inputs.Number(default=5.07, label="residual sugar (0.60 ...65.80)"),
        gr.inputs.Number(default=30.06, label="free sulfur dioxide (1.0 ...289.0)"),
        gr.inputs.Number(default=3.22, label="pH (2.72 ...4.01)"),
        gr.inputs.Number(default=0.53, label="sulphates (0.00 ...2.00)"),
    ],
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)

