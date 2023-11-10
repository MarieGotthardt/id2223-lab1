import gradio as gr
from PIL import Image
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


def generate_number_image(number, img_size=(100, 100), bg_color='white', text_color='black'):
    # Create an image with white background
    image = Image.new('RGB', img_size, bg_color)

    # Initialize the drawing context
    draw = ImageDraw.Draw(image)

    # Font settings (Default font in this case)
    font = ImageFont.load_default()

    # Calculate width and height of the text
    text_width, text_height = draw.textsize(str(number), font)

    # Calculate X, Y position of the text
    x = (img_size[0] - text_width) / 2
    y = (img_size[1] - text_height) / 2

    # Draw the number on the image
    draw.text((x, y), str(number), fill=text_color, font=font)

    return image

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
    wine_url = generate_number_image(int(res))
    img = Image.open(requests.get(wine_url, stream=True).raw)
    return img


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
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)

