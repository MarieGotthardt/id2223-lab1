import requests
from PIL import Image
from io import BytesIO

# URL of the raw image
image_url = "https://raw.githubusercontent.com/SamuelHarner/review-images/main/images/0_stars.png"

# Fetch the image
response = requests.get(image_url)

# Check the response status and act accordingly
if response.status_code == 200:
    # Open and display the image
    image = Image.open(BytesIO(response.content))
    image.show()
else:
    print(f"Failed to retrieve the image. HTTP status code: {response.status_code}")
    print("Response content:", response.text)
