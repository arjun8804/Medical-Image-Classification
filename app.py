
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np
import io

model = tf.keras.models.load_model(
    "models/pneumonia_mobilenetv2.keras"
)

app = FastAPI(
    title="Medical Image Classification API",
    description="Chest X-ray pneumonia classification API",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "message":
        "Medical Image Classification API is running",

        "model":
        "MobileNetV2"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:

        result = "PNEUMONIA"

    else:

        result = "NORMAL"

    return {

        "prediction": result,

        "pneumonia_probability":
        round(
            probability,
            4
        )
    }
