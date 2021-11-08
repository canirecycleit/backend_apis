from io import BytesIO

import mlflow.keras
import numpy as np
from fastapi import FastAPI, File, UploadFile
from mlflow.tracking import MlflowClient
from PIL import Image

app = FastAPI()

model = None


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/list")
async def list_models():
    client = MlflowClient()
    return client.list_registered_models()


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())

    image_width = 256
    image_height = 192

    image = np.asarray(image.resize((image_width, image_height)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0

    model_name = "ciri_trashnet_model"
    model_version = 1

    global model
    if not model:
        model = mlflow.keras.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )

    pred = model.predict(image)
    class_index = np.argmax(pred, axis=1)[0]
    print(pred)
    print(class_index)

    return {"prediction": int(class_index)}
