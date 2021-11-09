import json
import os

import mlflow.keras
import numpy as np
from fastapi import FastAPI, File, UploadFile
from mlflow.tracking import MlflowClient
from starlette.middleware.cors import CORSMiddleware

from ciri_api.utils import read_imagefile

app = FastAPI()

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "CIRI Application APIs"}


@app.get("/models/list")
async def list_models():
    """Lists available models registered in MLFlow."""
    client = MlflowClient()
    return client.list_registered_models()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Provides a class prediction based on uploaded image."""

    # Convert uploaded file into image for prediction:
    image = read_imagefile(await file.read())

    model_name = "ciri_trashnet_model"
    model_stage = "Production"

    # TODO: Cache this so don't have to reload from disk:
    model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{model_stage}")

    client = MlflowClient()

    model_artifact_store = ""
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == model_stage:
            model_artifact_store = os.path.split(mv.source)[0]

    pred = model.predict(image)
    print(pred)
    class_index = int(np.argmax(pred, axis=1)[0])
    print(class_index)

    if model_artifact_store:
        label_mapping_file = os.path.join(model_artifact_store, "mapping.json")
        index2label = {}
        with open(label_mapping_file, "r") as f:
            index2label = json.loads(f.read())

        class_index = index2label[str(class_index)]

    print(class_index)
    return {"prediction": class_index}
