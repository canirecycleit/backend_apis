from fastapi import FastAPI
from mlflow.tracking import MlflowClient

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/list")
async def list_models():
    client = MlflowClient()
    return client.list_registered_models()
