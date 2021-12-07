"""Configuration settings for Application API"""

import os

from dotenv import load_dotenv

load_dotenv()

# Define ML configurations:
VALIDATION_PCNT = os.environ.get("VALIDATION_PCNT", 0.2)
BATCH_SIZE = os.environ.get("BATCH_SIZE", 128)
IMAGE_WIDTH = os.environ.get("IMAGE_WIDTH", 256)
IMAGE_HEIGHT = os.environ.get("IMAGE_HEIGHT", 192)

# Model metadata:
MODEL_NAME = os.environ.get("MODEL_NAME", "ciri_trashnet_model")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")

# Image Store
GCS_PROJECT_NAME = os.environ.get("GCS_PROJECT_NAME", "CIRI")
GCS_DATA_BUCKET = os.environ.get("GCS_DATA_BUCKET", "canirecycleit-data")
