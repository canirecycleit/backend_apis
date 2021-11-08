FROM python:3.9

WORKDIR /app
COPY . /app

# Update PIP & install requirements
RUN python -m pip install --upgrade pip

# Install with DOCKER_BUILDKIT caching
# https://pythonspeed.com/articles/docker-cache-pip-downloads/
RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade -r requirements.txt

# Run the command on container startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload", "--port", "80"]