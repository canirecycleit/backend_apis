FROM python:3.9

WORKDIR /app
COPY . /app

# Update PIP & install requirements
RUN python -m pip install --upgrade pip
RUN pip install --upgrade -r requirements.txt

# Run the command on container startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]