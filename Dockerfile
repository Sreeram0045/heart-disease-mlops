# 1. Use your exact Python version (Slim version is faster to download)
FROM python:3.10.12-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
# We use --no-cache-dir to keep the Docker image file size small
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your necessary folders into the container
# We don't copy the data folder since the model is already trained!
COPY ./app /code/app
COPY ./src /code/src
COPY ./models /code/models

# 5. Expose port 8000 for the web server
EXPOSE 8000

# 6. Start the FastAPI server using Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
