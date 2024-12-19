# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables to prevent Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the Hugging Face Hub CLI
RUN pip install --no-cache-dir huggingface_hub

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install 'accelerate>=0.26.0'

# Copy the application code into the container
COPY . .

# Set the entrypoint for the container to run your training script
ENTRYPOINT ["python", "train.py"]