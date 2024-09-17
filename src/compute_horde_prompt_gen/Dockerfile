# Start from a CUDA-enabled base image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip

# Install PyTorch Hugging Face Transformers and other necessary packages
RUN pip3 install torch 
RUN pip3 install transformers sentencepiece accelerate bitsandbytes

# Create an output folder
RUN mkdir /output

# Copy your Python script into the container
COPY saved_models/ /app/saved_models/
COPY *.py ./

# Set the entrypoint to run your script
ENTRYPOINT ["python3", "run.py"]
