# Use an official Python runtime as a parent image (Python 3.11)
FROM python:3.11-slim

# Install Git
RUN apt-get update && apt-get install -y git

# Install the 'bark' package from GitHub
RUN pip install git+https://github.com/suno-ai/bark.git

# Set the working directory to /app
WORKDIR /app

# Copy all files except those listed in .dockerignore
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run main.py when the container launches
CMD ["python", "main.py"]
