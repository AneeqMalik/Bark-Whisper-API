# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Update the package list
RUN apt-get update && apt-get install -y git

# Install ffmpeg
RUN apt-get install -y ffmpeg

# Set the working directory to /app
WORKDIR /app

# Copy all files except those listed in .dockerignore
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install JAX and JAXlib with CUDA support
RUN pip install -U "jax[cuda11_local]==0.4.13" "jaxlib[cuda11_local]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install git+https://github.com/camenduru/whisper-jax.git datasets soundfile librosa yt_dlp cached_property

# Run app.py when the container launches
CMD ["python", "main.py"]
