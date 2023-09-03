from flask import Flask, request, jsonify, Response
from yt_dlp import YoutubeDL
import jax
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import tempfile
import os


pipeline = FlaxWhisperPipline("openai/whisper-medium", dtype=jnp.float16)
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("./jax_cache")

app = Flask(__name__)

@app.route('/process_video_url', methods=['POST'])
def process_video_url():
    if request.method == 'POST':
      data = request.json  # Request data is already parsed as JSON
      if 'video_url' in data:
          video_url = data['video_url']
          with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as temp_audio_file:
                temp_audio_filename = temp_audio_file.name
                with YoutubeDL({'overwrites':True, 'format':'bestaudio[ext=m4a]', 'outtmpl':f'./{temp_audio_filename}'}) as ydl:
                  ydl.download(video_url)
                # convert the audio
                outputs = pipeline(f"./{temp_audio_filename}")
                # Process the audio to text
                text = outputs["text"]
                os.remove(temp_audio_filename)
                # Send audio as a response
                return Response(text, content_type='text/plain')

      else:
          return jsonify({'error': 'No youtube video url provided'}), 400

@app.route('/process_audio', methods=['POST'])
def process_audio():
  if request.method == 'POST':
    audio_file = request.files['audio_file']
    if audio_file:
       with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
                temp_audio_filename = temp_wav_file.name
                audio_file.save(temp_audio_filename)
                output = pipeline(temp_audio_filename, task="transcribe", return_timestamps=False)
                text = output["text"]
                os.remove(temp_audio_filename)
                return Response(text, content_type='text/plain')
    else:
      return jsonify({'error': 'No audio file provided'}), 400

@app.route('/process_audio_timestamp', methods=['POST'])
def process_audio_timestamp():
  if request.method == 'POST':
    audio_file = request.files['audio_file']
    if audio_file:
      with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
                temp_audio_filename = temp_wav_file.name
                audio_file.save(temp_audio_filename)
                outputs = pipeline(temp_audio_filename,  task="transcribe", return_timestamps=True)
                chunks = outputs["chunks"]
                os.remove(temp_audio_filename)
                print(chunks)
                return jsonify(chunks)
    else:
      return jsonify({'error': 'No audio file provided'}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)