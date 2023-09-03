from flask import Flask, request, jsonify, Response
from bark import SAMPLE_RATE, generate_audio, preload_models
import scipy.io.wavfile as wavfile
import tempfile
import os

# Loading the Models
preload_models()

app = Flask(__name__)

@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        data = request.json  # Request data is already parsed as JSON

        if 'text' in data:
            text = data['text']
            # Generate the audio data from the text
            text_prompt = text
            audio_array = generate_audio(text_prompt)

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
                temp_wav_filename = temp_wav_file.name
                wavfile.write(temp_wav_filename, SAMPLE_RATE, audio_array)

                # Read the temporary WAV file
                with open(temp_wav_filename, 'rb') as wav_file:
                    audio_data = wav_file.read()

            # Delete the temporary WAV file
            os.remove(temp_wav_filename)

            # Send audio as a response
            return Response(audio_data, content_type='audio/wav')

        else:
            return jsonify({'error': 'No text data provided'}), 400

@app.route('/process_text_custom_voice', methods=['POST'])
def process_text_custom_voice():
    if request.method == 'POST':
        data = request.json  # Request data is already parsed as JSON

        if 'text' in data:
            text = data['text']
            history_prompt = data['history_prompt']
            # Generate the audio data from the text
            text_prompt = text
            audio_array = generate_audio(text_prompt, history_prompt=history_prompt)

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
                temp_wav_filename = temp_wav_file.name
                wavfile.write(temp_wav_filename, SAMPLE_RATE, audio_array)

                # Read the temporary WAV file
                with open(temp_wav_filename, 'rb') as wav_file:
                    audio_data = wav_file.read()

            # Delete the temporary WAV file
            os.remove(temp_wav_filename)

            # Send audio as a response
            return Response(audio_data, content_type='audio/wav')

        else:
            return jsonify({'error': 'No text data provided'}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
