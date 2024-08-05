import json
import requests
import numpy as np
import speech_recognition as sr
from pathlib import Path
from django.shortcuts import redirect, render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.urls import reverse
from django.views.decorators.cache import cache_page
from recognizer.forms import AudioForm
from .process_audio import extract_features
import os

app_name = "recognizer"

# Model Server URL (Docker)
url = 'http://localhost:8502/v1/models/Speech-Emotion-Recognizer:predict'
cache_time_in_minutes = 10

@cache_page(60 * cache_time_in_minutes)
def index(request):
    return render(request, 'recognizer/index.html')

@cache_page(60 * cache_time_in_minutes)
def privacy(request):
    return render(request, 'recognizer/privacy.html')

def result(request):
    return render(request, 'recognizer/result.html')

def get_emotion_recording(request):
    if request.method == "POST":
        audio_data = request.FILES.get('data')
        if not audio_data:
            return JsonResponse({'error': 'No audio data provided'}, status=400)

        HOME_path = Path('media/recordings')
        path = default_storage.save(
            str(HOME_path / 'recording.wav'), ContentFile(audio_data.read())
        )
        recent_file_path = get_latest_file_path()

        if recent_file_path:
            features = process_audio(recent_file_path)
            transcription = get_transcription(recent_file_path)
            print(transcription['success'], transcription['transcription'])

            try:
                predictions = make_prediction(
                    [{"conv2d_input": features.tolist(), "keras_layer_input": transcription['transcription']}]
                )
            except requests.ConnectionError:
                return JsonResponse({'error': 'Model server connection failed'}, status=500)
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)

            result_url = reverse('recognizer:result')
            request.session['prediction'] = predictions
            return JsonResponse({'prediction': predictions, 'url': request.build_absolute_uri(result_url)})

    return render(request, 'recognizer/get_emotion.html')

def get_emotion_upload(request):
    if request.method == "POST":
        filename = request.FILES.get('record').name
        file_extension = filename.split('.')[-1]

        if file_extension.lower() != 'wav':
            return JsonResponse({'error': 'Invalid file format'}, status=400)

        form = AudioForm(request.POST, request.FILES or None)

        if form.is_valid():
            form.save()
            recent_file_path = get_latest_file_path()

            if recent_file_path:
                features = process_audio(recent_file_path)
                transcription = get_transcription(recent_file_path)
                print(transcription['success'], transcription['transcription'])

                try:
                    predictions = make_prediction(
                        [{"conv2d_input": features.tolist(), "keras_layer_input": transcription['transcription']}]
                    )
                except requests.ConnectionError:
                    return JsonResponse({'error': 'Model server connection failed'}, status=500)
                except Exception as e:
                    return JsonResponse({'error': str(e)}, status=500)

                request.session['prediction'] = predictions
                return redirect("recognizer:result")

    else:
        form = AudioForm()

    return render(request, 'recognizer/get_emotion.html', {'form': form})

def make_prediction(instances):
    emotion_class = ['Angry', 'Happy', 'Neutral', 'Sad']
    data = json.dumps({"signature_name": "serving_default", "instances": instances})
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        predictions = result.get('predictions', [])
        if not predictions:
            raise ValueError("No predictions found in the response")
        return emotion_class[np.argmax(predictions)]
    except requests.ConnectionError as conn_err:
        print(f"ConnectionError: {conn_err}")
        raise ConnectionError("Failed to connect to the model server")
    except requests.HTTPError as http_err:
        print(f"HTTPError: {http_err}")
        raise ConnectionError(f"HTTP error occurred: {http_err}")
    except ValueError as value_err:
        print(f"ValueError: {value_err}")
        raise ValueError(f"Error in prediction response: {value_err}")
    except Exception as err:
        print(f"OtherError: {err}")
        raise ConnectionError(f"Other error occurred: {err}")

def process_audio(audio_file):
    features = extract_features(audio_file)
    return features

def get_transcription(audio_file):
    response = {"success": True, "error": None, "transcription": None}
    r = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
            response["transcription"] = r.recognize_google(audio, language="en-US")
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["success"] = False
        response["error"] = "Unable to recognize speech"

    os.remove(audio_file)
    return response

def get_latest_file_path():
    HOME_path = Path('media/recordings')
    files = list(HOME_path.glob('*.wav'))
    if files:
        latest_file = max(files, key=os.path.getctime)
        return str(latest_file)
    return None







# Define the JSON payload
payload = {
    "signature_name": "serving_default",
    "instances": [
        {
            "input_tensor": [
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            ]
        }
    ]
}
