from flask import Flask, request, jsonify,render_template
import os
app = Flask(__name__)
import numpy as np
from transformers import pipeline
import torchaudio
import soundfile as sf
import librosa


def predict_bird_name(audio_file_path):
    print("predicting on audio path: ", audio_file_path)
    
    if not audio_file_path.endswith(".wav"):
        waveform, sample_rate = torchaudio.load(audio_file_path)
        wav_file_path = "C:\\Users\\beher\\Downloads\\B.Tech Project Flask App\\audio.wav"
        sf.write(wav_file_path, waveform.t().numpy(), sample_rate)
        audio_file_path = wav_file_path


    pipe = pipeline("audio-classification", model="Holonium27/distilhubert-finetuned-gtzan")

    target_sampling_rate = 16000
    audio, sr = librosa.load(audio_file_path, sr=None)

    # Resample 
    if sr != target_sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sampling_rate)

    # Normalize 
    audio = (audio - np.mean(audio)) / np.std(audio)

    audio_data = [{"raw": audio, "sampling_rate": target_sampling_rate}]

    # Make predictions
    predictions = pipe(audio_data)
    top_predictions = sorted(predictions[0], key=lambda x: x['score'], reverse=True)[:3]

    # Labelling
    for item in top_predictions:
        label = item['label']
        score = round(item['score'], 2)
        print(f"Label: {label}, Score: {score}")
    prediction = top_predictions[0]['label']   
    return prediction

@app.route("/", methods=["GET","POST"])
def dashboard():
    if request.method == 'POST':
        print("form posted")
        if 'audioFile' not in request.files:
            return jsonify({"prediction": "No audio file provided."})

        audio_file = request.files['audioFile']

        if audio_file.filename == '':
            return jsonify({"prediction": "No selected file."})

        if audio_file:
            file_path = os.path.join(app.root_path, audio_file.filename)
            audio_file.save(file_path)
    
            print("audio downloaded successfully")

            prediction = predict_bird_name(audio_file.filename)
            # os.remove(audio_file.filename)
            decoded = ['abhori1 - African Black-headed Oriole',
                        'abythr1 - Abyssinian Thrush',
                        'afdfly1 - African Dusky Flycatcher',
                        'afecuc1 - African Emerald Cuckoo',
                        'affeag1 - African Fish-Eagle',
                        'afghor1 - African Grey Hornbill',
                        'afmdov1 - Mourning Collared Dove',
                        'afpfly1 - African Paradise-Flycatcher',
                        'afpwag1 - African Pied Wagtail',
                        'afrgos1 - African Goshawk']

            prediction = decoded[int(prediction[-1])]
            print(prediction)
            image_path = prediction[:7]
            link = "https://ebird.org/species/" + prediction[:7]
            print(image_path,link)
            return render_template('prediction.html', prediction = prediction, image_path = image_path, link = link )
    return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)
