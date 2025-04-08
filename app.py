import numpy as np 
import librosa
from tensorflow.keras.models import load_model
import os
from flask import Flask, request, jsonify
import noisereduce as nr
import pywt
from scipy.signal import butter, filtfilt
import joblib
import random

app = Flask(__name__)

respiratory_classifier=load_model('respiratory_sound_classifier.h5')
scaler = joblib.load('scaler.pkl')

def highpass_filter(y, sr, cutoff=100):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, y)


def wavelet_denoise(y, wavelet='db4', level=3):
    coeffs = pywt.wavedec(y, wavelet, mode="per")
    coeffs[1:] = [pywt.threshold(c, np.std(c), mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")



def denoising_audio(filename):
  sound, sample_rate = librosa.load(filename)
  reduced_noise = nr.reduce_noise(y=sound, sr=sample_rate,prop_decrease=0.7)
  reduced_noise = highpass_filter(reduced_noise, sample_rate)
  reduced_noise = wavelet_denoise(reduced_noise)
  return reduced_noise,sample_rate

def audio_features(filename):
  sound,sample_rate = denoising_audio(filename)
  stft = np.abs(librosa.stft(sound))

  mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40),axis=1)
  chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
  mel = np.mean(librosa.feature.melspectrogram(y=sound, sr=sample_rate), axis=1)
  contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate),axis=1)
  tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate),axis=1)

  concat = np.concatenate((mfccs,chroma,mel,contrast,tonnetz))
  return concat

def preprocess_audio(audio_path):
    audio = audio_features(audio_path)
    audio = np.expand_dims(audio, axis=0)
    audio = scaler.transform(audio)
    return audio

def predict_audio_with_temperature(audio_path, temperature=0.3):
    matrix_index = ["COPD", "Healthy", "URTI", "Bronchiectasis", "Pneumonia", "Bronchiolitis"]
    audio = preprocess_audio(audio_path)
    logits = respiratory_classifier.predict(audio)
    # Apply temperature scaling
    adjusted_logits = logits / temperature
    probabilities = np.exp(adjusted_logits) / np.sum(np.exp(adjusted_logits))
    prob = np.max(probabilities)
    prediction = np.argmax(probabilities)
    return matrix_index[prediction], int(prob * 100)

def handling_output(audio_path):
    prediction,prob=predict_audio_with_temperature(audio_path,temperature=random.random())
    if prediction=="COPD":
        return f"Diagnosis: You have a {prob}% probability of having {prediction}.\nRecommendation:\nQuit smoking immediately to prevent progression.\nAvoid air pollutants and secondhand smoke.\nEngage in pulmonary rehabilitation for better breathing techniques.\nGet annual flu and pneumonia vaccines."
    elif prediction=="URTI":
        return f"Diagnosis: You have a {prob}% probability of having {prediction}.\nRecommendation:\nGet plenty of rest and stay hydrated.\nUse over-the-counter medications for symptom relief.\nPractice good hand hygiene to prevent spreading the infection."
    elif prediction=="Bronchiectasis":
        return f"Diagnosis: You have a {prob}% probability of having {prediction}.\nRecommendation:\nPerform airway clearance techniques to remove mucus.\nEngage in regular exercise to strengthen lung function.\nAvoid exposure to dust, smoke, and other pollutants.\nStay up-to-date on vaccinations to prevent infections."
    elif prediction=="Pneumonia":
        return f"Diagnosis: You have a {prob}% probability of having {prediction}.\nRecommendation:\nSeek immediate medical attention if symptoms worsen.\nComplete the full course of prescribed antibiotics if bacterial.\nGet adequate rest and hydration to recover.\nAvoid smoking and exposure to irritants."
    elif prediction=="Bronchiolitis":
        return f"Diagnosis: You have a {prob}% probability of having {prediction}.\nRecommendation:\nMonitor breathing and hydration, especially in children.\nUse a humidifier to ease breathing difficulties.\nAvoid crowded places to reduce the risk of further infection."
    else: 
        return f"Diagnosis: You are {prediction}.\n"




@app.route('/predict',methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error":"No Audio file Provided."}),400
    
    file=request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No Audio file provided."}), 400
    
    temp_path='temp_audio.wav'
    file.save(temp_path)
    

    try:
        result=handling_output(temp_path)
    
    except Exception as e:
        os.remove(temp_path)
        return jsonify({'error':str(e)}),500
    
    os.remove(temp_path)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)