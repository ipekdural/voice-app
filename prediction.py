import librosa
import numpy as np
from keras.models import load_model

def predict_class(audio_file_path, model, label_encoder):
    sound_signal, sample_rate = librosa.load(audio_file_path, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    result_array = model.predict(mfccs_scaled_features)
    predicted_class_index = np.argmax(result_array[0])
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_class
