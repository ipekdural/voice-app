import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def extract_features(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_scaled_features = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs_scaled_features

def process_data(csv_file_path, audio_dataset_path):
    metadata = pd.read_csv(csv_file_path)
    extracted_features = []
    for _, row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path), str(row["file_path"]))
        extracted_features.append([extract_features(file_name), row["class"]])

    extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
    X = np.array(extracted_features_df['feature'].tolist())
    y = np.array(extracted_features_df['class'].tolist())

    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test, label_encoder
