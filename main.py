from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from audio_processing import process_data
from prediction import predict_class
import matplotlib.pyplot as plt

#videoyu wav a cevirme
from video_to_wav import video_to_audio
video_file = "video.mp4"
audio_file = "output_audio.wav" #burasi predict fonksiyonuna gonderilecek

video_to_audio(video_file, audio_file)


#sesi texte cevirme
from speech_to_text import transcribe_audio

audio_file_path = "output_audio.wav"
transcript = transcribe_audio(audio_file_path)
print("Transcript:", transcript)

# datayi yukleme
csv_file_path = "archive(3)\\train.csv"
audio_dataset_path = 'archive(3)\\train'
X_train, X_test, y_train, y_test, label_encoder = process_data(csv_file_path, audio_dataset_path)

# model olusturma
num_labels = len(label_encoder.classes_)
model = Sequential([
    Dense(125, input_shape=(40,), activation='relu'),
    Dropout(0.5),
    Dense(250, activation='relu'),
    Dropout(0.5),
    Dense(125, activation='relu'),
    Dropout(0.5),
    Dense(num_labels, activation='softmax')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# train
epochs = 300
batch_size = 32
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

# performans olcme
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", test_accuracy)

# tahmin
audio_file_path = "archive(3)\\test\\EYQE86873257023280.wav"
predicted_class = predict_class(audio_file_path, model, label_encoder)
print("Predicted Class:", predicted_class)

