from google.cloud import speech_v1p1beta1 as speech

def transcribe_audio(audio_file_path):

    client = speech.SpeechClient()

    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript

if __name__ == "__main__":
    audio_file_path = "output_audio.wav"
    transcript = transcribe_audio(audio_file_path)
    print("Transcript:", transcript)
