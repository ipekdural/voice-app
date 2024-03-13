#video vaw dosyasina donusturuluyor
import os
import subprocess

def video_to_audio(video_file, audio_file):
    try:
        if not os.path.exists(video_file):
            print("Video file not found.")
            return
        command = f"ffmpeg -i {video_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_file}"
        subprocess.run(command, shell=True, check=True)
        print("Audio extracted successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")
