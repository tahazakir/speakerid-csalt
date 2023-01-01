import sounddevice as sd
import numpy as np
import time
from pywebio.input import *
from pywebio.output import *
from resemblyzer import preprocess_wav, VoiceEncoder


# Set the audio recording parameters
fs = 16000  # Sample rate
duration = 3  # Recording duration in seconds


def username_validation(username):
    if username == "":
        return "You can't have no name. Please enter something."

def start_rec(fs, duration):
    put_text("Recording")
    aud = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = np.ravel(aud, order='C')
    put_text("Finished recording.")
    return audio

username_l = []
username = input("Enter name for this user:", validate=username_validation)
username_l.append(username)

audio1 = start_rec(fs, duration)

username2 = input("Enter name for this user:", validate=username_validation)
username_l.append(username2)

audio2 = start_rec(fs, duration)

# initialize the voice encoder
encoder = VoiceEncoder()

# # preprocess the audio signals using the preprocess_wav function
prep_audio1 = preprocess_wav(audio1, source_sr=fs)
prep_audio2 = preprocess_wav(audio2, source_sr=fs)
# # prep_audiot = preprocess_wav(audiot, source_sr=fs)

# # # encode the audio signals using the encoder
encoded_audio1 = encoder.embed_utterance(audio1)
encoded_audio2 = encoder.embed_utterance(audio2)
# # encoded_audiot= encoder.embed_utterance(prep_audiot)

# # compare the encoded audio signals to determine the speaker
speaker1_similarity = np.dot(encoded_audio1, encoded_audio2)

# # # compare the encoded audio signals to determine the speaker
# # speaker2_similarity = np.dot(encoded_audio2, encoded_audiot)

print("Similarity Score", speaker1_similarity)
# print("Similarity Score for second Spekaer", speaker2_similarity)