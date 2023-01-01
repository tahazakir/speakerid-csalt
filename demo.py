import sounddevice as sd
import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder
import time

# Set the audio recording parameters
fs = 16000  # Sample rate
duration = 3  # Recording duration in seconds

# Record first audio input
print('Recording first audio...')
time.sleep(2.5)
print("Start Speaking!")
audio1 = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()

audio1 = np.ravel(audio1, order='C')

# Record second audio input
print('Recording second audio...')
time.sleep(2.5)
print("Start Speaking!")
audio2 = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()

audio2 = np.ravel(audio2, order='C')

# # Record test audio input
# print('Recording test audio...')
# time.sleep(2.5)
# print("Start Speaking!")
# audiot = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
# sd.wait()

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