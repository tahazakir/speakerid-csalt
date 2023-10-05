# !pip install resemblyzer - done
# !pip install pyeer - done
# !pip install ffmpeg-python - done

import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
from pywebio.input import *
from pywebio.output import *
import pyaudio
import wave
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

put_markdown('## Speaker Identification Demo')

# function for audio recording

@use_scope('recorder_scope', clear=True)
def recorder(name, recording_number, record_seconds):

    # the file name output you want to record into
    filename = str(name) + str(recording_number) + ".wav"
    # set the chunk size of 1024 samples
    chunk = 1024
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, change to 2 if you want stereo
    channels = 1
    # 44100 samples per second
    sample_rate = 44100
    # record_seconds = 5
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    put_text("Recording")
    for i in range(int(sample_rate / chunk * record_seconds)):

        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    put_text("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()

    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()

# validation for number of users


def user_validation(number_of_users):
    if number_of_users < 2:
        return "Please enter a number greater than 1"

# validation for number of audios


def audio_validation(number_of_audios):
    if number_of_audios < 1:
        return "Please enter a number greater than 0"
    if number_of_audios > 4:
        return "Please enter a number less than 5"

# validation for username


def username_validation(username):
    if username == "":
        return "You can't have no name. Please enter something."


# start record func


@use_scope("button_scope", clear=True)
def start_record(number_of_audios, username, length_of_recording, aud_num):
    if aud_num <= number_of_audios:
        message = 'Start Recording ' + \
            str(username) + "\'s " + 'Audio Number ' + str(aud_num)
        put_button(message, onclick=lambda: recorder(
            username, aud_num, length_of_recording))
        confirm = actions('Confirm to save audio?', [
                          'Confirm'], help_text='You will not be able to resubmit audio')
        clear("recorder_scope")
        if confirm == 'Confirm':
            aud_num += 1
            start_record(number_of_audios, username,
                         length_of_recording, aud_num)
    else:
        confirm = actions('Registration for ' +
                          str(username) + ' complete!', ['Next'])
        return

# register one user attempt 2


username_list = []

# made dictionary for users with username 
users_dict = {}

def register_user(number_of_audios, length_of_recording, aud_num=1):
    username = input("Enter name for this user:", validate=username_validation)
    username_list.append(username)
    aud_num = 1
    start_record(number_of_audios, username, length_of_recording, aud_num)

# register all users


@use_scope('register_all_scope')
def register_all(number_of_users, number_of_audios, length_of_recording, user_num):
    if user_num <= number_of_users:
        register_user(number_of_audios, length_of_recording, aud_num=1)
        user_num += 1
        register_all(number_of_users, number_of_audios,
                     length_of_recording, user_num)
    else:
        confirm = actions('Registration for all complete!', ['Next'])
        return


@use_scope("loading_scope")
def loading():
    put_text("Processing...")

# resemblyzer --------------------------------------------------------------


speaker_embed_list = []
encoder = VoiceEncoder()


@use_scope('resemblyzer_scope', clear=True)
def resemblyzer_magic(number_of_audios):
    # Group the wavs per speaker and load them using the preprocessing function provided with
    # resemblyzer to load wavs in memory. It normalizes the volume, trims long silences and resamples
    # the wav to the correct sampling rate.
    speaker_wavs_list = []
    wav_fpaths = []
    for name in username_list:
        for i in range(number_of_audios):
            x = i+1
            path = str(name) + str(x) + '.wav'
            # get the paths where audio files are saved
            wav_fpaths.append(Path(path))
        # pre-processed audios ki dictionary banjati with speaker as key and audio as pair
        speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                        groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"),
                                lambda wav_fpath: wav_fpath.parent.stem)}
        speaker_wavs_list.append(speaker_wavs)

    # make a list of the pre-processed audios ki arrays
    for sp_wvs in speaker_wavs_list:
        speaker_embed_list.append(
            np.array([encoder.embed_speaker(wavs) for wavs in sp_wvs.values()]))


@use_scope('test_scope', clear=True)
def test_pp():
    loading()
    # does the same as above for the test file
    # hm list(Path("test1.wav")) hona chahiye but Path object is not iterable
    wav_fpaths = []
    test_fpath = Path("test1.wav")
    wav_fpaths.append(test_fpath)
    test_pos_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                     groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"),
                             lambda wav_fpath: wav_fpath.parent.stem)}
    test_pos_emb = np.array([encoder.embed_speaker(wavs)
                            for wavs in test_pos_wavs.values()])

    # calculates cosine similarity between the ground truth (test file) and registered audios
    speakers = {}
    val = 0
    for spkr_embd in speaker_embed_list:
        key_val = username_list[val]
        spkr_sim = cosine_similarity(spkr_embd, test_pos_emb)[0][0]
        speakers[key_val] = spkr_sim
        val += 1

    norm = [float(i)/sum(speakers.values()) for i in speakers.values()]
    for i in range(len(norm)):
        key_val = username_list[i]
        speakers[key_val] = norm[i]

    clear("loading_scope")
    remove("loading_scope")
    identified = max(speakers, key=speakers.get)
    print("\nThe identity of the test speaker:\n", identified, "with a similarity with test of",
          speakers[identified]*100, "percent match as compared to all.")
    put_markdown('## Test Audio Belonged To: {}'.format(identified))
# -------------------------------------------------------------------


@use_scope('test_scope', clear=True)
def test_taking(length_of_recording):
    put_button("Record Test Audio", onclick=lambda: recorder(
        "test", 1, length_of_recording))
    test_save = actions('Confirm to save audio?', [
                        'Confirm'], help_text='You will not be able to resubmit audio')


@use_scope('pp_load', clear=True)
def pp_load():
    put_text("Please Wait. Processing registration audios.")


number_of_users = input("Enter the number of users for this session:",
                        type=NUMBER, validate=user_validation)

number_of_audios = input(
    "Enter the number of audios each user will register:", type=NUMBER, validate=audio_validation)

length_of_recording = select(
    label="Select the length of each recording (in seconds)", options=[3, 5, 7])

register_all(number_of_users, number_of_audios, length_of_recording, 1)
pp_load()
resemblyzer_magic(number_of_audios)

clear("pp_load")
remove("pp_load")


clear("register_all_scope")
remove("register_all_scope")

clear("recorder_scope")
remove("recorder_scope")

put_markdown('## Test Audio Time')

test_taking(length_of_recording)
clear("test_scope")
remove("test_scope")

clear("recorder_scope")
remove("recorder_scope")

generate_button = actions(
    'Click to perform Speaker Recognition Magic', ['Generate Result!'])
if generate_button == 'Generate Result!':
    test_pp()

# clear("resemblyzer_scope")
# remove("resemblyzer_scope")

while(True):
    again = select(
        label="Do you want to enroll another test audio?", options=["Yes", "No"])
    if again == "Yes":
        test_taking(length_of_recording)
        clear("test_scope")
        remove("test_scope")

        clear("recorder_scope")
        remove("recorder_scope")

        generate_button = actions(
            'Click to perform Speaker Recognition Magic', ['Generate Result!'])
        if generate_button == 'Generate Result!':
            test_pp()
    else:
        put_text("Thank you for trying out our system!")
        break

#register_user(number_of_audios, length_of_recording, 1)

#register_all(number_of_users, number_of_audios, length_of_recording)

#put_button('Start Recording Test Audio', onclick=lambda: recorder("test", 1, length_of_recording))
