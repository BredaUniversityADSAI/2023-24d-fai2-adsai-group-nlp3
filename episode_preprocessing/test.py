import datetime
import io
import wave

import numpy as np
import pandas as pd
import pydub
import webrtcvad
import whisper
from pydub import AudioSegment
from scipy.signal import resample


def load_audio_file(file_path) -> io.BytesIO:
    """
    Function that loads audio from file path.

    Input:
        file_path: file path to the file containing audio

    Output:
        wav_file: io.BytesIO object with .wav file stored
    """
    audio = AudioSegment.from_file(file_path)
    split_audio = audio.split_to_mono()

    wav_file = io.BytesIO()
    split_audio[1].export(wav_file, format="wav")

    return wav_file


def get_segments(
    audio_segment: pydub.AudioSegment,
    min_silence_len: int = 1000,
    silence_threshold: int = -16,
    keep_silence=100,
) -> list[pydub.AudioSegment]:
    """
    Function that splits pydub.AudioSegment into list
    of pydub.AudioSegment objects based on detected silence.

    Input:
        audio_segment: a pydub.AudioSegment object.
        min_silence_len: min time of silence (in ms) to split the audio. default: 1000
        silence_threshold: threshold in dB to treat the audio as silence. default: -16
        keep_silence: (in ms or True/False) margin of silence to keep
            (if True, all silence is kept, if False, none is). default: 100
    """
    segments = pydub.silence.split_on_silence(
        audio_segment, min_silence_len, silence_threshold, keep_silence
    )

    return segments


def transcribe_translate_segments(
    segments: list[pydub.AudioSegment],
    transcription_model_type: str = "base",
    use_fp16: bool = True,
    episode_value=datetime.datetime.now().date(),
) -> pd.DataFrame:
    """
    Function that transcribes and translates a pydub.AudioSegment
    using whisper model of the chosen size.

    Input:
        segments: list of pydub.AudioSegment objects returned by get_segments function
        transcription_model_type: size of whisper model used for
            transcription and translation,
            see: https://pypi.org/project/openai-whisper/. default: base
        fp16 (bool): Whether to use FP16 format for model prediction,
            needs to be False for CPU. Defaults to True.
        episode_value: a value assigned to the whole episode in the pandas.DataFrame,
            default: current date (YYYY-MM-DD)
    Output:
        transcripts: pandas.DataFrame with transcription in english,
            numbered segments, and date
    """

    # load the whisper model for transcription
    transcription_model = whisper.load_model(transcription_model_type)

    # get numbered segment transcriptions and translations
    transcripts = map(
        lambda x: (
            x[0],
            transcription_model.transcribe(x[1], fp16=use_fp16, language="en"),
        ),
        enumerate(segments, start=1),
    )

    transcripts = pd.DataFrame(transcripts, columns=["segment_number", "transcription"])
    transcripts.insert(loc=0, column="date", value=episode_value)

    return transcripts


def split_into_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that splits text segments into separate rows.

    Input:
        df: pandas.DataFrame with segment level text in each row

    Output: df: pandas.DataFrame with sentence level text in each row
    """

    # TODO: need to split sentences first to get a list
    """
    to check: spacy sentencizer pipeline component


    import spacy
    nlp = spacy.load('en')

    text = '''Your text here'''
    tokens = nlp(text)

    for sent in tokens.sents:
        print(sent.string.strip())
    """

    df = (
        df.explode(column="transcription", ignore_index=False)
        .reset_index(drop=True)
        .rename(columns={"transcription": "sentence"})
    )

    return df


def export_csv(df: pd.DataFrame, format: str = "csv", output_path=None) -> None:
    """
    Function that saves a pandas.DataFrame. Both csv and json formats are available
    to chose as well as a default output file path.

    Input:
        df: pandas.DataFrame to be saved
        format: format of saved data, csv or json, default: csv
        output_path: file path to save the data,
            default: None (saved in current directory as output.<chosen extension>)
    """
    if output_path is None:
        out_path = f"output.{format}"

    if format == "json":
        df.to_json(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)


audio_file = load_audio_file("ep_1.mov")
print("got audio file")

# get vad with strict: 0/3
vad = webrtcvad.Vad(0)
print("created vad object")

frame_time = 30  # ms


with wave.open(audio_file, "rb") as wave_file:
    print("opened audio file")

    original_sample_rate = wave_file.getframerate()
    print(f"original sample_rate: {original_sample_rate}")
    target_sample_rate = 32_000
    print(f"target sample rate: {target_sample_rate}")
    number_of_frames = wave_file.getnframes()
    print(f"number of frames: {number_of_frames}")

    audio_data = wave_file.readframes(number_of_frames)

resampled_number_of_frames = int(
    number_of_frames * target_sample_rate / original_sample_rate
)
print(f"resampled number of frames: {resampled_number_of_frames}")

t = np.frombuffer(audio_data, dtype=np.int16)
resampled_audio_data = resample(
    t, int(number_of_frames * target_sample_rate / original_sample_rate)
)
print("resampled")
print(f"resampled data: {resampled_audio_data[:10]}")

# Specify audio parameters
number_of_channels = 1  # Assuming mono audio
sample_width = 2  # 16-bit audio
byte_resampled_audio = resampled_audio_data.tobytes()

# Create a WAV file
with wave.open("downsampled_audio.wav", "wb") as wf:
    wf.setnchannels(number_of_channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(target_sample_rate)
    wf.writeframes(byte_resampled_audio)


"""    frame_size = int(frame_time * target_sample_rate)  # Calculate frame size
    overlap = 0.5  # 50% overlap

    # Generate frame indices with overlap
    start_indices = np.arange(
        0,
        len(resampled_audio_data) - frame_size + 1, int(frame_size * (1 - overlap))
    )

    # Split audio into frames
    frames = [resampled_audio_data[i:i + frame_size] for i in start_indices]

    print(frames[0])"""
