import datetime
import io
import wave

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import spacy
import webrtcvad
import whisper
from pydub import AudioSegment
from tqdm import tqdm


class TBD:
    """
    TODO
    values to check later
    """

    pass


def load_audio_from_video(file_path: str, target_sample_rate: int) -> TBD:
    """
    A function that loads audio data from video file.
    Used by providing the path to the episode and desired sample rate.
    The function assumes the audio is multi-channel and
    automatically converts it to mono.

    Input:
        file_path (str): file path to the video
        target_sample_rate (int): the sample rate the audio file will be converted to

    Output:
        audio (TBD): mono audio file with specified sample rate
    """
    # load audio from video file
    audio = AudioSegment.from_file(file_path)

    # export the audio to in-memory object
    wav_file = io.BytesIO()
    audio.export(wav_file, format="wav")

    # load the audio and downsample it to target sample rate
    audio, sample_rate = librosa.load(wav_file, sr=target_sample_rate)
    print(f"got audio with sample rate: {sample_rate}")

    # convert the audio to mono
    audio = librosa.to_mono(audio)
    print("converted to mono")

    return audio


def get_segments_for_vad(
    audio: TBD, sample_rate: int, segment_seconds_length: float
) -> list[TBD]:
    """
    A function that adapts the audio file so that it is compatible with
    webrtcvad.Vad object. The sample rate must be 8kHz, 16kHz, or 32kHz, and
    the segment length must be 10ms, 20ms, or 30ms. It returns a list of
    audio segments of the chosen length.

    Input:
        audio (TBD): audio file obtained from load_audio_from_video function
        sample_rate (int): sample rate of the audio file
        segment_seconds_length (float): segment length in seconds

    Output:
        segments (list[TBD]): list of cutouts from the audio file
    """

    # get segment length in frame number
    segment_frames_length = int(segment_seconds_length * sample_rate)

    # get number of segments in the audio
    num_segments = len(audio) // segment_frames_length
    print(f"segment length: {segment_frames_length} frames")
    print(f"number of segments: {num_segments}")

    # split audio into list of segments
    segments = [
        audio[i * segment_frames_length : (i + 1) * segment_frames_length]
        for i in range(num_segments)
    ]
    print(f"split into {len(segments)} segments")

    return segments


def get_vad_per_segment(
    segments: list[TBD],
    vad_aggressiveness: int,
    sample_rate: int,
    segment_frames_length: int,
) -> np.array:
    """
    A function that decides whether an audio file preciously cut to segments using
    get_segments_for_vad function contains speech or not based on voice
    activation detection. VAD aggressiveness, is adjustable (int values from 0 to 3),
    and on top of that, audio sample rate and segment length in frames is required.

    Input:
        segments (list[TBD]): list with cut audio obtained from
            get_segments_for_vad function
        vad_aggressiveness (int): how aggressive should the function be
            when filtering out non-speech
        sample_rate (int): sample rate of the audio file
        segment_frames_length: segment length measured by number of frames

    Output:
        segments_is_speech (np.array): array with bool values representing
            detected speech for a given fragment. True means that the fragment
            contains speech
    """

    # instantiate VAD with aggressiveness from 0 to 3
    vad = webrtcvad.Vad(vad_aggressiveness)

    # prepare an in-memory location for segments
    segment_file = io.BytesIO()

    segments_is_speech = []
    for segment in tqdm(segments):
        # write the segments to in-memory location
        sf.write(segment_file, segment, sample_rate, format="wav", subtype="PCM_16")
        # open the segment with wave library
        segment_bytes_file = wave.open(segment_file, mode="rb")
        # get bytes data from the segment
        segment_bytes = segment_bytes_file.readframes(segment_frames_length)
        # detect speech in segment (outputs bool values)
        is_speech = vad.is_speech(segment_bytes, sample_rate)
        segments_is_speech.append(is_speech)

    # get np.array with speech bool values
    segments_is_speech = np.array(segments_is_speech)
    return segments_is_speech


def segment_number_to_frames(
    segment_number: int, sample_rate: int, segment_seconds_length: float
) -> int:
    """
    A function that converts segment number to the number of
    frames from the start of the audio file using segment number,
    audio sample rate, and segment length in seconds

    Input:
        segment_number (int): number of the segment from the np.array
            obtained from get_vad_per_segment function
        sample_rate (int): sample rate of the audio file
        segment_seconds_length (float): length of the audio segment in seconds

    Output:
        frames (int): the number of frames from the start of the audio
        file that corresponds to the segment number
    """
    return int(sample_rate * segment_seconds_length * segment_number)


def adjust_part_start_frame(start_part_frame: int, sample_rate: int) -> int:
    """
    A function that moves the start of the larger part (a couple of minutes)
    to start 0.125 seconds earlier if the new start does not go below 0.

    Input:
        start_part_frame (int): the frame number that corresponds to start of
            the segment with no speech detected
        sample_rate (int): sample rate of the audio file

    Output:
        start_part_frame (int): adjusted (if possible) start_part_frame value
    """
    if start_part_frame - int(0.125 * sample_rate) >= 0:
        start_part_frame = start_part_frame - int(0.125 * sample_rate)

    return start_part_frame


def adjust_part_end_frame(
    end_part_frame: int, sample_rate: int, full_audio_length_frames: int
) -> int:
    """
    A function that moves the end of the larger part (a couple of minutes)
    to end 0.125 seconds later if the new end does not go over
    the full audio duration. Used by get_frame_segments_from_vad_output function.

    Input:
        end_part_frame (int): the frame number that corresponds to start of
            the last used in this part segment with no speech detected
        sample_rate (int): sample rate of the audio file

    Output:
        end_part_frame (int): adjusted (if possible) end_part_frame value
    """
    if end_part_frame + int(0.125 * sample_rate) <= full_audio_length_frames:
        end_part_frame = end_part_frame + int(0.125 * sample_rate)

    return end_part_frame


def get_frame_segments_from_vad_output(
    speech_array: np.array,
    sample_rate: int,
    segment_seconds_length: float,
    full_audio_length_frames: int,
) -> list[tuple[int, int]]:
    """
    A function that connects the small segments (10/20/30ms) into larger (5-6 min)
    parts based on the results from get_vad_per_segment function.
    It combines small fragment until a specified threshold is reached,
    and the start and end in frames is saved. The function returns
    (start, end) pairs in a list.

    Input:
        speech_array (np.array): np.array with bool values obtained from
            get_vad_per_segment function
        sample_rate (int):
        segment_seconds_length (float):
        full_audio_length_frames (int): the total number of frames in
            the entire audio file

    Output:
        cut_segments_frames (list[tuple[int, int]]): list of
            (start, end) frame number pairs
    """

    # this returns a tuple for some reason
    # np.array with segment numbers for segment where we can cut the audio
    cutable_segments = np.where(speech_array is False)[0]

    # a target so we have an amount of frames to aim for
    # TODO make this a separate function
    part_min_length_frame = segment_number_to_frames(10_000, 32_000, 0.03)

    # frame number from which the part will start
    segment_start_frame = 0
    # start and end of each part in frames
    cut_segments_frames = []

    for segment_number in cutable_segments:
        # get end of the segment in frames
        segment_end_frame = segment_number_to_frames(
            segment_number, sample_rate, segment_seconds_length
        )

        # if the total time (in frames) from start is long enough:
        if segment_end_frame - segment_start_frame >= part_min_length_frame:
            # start a bit earlier and end a bit later if possible
            adjusted_segment_start_frame = adjust_part_start_frame(
                segment_start_frame, sample_rate
            )
            adjusted_segment_end_frame = adjust_part_end_frame(
                segment_end_frame, sample_rate, full_audio_length_frames
            )

            cut_segments_frames.append(
                (adjusted_segment_start_frame, adjusted_segment_end_frame)
            )
            segment_start_frame = segment_end_frame

    # full_audio_length_frames = int(
    #    librosa.get_duration(y=audio, sr=sample_rate) * sample_rate
    # )
    cut_segments_frames.append((adjusted_segment_start_frame, full_audio_length_frames))

    return cut_segments_frames


def clean_transcript_df(
    df: pd.DataFrame, episode_value=datetime.datetime.now().date()
) -> pd.DataFrame:
    df.insert(loc=0, column="episode", value=episode_value)
    df.columns = ["episode", "segment", "segment_start", "segment_end", "sentence"]

    nlp = spacy.load("en")
    nlp.add_pipe("sentencizer", "sentence_splitter", first=True)
    nlp.disable_pipes("tagger", "parser", "ner", "lemmatizer", "textcat")

    results = map(
        lambda doc: list(map(lambda span: span.doc.text, doc.sents)),
        nlp.pipe(df["sentence"]),
    )

    df = df.assign(sentence=pd.Series(results))

    df = df.explode(column="sentence", ignore_index=False).reset_index(drop=True)

    return df


def transcribe_translate_parts(
    audio: TBD,
    cut_segments_frames: list[tuple[int, int]],
    use_fp16: bool = True,
    transcription_model_size: str = "base",
) -> pd.DataFrame:
    transcription_model = whisper.load_model(transcription_model_size)

    transcriptions = []

    for index, (start, end) in enumerate(cut_segments_frames, start=1):
        print(f"transcribing segments: {index}/{len(cut_segments_frames)}")
        part_text = transcription_model.transcribe(
            audio[start:end], fp16=use_fp16, language="en"
        )
        transcriptions.append((index, start, end, part_text["text"]))

    data = pd.DataFrame(transcriptions)

    data = clean_transcript_df(data)

    return data
