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


def load_audio_from_video(file_path: str, target_sample_rate: int) -> np.array:
    """
    A function that loads audio data from video file.
    Used by providing the path to the episode and desired sample rate.
    The function assumes the audio is multi-channel and
    automatically converts it to mono, but can also handle mono input.

    Input:
        file_path (str): file path to the video
        target_sample_rate (int): the sample rate the audio file will be converted to

    Output:
        audio (np.array): mono audio file with specified sample rate represented as np.array
    """
    # load audio from video file
    audio = AudioSegment.from_file(file_path)

    # export the audio to in-memory object
    wav_file = io.BytesIO()
    audio.export(wav_file, format="wav")

    # load the audio and downsample it to target sample rate
    audio, sample_rate = librosa.load(wav_file, sr=target_sample_rate)

    # convert the audio to mono
    audio = librosa.to_mono(audio)

    return audio


def get_segments_for_vad(
    audio: np.array, sample_rate: int, segment_seconds_length: float
) -> list[np.array]:
    """
    A function that adapts the audio file so that it is compatible with
    webrtcvad.Vad object. The sample rate must be 8kHz, 16kHz, or 32kHz, and
    the segment length must be 10ms, 20ms, or 30ms. It returns a list of
    audio segments of the chosen length.

    Input:
        audio (np.array): audio file obtained from load_audio_from_video function
        sample_rate (int): sample rate of the audio file
        segment_seconds_length (float): segment length in seconds

    Output:
        segments (list[np.array]): list of cutouts from the audio file
    """

    # get segment length in frame number
    segment_frames_length = int(segment_seconds_length * sample_rate)

    # get number of segments in the audio
    num_segments = len(audio) // segment_frames_length
    print(f"segment length: {segment_frames_length} frames")

    # split audio into list of segments
    segments = [
        audio[i * segment_frames_length : (i + 1) * segment_frames_length]
        for i in range(num_segments)
    ]
    print(f"split into {len(segments)} segments")

    return segments


def get_vad_per_segment(
    segments: list[np.array],
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
        segments (list[np.array]): list with cut audio obtained from
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


    segments_is_speech = []
    for segment in tqdm(segments):
        # prepare an in-memory location for segments
        segment_file = io.BytesIO()
        # write the segments to in-memory location
        sf.write(segment_file, segment, sample_rate, format="wav", subtype="PCM_16")

        # prepare the BytesIO object for opening with wave library
        segment_file.seek(0)
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
            get_vad_per_segment functionfull_audio_length_frames
        sample_rate (int):
        segment_seconds_length (float):
        full_audio_length_frames (int): the total number of frames in
            the entire audio file

    Output:
        cut_segments_frames (list[tuple[int, int]]): list of
            (start, end) frame number pairs
    """

    # np.array with segment numbers for segment where we can cut the audio
    # this returns a tuple for each dimention (here it's 1, so we can just unpack it)
    # speech_array == 0 is a working substitute of speech_array is False
    # (this returns False as 1 value)
    cutable_segments = np.where(speech_array == 0)[0]

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

            # append the (start, end) pair to a list
            cut_segments_frames.append(
                (adjusted_segment_start_frame, adjusted_segment_end_frame)
            )

            # update the start to be the end of the previous part
            segment_start_frame = segment_end_frame

    # adjust the start frame for the last part
    adjusted_segment_start_frame = adjust_part_start_frame(segment_start_frame, sample_rate)

    # append last part's (start, end) pair to the list
    cut_segments_frames.append((adjusted_segment_start_frame, full_audio_length_frames))

    return cut_segments_frames


def clean_transcript_df(
    df: pd.DataFrame,
    episode_value=datetime.datetime.now().date()
) -> pd.DataFrame:
    """
    A function that cleans the output of the transcribe_translate_parts function.
    It adds a column to identify the episode later
    (current date in YYYY-MM-DD format by deafut). This can be changed by specifying
    the episode_value argument in the transcribe_translate_parts function.
    Additionaly, this function renames columns, and splits transcription for an episode
    part into separate sentences.

    Input:
        df (pd.DataFrame): a dataframe returned by transcribe_translate_parts function
        episode_value (Any): value assigned to each row for this episode,
            default: current date in YYYY-MM-DD format (e.g. 2024-05-16)
    
    Output:
        df (pd.DataFrame): a cleaned dataframe with one sentence in each row
    """

    # insert a column with episode_value for each row
    df.insert(loc=0, column="episode", value=episode_value)
    df.columns = ["episode", "segment", "segment_start", "segment_end", "sentence"]

    nlp = spacy.load("en_core_web_md")

    # disable all pipeline components and add sentencizer to it
    nlp.add_pipe("sentencizer", "sentence_splitter", first=True)
    nlp.disable_pipes("tagger", "parser", "ner", "lemmatizer")

    # get list of sentences instead of the whole doc for each part
    results = []

    for doc in tqdm(nlp.pipe(df["sentence"])):
        sent_list = doc.sents
        sent_list = list(map(str, sent_list))
        results.append(sent_list)

    # overwrite the column with sentence lists
    df = df.assign(sentence=pd.Series(results))

    # split the list into separate rows and reset the index
    df = df.explode(column="sentence", ignore_index=False)\
        .reset_index(drop=True)\
        .drop_duplicates(subset="sentence")

    return df


def transcribe_translate_parts(
    audio: np.array,
    cut_segments_frames: list[tuple[int, int]],
    use_fp16: bool = True,
    transcription_model_size: str = "base",
    episode_value=datetime.datetime.now().date()
) -> pd.DataFrame:
    """
    A function that transcribes and translates the audio parts using openai-whisper
    model, and returns a pandas.DataFrame with English sentences per part.
    The size of the model can be adjusted.

    Input:
        audio (np.array): full audio file loaded with load_audio_from_video function
        cut_audio_frames (list[tuple[int, int]]): list of
            (start, end) frame number pairs
        use_fp16 (bool): Whether to use FP16 format for model prediction,
            needs to be False for CPU. Defaults to True.
        transcription_model_type: size of whisper model used for
            transcription and translation,
            see: https://pypi.org/project/openai-whisper/. default: "base"
        episode_value (Any): value assigned to each row for this episode,
            default: current date in YYYY-MM-DD format (e.g. 2024-05-16)
    """

    # load the whisper model of choice
    transcription_model = whisper.load_model(transcription_model_size)

    transcriptions = []

    # trancribe and translate all episode parts
    for index, (start, end) in enumerate(tqdm(cut_segments_frames), start=1):

        # transcribe and translate
        part_text = transcription_model.transcribe(
            audio[start:end], fp16=use_fp16, language="en"
        )

        # add additional information to the transcript text
        transcriptions.append((index, start, end, part_text["text"]))

    # create a dataframe from the transcripts
    data = pd.DataFrame(transcriptions)

    # clean the dataframe
    data = clean_transcript_df(data, episode_value)

    return data


if __name__ == "__main__":
    
    target_sample_rate = 32_000
    segment_seconds_length = 0.03
    segment_frames_length = segment_number_to_frames(
        1,
        sample_rate=target_sample_rate,
        segment_seconds_length=segment_seconds_length
    )
    
    audio = load_audio_from_video(
        "ep_1.mov",
        target_sample_rate=target_sample_rate
    )
    
    full_audio_length_frames = len(audio)

    print("get_segments_for_vad")
    segments = get_segments_for_vad(
        audio,
        target_sample_rate,
        segment_seconds_length=segment_seconds_length
    )

    print("get_vad_per_segment")
    speech_array = get_vad_per_segment(
        segments,
        vad_aggressiveness=0,
        sample_rate=target_sample_rate,
        segment_frames_length=segment_frames_length
    )

    print("get_frame_segments_from_vad_output")
    cut_segments_frames = get_frame_segments_from_vad_output(
        speech_array,
        sample_rate=target_sample_rate,
        segment_seconds_length=segment_seconds_length,
        full_audio_length_frames=full_audio_length_frames
    )
    
    print("transcribe_translate_parts")
    data_df = transcribe_translate_parts(
        audio,
        cut_segments_frames=cut_segments_frames,
        use_fp16=True,
        transcription_model_size="large"
    )
    
    data_df.to_csv("test_output.csv")
    