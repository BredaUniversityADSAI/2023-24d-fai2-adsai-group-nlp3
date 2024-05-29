import datetime
import io
import logging
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

epp_logger = logging.getLogger("main.episode_preprocessing_pipeline")

"""
Pipeline functions are main components of this pipeline.

They are ment to be used outside of this module,
and when used in order, provide the video or audio to sentences pipeline.
"""


def load_audio(file_path: str, target_sample_rate: int) -> np.array:
    """
    A function that loads audio data from video file
    or directly loads audio from input.
    Used by providing the path to the episode and desired sample rate.
    The function assumes the audio is multi-channel and
    automatically converts it to mono, but can also handle mono input.

    Input:
        file_path (str): file path to the video or audio
        target_sample_rate (int): the sample rate the audio file will be converted to

    Output:
        audio (np.array): mono audio file with specified sample rate
            represented as np.array
    """

    epp_logger.info("loading audio file")

    # load audio from the file
    audio = AudioSegment.from_file(file_path)

    # export the audio to in-memory object
    wav_file = io.BytesIO()
    audio.export(wav_file, format="wav")

    # load the audio and downsample it to target sample rate
    audio, _ = librosa.load(wav_file, sr=target_sample_rate)

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
        audio (np.array): audio file obtained from load_audio function
        sample_rate (int): sample rate of the audio file
        segment_seconds_length (float): segment length in seconds

    Output:
        segments (list[np.array]): list of cutouts from the audio file

    Author - Wojciech Stachowiak
    """

    epp_logger.info("splitting into segments")

    # get segment length in frame number
    segment_frames_length = int(segment_seconds_length * sample_rate)

    # get number of segments in the audio
    num_segments = len(audio) // segment_frames_length

    # split audio into list of segments
    segments = [
        audio[i * segment_frames_length : (i + 1) * segment_frames_length]
        for i in range(num_segments)
    ]

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
            detected speech for a given segment. True means that the segment
            contains speech

    Author - Wojciech Stachowiak
    """

    epp_logger.info("getting vad per segment")

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

    # Check if there are no speech segments
    if not any(segments_is_speech):
        epp_logger.warning("No speech segments found in the audio.")

    return segments_is_speech


def get_frame_segments_from_vad_output(
    speech_array: np.array,
    sample_rate: int,
    min_fragment_length_seconds: int,
    segment_seconds_length: float,
    full_audio_length_frames: int,
) -> list[tuple[int, int]]:
    """
    A function that connects the small segments (10/20/30ms) into larger (5-6 min)
    fragments based on the results from get_vad_per_segment function.
    It combines small segments until a specified threshold is reached,
    and the start and end in frames is saved. The function returns
    (start, end) pairs in a list.

    Input:
        speech_array (np.array): np.array with bool values obtained from
            get_vad_per_segment function
        sample_rate (int): sample rate of the audio file
        min_fragment_length_seconds (int): min amount of seconds per fragment
            generated from 10/20/30ms segments
        segment_seconds_length (float): length of one segment in seconds
        full_audio_length_frames (int): the total number of frames in
            the entire audio file

    Output:
        cut_fragments_frames (list[tuple[int, int]]): list of
            (start, end) frame number pairs

    Author - Wojciech Stachowiak
    """

    epp_logger.info("getting frame segments from vad output")

    # np.array with segment numbers for segment where we can cut the audio
    # this returns a tuple for each dimension (here it's 1, so we can just unpack it)
    # speech_array == 0 is a working substitute of speech_array is False
    cutable_segments = np.where(speech_array == 0)[0]

    # a target so we have an amount of frames to aim for
    fragment_min_length_frame = get_target_length_frames(
        min_fragment_length_seconds, sample_rate
    )

    # frame number from which the fragment will start
    fragment_start_frame = 0
    # start and end of each fragment in frames
    cut_fragments_frames = []

    for segment_number in cutable_segments:
        # get end of the segment in frames
        segment_end_frame = segment_number_to_frames(
            segment_number, sample_rate, segment_seconds_length
        )

        # if the total time (in frames) from start is long enough:
        if segment_end_frame - fragment_start_frame >= fragment_min_length_frame:
            # start a bit earlier and end a bit later if possible
            adjusted_fragment_start_frame = adjust_fragment_start_frame(
                fragment_start_frame, sample_rate
            )
            adjusted_segment_end_frame = adjust_fragment_end_frame(
                segment_end_frame, sample_rate, full_audio_length_frames
            )

            # append the (start, end) pair to a list
            cut_fragments_frames.append(
                (adjusted_fragment_start_frame, adjusted_segment_end_frame)
            )

            # update the start to be the end of the previous fragment
            fragment_start_frame = segment_end_frame

    # adjust the start frame for the last fragment
    adjusted_fragment_start_frame = adjust_fragment_start_frame(
        fragment_start_frame, sample_rate
    )

    # append last fragment's (start, end) pair to the list
    cut_fragments_frames.append(
        (adjusted_fragment_start_frame, full_audio_length_frames)
    )

    return cut_fragments_frames


def transcribe_translate_fragments(
    audio: np.array,
    cut_fragments_frames: list[tuple[int, int]],
    sample_rate: int,
    use_fp16: bool = True,
    transcription_model_size: str = "base",
    episode_value=datetime.datetime.now().date(),
) -> pd.DataFrame:
    """
    A function that transcribes and translates the audio fragments using openai-whisper
    model, and returns a pandas.DataFrame with English sentences
    (one sentence per row). The size of the model can be adjusted.

    Input:
        audio (np.array): full audio file loaded with load_audio function
        cut_audio_frames (list[tuple[int, int]]): list of
            (start, end) frame number pairs
        sample_rate (int): sample rate of the audio file
        use_fp16 (bool): Whether to use FP16 format for model prediction,
            needs to be False for CPU. Defaults to True.
        transcription_model_type: size of whisper model used for
            transcription and translation,
            see: https://pypi.org/project/openai-whisper/. default: "base"
        episode_value (Any): value assigned to each row for this episode,
            default: current date in YYYY-MM-DD format (e.g. 2024-05-16)

    Output:
        data (pd.DataFrame): dataframe with english sentences assigned to
            the episode value and segment number

    Author - Wojciech Stachowiak
    """

    epp_logger.info("transcribing and translating")

    # load the whisper model of choice
    transcription_model = whisper.load_model(transcription_model_size)

    transcriptions = []

    # transcribe and translate all episode fragments
    for index, (start, end) in enumerate(tqdm(cut_fragments_frames), start=1):
        # transcribe and translate
        fragment_text = transcription_model.transcribe(
            audio[start:end], fp16=use_fp16, language="en"
        )

        # add additional information to the transcript text
        transcriptions.append((index, start, end, fragment_text["text"]))

    # create a dataframe from the transcripts
    data = pd.DataFrame(transcriptions)

    # clean the dataframe
    data = clean_transcript_df(data, sample_rate, episode_value)

    return data


def save_data(
    df: pd.DataFrame,
    output_path: str = "output.csv",
) -> None:
    """
    A function that abstracts pd.DataFrame's saving funcitons with
    an option to chose json or scv format. If output path is not provided,
    the default path is "output.csv" in the current directory.

    Input:
        df (pd.DataFrame): dataframe to save
        output_format (str): file path to the saved file,
            default: "output.csv"

    Output: None

    Author - Wojciech Stachowiak
    """

    # Check if there is no data to save
    if df.empty:
        epp_logger.warning("No data to save.")

    epp_logger.info("saving to file")

    format = output_path.split(".")[1]

    if format == "json":
        df.to_json(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    epp_logger.info("done")


"""
Utils functions that are called from the above functions.
Utils functions are not ment to be used as a standalone,
and are an abstraction over some more complex parts of above pipeline functions.
"""


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

    Author - Wojciech Stachowiak
    """
    return int(sample_rate * segment_seconds_length * segment_number)


def get_target_length_frames(min_length_seconds: int, sample_rate: int) -> int:
    """
    A function that converts duration in seconds into number of frames
    representing this duration given the audio sample rate.

    Input:
        min_length_seconds (int): number of seconds
        sample_rate (int): sample rate of the audio file

    Output:
        target_length_frames (int): the number of frames that correspond
        to the number of seconds

    Author - Wojciech Stachowiak
    """
    return min_length_seconds * sample_rate


def adjust_fragment_start_frame(start_fragment_frame: int, sample_rate: int) -> int:
    """
    A function that moves the start of the larger fragment (a couple of minutes)
    to start 0.125 seconds earlier if the new start does not go below 0.

    Input:
        start_fragment_frame (int): the frame number that corresponds to start of
            the segment with no speech detected
        sample_rate (int): sample rate of the audio file

    Output:
        start_fragment_frame (int): adjusted (if possible) start_fragment_frame value

    Author - Wojciech Stachowiak
    """
    if start_fragment_frame - int(0.125 * sample_rate) >= 0:
        start_fragment_frame = start_fragment_frame - int(0.125 * sample_rate)

    return start_fragment_frame


def adjust_fragment_end_frame(
    end_fragment_frame: int, sample_rate: int, full_audio_length_frames: int
) -> int:
    """
    A function that moves the end of the larger fragment (a couple of minutes)
    to end 0.125 seconds later if the new end does not go over
    the full audio duration. Used by get_frame_segments_from_vad_output function.

    Input:
        end_fragment_frame (int): the frame number that corresponds to start of
            the last used in this fragment segment with no speech detected
        sample_rate (int): sample rate of the audio file

    Output:
        end_fragment_frame (int): adjusted (if possible) end_fragment_frame value

    Author - Wojciech Stachowiak
    """
    if end_fragment_frame + int(0.125 * sample_rate) <= full_audio_length_frames:
        end_fragment_frame = end_fragment_frame + int(0.125 * sample_rate)

    return end_fragment_frame


def clean_transcript_df(
    df: pd.DataFrame,
    sample_rate: int,
    episode_value=datetime.datetime.now().date(),
) -> pd.DataFrame:
    """
    A function that cleans the output of the transcribe_translate_fragments function.
    It adds a column to identify the episode later
    (current date in YYYY-MM-DD format by default). This can be changed by specifying
    the episode_value argument in the transcribe_translate_fragments function.
    Additionally, this function renames columns, and splits transcription for an episode
    fragment into separate sentences.

    Input:
        df (pd.DataFrame): a dataframe returned by
            transcribe_translate_fragments function
        sample_rate (int): sample rate of the audio file
        episode_value (Any): value assigned to each row for this episode,
            default: current date in YYYY-MM-DD format (e.g. 2024-05-16)

    Output:
        df (pd.DataFrame): a cleaned dataframe with one sentence in each row

    Author - Wojciech Stachowiak
    """

    # insert a column with episode_value for each row
    df.insert(loc=0, column="episode", value=episode_value)
    df.columns = [
        "episode",
        "segment",
        "segment_start_seconds",
        "segment_end_seconds",
        "sentence",
    ]

    nlp = spacy.load("en_core_web_md")

    # disable all pipeline components and add sentencizer to it
    nlp.add_pipe("sentencizer", "sentence_splitter", first=True)
    nlp.disable_pipes("tagger", "parser", "ner", "lemmatizer")

    # get list of sentences instead of the whole doc for each fragment
    results = []

    for doc in tqdm(nlp.pipe(df["sentence"])):
        sent_list = doc.sents
        sent_list = list(map(str, sent_list))
        results.append(sent_list)

    # overwrite the column with sentence lists
    df = df.assign(
        sentence=pd.Series(results),
        segment_start_seconds=(df["segment_start_seconds"] / sample_rate).round(2),
        segment_end_seconds=(df["segment_end_seconds"] / sample_rate).round(2),
    )

    # split the list into separate rows and reset the index
    df = (
        df.explode(column="sentence", ignore_index=False)
        .drop_duplicates(subset="sentence")
        .reset_index(drop=True)
    )

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="string, file path to the audio file",
    )
    parser.add_argument(
        "--output_path",
        required=False,
        type=str,
        default="output.csv",
        help="string, file path to saved pipeline output (default: output.csv)",
    )
    parser.add_argument(
        "--target_sr",
        required=False,
        type=int,
        default=32_000,
        help="""
        int, chosen audio file sample rate,
        pipeline handles 8000/16000/32000 (default: 32000)
        """,
    )
    parser.add_argument(
        "--segment_length",
        required=False,
        type=float,
        default=0.03,
        help="""
        float, length of a single segment for vad analysis in seconds,
        pipeline handles 0.01/0.02/0.03 (default: 0.03)
        """,
    )
    parser.add_argument(
        "--min_fragment_len",
        required=False,
        type=int,
        default=300,
        help="int, minimal length of the fragment in seconds (default: 300)",
    )
    parser.add_argument(
        "--vad_aggressiveness",
        required=False,
        type=int,
        default=0,
        help="""
        int, aggressiveness parameter of vad object,
        defines how strict speech isolation is on a scale from 0 to 3 (default: 0)
        """,
    )
    parser.add_argument(
        "--use_fp16",
        type=bool,
        required=False,
        default=True,
        help="""
        bool, whether to use FP16 format for model prediction,
        needs to be False for CPU (default: True)
        """,
    )
    parser.add_argument(
        "--transcript_model_size",
        required=False,
        type=str,
        default="large",
        help="""
        string, size of whisper model used for transcription and translation,
        see: https://pypi.org/project/openai-whisper/. default: large
        """,
    )

    args = parser.parse_args()

    # get segment length in frames
    segment_frames_length = segment_number_to_frames(
        1, sample_rate=args.target_sr, segment_seconds_length=args.segment_length
    )

    # load audio file and set sample rate to the chosen value
    audio = load_audio(file_path=args.input_path, target_sample_rate=args.target_sr)

    # get full audio length in frames
    full_audio_length_frames = len(audio)

    # get segments for vad analysis
    segments = get_segments_for_vad(
        audio=audio,
        sample_rate=args.target_sr,
        segment_seconds_length=args.segment_length,
    )

    # get vad output per segment
    speech_array = get_vad_per_segment(
        segments=segments,
        vad_aggressiveness=args.vad_aggressiveness,
        sample_rate=args.target_sr,
        segment_frames_length=segment_frames_length,
    )

    # get fragment sizes of chosen length
    cut_fragments_frames = get_frame_segments_from_vad_output(
        speech_array=speech_array,
        sample_rate=args.target_sr,
        min_fragment_length_seconds=args.min_fragment_len,
        segment_seconds_length=args.segment_length,
        full_audio_length_frames=full_audio_length_frames,
    )

    # transcribe and translate fragments to get sentences in df
    data_df = transcribe_translate_fragments(
        audio=audio,
        cut_fragments_frames=cut_fragments_frames,
        sample_rate=args.target_sr,
        use_fp16=args.use_fp16,
        transcription_model_size=args.transcript_model_size,
    )

    # save the data to chosen place with chosen format
    save_data(data_df, args.output_path)
