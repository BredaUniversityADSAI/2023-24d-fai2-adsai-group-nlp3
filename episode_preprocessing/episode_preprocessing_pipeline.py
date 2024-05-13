import datetime

import pandas as pd
import pydub
import whisper


def load_audio_file(file_path) -> pydub.AudioSegment:
    """
    Function that loads audio from file path.

    Input:
        file_path: file path to the file containing audio

    Output:
        audio: pydub.AudioSegment object with audio
    """
    audio = pydub.AudioSegment.from_file(file_path)

    return audio


def get_segments(
    audio_segment: pydub.AudioSegment,
    min_silence_len: int = 1000,
    silence_threshold: int = -16,
    keep_silence=100,
) -> list[pydub.AudioSegment]:
    """
    Function that splits pydub.AudioSegment into list of pydub.AudioSegment
    objects based on detected silence.

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
        output_path: file path to save the data, default: None
            (saved in current directory as output.<chosen extension>)
    """
    if output_path is None:
        out_path = f"output.{format}"

    if format == "json":
        df.to_json(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    import argparse

    # instantiate argument parser
    parser = argparse.ArgumentParser(
        description="episode processing pipeline functionalities"
    )

    # add arguments
    parser.add_argument("episode_path", help="file path to episode file")
    parser.add_argument(
        "output_path",
        help="path to save output csv file ending in file name .csv",
        default="output.csv",
    )
    parser.add_argument(
        "min_silence_len",
        help="min time of silence (in ms) to split the audio. default: 1000",
        default=1000,
    )
    parser.add_argument(
        "silence_thresh",
        help="threshold in dB to treat the audio as silence. default: -16",
        default=-16,
    )
    parser.add_argument(
        "keep_silence",
        help="""
        (in ms or True/False) margin of silence to keep (if True, all silence is kept,
        if False, none is). default: 100",
        default=100,
        """,
    )
    parser.add_argument(
        "transcription_model_type",
        help="""
        size of whisper model used for transcription and translation
        (see: https://pypi.org/project/openai-whisper/), default: base
        """,
        default="base",
    )
    parser.add_argument(
        "fp16",
        help="""
        (bool): Whether to use FP16 format for model prediction,
        needs to be False for CPU. Defaults to True.",
        default=True
        """,
    )
    parser.add_argument(
        "episode_value",
        help="""
        a value assigned to the whole episode in the pandas.DataFrame,
        default: current date (YYYY-MM-DD)",
        default=datetime.datetime.now().date()
        """,
    )
    parser.add_argument(
        "output_format",
        help="""
        format in which the data will be saved, only "csv" or "json" are valid formats,
        default: csv'
        """,
        default="csv",
    )

    # get args values
    args = parser.parse_args()

    # load audio and get segments
    audio_file = load_audio_file(args["episode_path"])
    segments = get_segments(
        audio_file,
        min_silence_len=args["min_silence_len"],
        silence_threshold=args["silence_thresh"],
        keep_silence=args["keep_silence"],
    )

    del audio_file

    # get transcription + translations
    transcript_df = transcribe_translate_segments(
        segments,
        transcription_model_type=args["transcription_model_type"],
        fp16=args["fp16"],
        episode_value=args["episode_value"],
    )

    del segments

    # split segments into sentences and save to csv
    transcript_df = split_into_sentences(transcript_df)
    export_csv(transcript_df, args["output_format"], output_path=args["output_path"])
