"""
requirements:

episode_preprocessing_pipeline:
    - datetime
    - io
    - logging
    - wave
    - librosa
    - numpy
    - pandas
    - soundfile
    - spacy
    - webrtcvad
    - whisper
    - pydub
    - tqdm
    - argparse

new_episode_results (visual.py):
    - collections
    - matplotlib

"""


import argparse

import episode_preprocessing_pipeline as epp
import model_training as mt
import model_output_information as moi

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    accuracy_score
    )

"""
command line args:

task (mandatory, positional)
input_path
output_path
save (data from preprocessing)
target_sr
segment_length
min_fragment_len
vad_aggressiveness
use_fp16
transcript_model_size
"""


TEMP_EMOTIONS_LABELS = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise"
}



def get_args() -> argparse.Namespace:
    """
    A function that groups all positional and optional arguments from command line,
    and returns an argparse.Namespace object.
    A way to group all of user input in one abstraction.
    Also handles ArgumentParser.

    Input: None

    Output:
        args (argparse.Namespace): object holding all of arguments
            used later in other functions
    """
    parser = argparse.ArgumentParser(
        prog="app_name",
        description="""
        A command line tool for working with audio/video input, and training models.
        Functionality: process audio/video input into pd.DataFrame with sentences,
        train models to detect emotions, predict on new audio/video input.
        """,
    )


    # episode_preprocessing args
    parser.add_argument(
        "task",
        type=str,
        choices=["preprocess", "train", "predict", "add"],
        help="""
        string, task that has to be performed (preprocess/train/predict/add).
        preprocess: from video/audio to sentences,
        train: get new model from exisitng data,
        predict: get emotions from new episode,
        add: add prepared data to database
        """
    )
    parser.add_argument(
        "--input_path",
        required=False,
        type=str,
        default="",
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
        "--save",
        required=False,
        type=bool,
        default=False,
        choices=[True, False],
        help="""
        bool, whether to save the processed video episode as a csv file.
        default: False"
        """,
    )
    parser.add_argument(
        "--target_sr",
        required=False,
        type=int,
        default=32_000,
        choices=[32_000, 16_000, 8_000],
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
        choices=[0.03, 0.02, 0.01],
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
        choices=[0, 1, 2, 3],
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
        choices=[True, False],
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
        choices=["tiny", "base", "small", "medium", "large"],
        help="""
        string, size of whisper model used for transcription and translation,
        see: https://pypi.org/project/openai-whisper/. default: large
        """,
    )

    # model_training args
    parser.add_argument(
    "--model_path",
    required=False,
    type=str,
    help="Path to the model configuration and weights file."
    )

    parser.add_argument(
        "--train_data",
        required=False,
        type=str,
        help="Path to the training data CSV file."
    )

    parser.add_argument(
        "--eval_data",
        required=False,
        type=str,
        help="Path to the evaluation data CSV file."
    )

    args = parser.parse_args()

    return args


def episode_preprocessing(args: argparse.Namespace) -> pd.DataFrame:
    """
    Function that follows the episode_preprocessing_pipeline module.
    It uses module's functions to get the final output: pd.DataFrame with sentences.
    For more information on the functions used here, see the
    episode_preprocessing_pipeline module, and check the docstrings
    for separate functions.

    Input:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line
        
    Output (pd.DataFrame): result of episode_preprocessing_pipeline.
        pd.DataFrame with sentences from the audio/video.
    """
    # get segment length in frames
    segment_frames_length = epp.segment_number_to_frames(
        1, sample_rate=args.target_sr, segment_seconds_length=args.segment_length
    )

    # load audio file and set sample rate to the chosen value
    audio = epp.load_audio(
        file_path=args.input_path, target_sample_rate=args.target_sr
    )

    # get full audio length in frames
    full_audio_length_frames = len(audio)

    # get segments for vad analysis
    segments = epp.get_segments_for_vad(
        audio=audio,
        sample_rate=args.target_sr,
        segment_seconds_length=args.segment_length,
    )

    # get vad output per segment
    speech_array = epp.get_vad_per_segment(
        segments=segments,
        vad_aggressiveness=args.vad_aggressiveness,
        sample_rate=args.target_sr,
        segment_frames_length=segment_frames_length,
    )

    # get fragment sizes of chosen length
    cut_fragments_frames = (
        epp.get_frame_segments_from_vad_output(
            speech_array=speech_array,
            sample_rate=args.target_sr,
            min_fragment_length_seconds=args.min_fragment_len,
            segment_seconds_length=args.segment_length,
            full_audio_length_frames=full_audio_length_frames,
        )
    )

    # transcribe and translate fragments to get sentences in df
    data_df = epp.transcribe_translate_fragments(
        audio=audio,
        cut_fragments_frames=cut_fragments_frames,
        sample_rate=args.target_sr,
        use_fp16=args.use_fp16,
        transcription_model_size=args.transcript_model_size,
    )

    if args.save:
        epp.save_data(data_df, args.output_path)

    return data_df


def model_training(args):
    """
    A function that follows parts of the model_training module that only train models.
    It loads and pre-processes the data for model training, creates a new model,
    and fits the data into the model.

    Input:
        args ():

    Output:
        model: a trained roBERTa transformer model 
    """
    data, labels_dict = mt.load_data(args.input_path, "train")
    label_encoder = LabelEncoder()
    label_encoder.fit(list(labels_dict.values()))

    training_dataset, validation_dataset, num_classes, tokenizer = mt.preprocess_data(
        data,
        tokenizer
    )

    model, tokenizer = mt.get_model(args.model_path, num_classes)

    model = mt.train_model(model, training_dataset, validation_dataset)

    return model, tokenizer, label_encoder


def evaluate_model(args, model, tokenizer, label_encoder):
    print("entered evaluation")
    eval_data, _ = mt.load_data(args.eval_path, "eval")

    predicted_emotions, confidence_scores, total_accuracy, report = mt.evaluate(
        eval_data,
        model,
        tokenizer,
        label_encoder
    )

    return predicted_emotions, confidence_scores, total_accuracy, report


def predict(args, data_df):
    classes = TEMP_EMOTIONS_LABELS

    label_encoder = LabelEncoder()
    label_encoder.fit(list(classes.values()))

    model, tokenizer = mt.get_model(args.model_path, len(classes))
    predicted_emotions, highest_probabilities = mt.predict(
        model,
        data_df["sentence"].to_list(),
        tokenizer,
        label_encoder
    )

    return predicted_emotions, highest_probabilities


def model_output_information(predicted_emotions, confidence_scores):
    moi.plot_emotion_distribution(predicted_emotions)
    moi.calculate_episode_confidence(confidence_scores)


def main():
    """
    A function that handles the correct execution of different modules given
    the command line arguments specified by the user. It calls the higher level
    functions named after modules depending on the specified task (see: --help)

    Input:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line

    Output:
        None: the inputs and outputs are defined in other functions and this only
            serves as a way of groupping them, and handling the common logic
    """
    # get arguments from argparser
    args = get_args()
    print(args.task)

    # handle data adding
    if args.task == "add":
        pass

    # handle episode preprocessing
    if args.task in ["preprocess", "predict"]:
        data_df = episode_preprocessing(args)

    # handle predicting
    if args.task == "predict":
        predicted_emotions, highest_probabilities = predict(args, data_df)

        total_confidence = moi.calculate_episode_confidence(highest_probabilities)
        print(f"total confidence: {total_confidence}")
        moi.plot_emotion_distribution(predicted_emotions)
    
    # handle training
    if args.task == "train":
        model, tokenizer, label_encoder = model_training(args)
        predicted_emotions, confidence_scores, total_accuracy, report = evaluate_model(
            args,
            model,
            tokenizer,
            label_encoder
        )

        # a way to calm down pre-commits
        (confidence_scores, total_accuracy, report)


if __name__ == "__main__":
    main()
