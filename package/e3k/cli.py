import argparse
import datetime
import logging
from typing import Dict, List, Tuple

import episode_preprocessing_pipeline as epp
import mlflow
import model_evaluate as me
import model_output_information as moi
import model_predict as mp
import model_training as mt
import numpy as np
import pandas as pd
import preprocessing
import split_register_data as splitting
import typeguard
from matplotlib.figure import Figure
from tensorflow import config as tf_config
from transformers import TFRobertaForSequenceClassification

# setting up logger
logger = logging.getLogger("main")
logger.propagate = False
# attach handlers
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("logs.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


@typeguard.typechecked
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

    Author - Wojciech Stachowiak
    """
    parser = argparse.ArgumentParser(
        prog="app_name",
        description="""
        A command line tool for working with audio/video input, and training models.
        Functionality: process audio/video input into pd.DataFrame with sentences,
        train models to detect emotions, predict on new audio/video input.
        """,
    )

    # general args
    parser.add_argument(
        "task",
        type=str,
        choices=["preprocess", "train", "predict"],
        help="""
        string, task that has to be performed (preprocess/train/predict/add).
        preprocess: from video/audio to sentences,
        train: get new model from existing data,
        predict: get emotions from new episode
        """,
    )
    parser.add_argument(
        "--cloud",
        required=False,
        type=str,
        choices=["True", "False"],
        help="""
        string, used to run either cloud flow, or the local flow.
        """,
    )

    # episode_preprocessing args
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
        help="""
        string, file path to saved pipeline output (local only, default: output.csv)
        """,
    )
    parser.add_argument(
        "--save",
        required=False,
        type=bool,
        default=False,
        choices=[True, False],
        help="""
        bool, whether to save the processed video episode as a csv file.
        (local only, default: False)
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

    # data_loading args
    parser.add_argument(
        "--train_data",
        required=False,
        type=str,
        default="new_test_data/train_eval_sample.csv",
        help="Path to the training data [CSV file (local) | dataset (cloud)].",
    )
    parser.add_argument(
        "--val_data",
        required=False,
        type=str,
        default="new_test_data/train_eval_sample.csv",
        help="""Path to the validation data [CSV file (local) | not used,
        val dataset created automatically from train data (cloud)]""",
    )
    parser.add_argument(
        "--val_size",
        required=False,
        type=float,
        default=0.2,
        help="Proportion of train data used as validation data to train the model",
    )

    # model_training args
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=32,
        help="number of examples in one training batch",
    )
    parser.add_argument(
        "--max_length",
        required=False,
        type=int,
        default=128,
        help="max number of tokens used from one input sentence",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default=5,
        help="number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        required=False,
        type=float,
        default=1e-5,
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "--early_stopping_patience",
        required=False,
        type=int,
        default=3,
        help="patience parameter in the early stopping callback",
    )
    parser.add_argument(
        "--model_save_path",
        required=False,
        type=str,
        default="new_model",
        help="""
        path to the directory where the trained model will be saved (local only)
        """,
    )
    parser.add_argument(
        "--threshold",
        required=False,
        type=float,
        default=0.8,
        help="accuracy threshold for the model to be saved or register",
    )
    parser.add_argument(
        "--test_data", type=str, help="path to test data for model evaluation"
    )
    parser.add_argument(
        "--model_name",
        required=False,
        type=str,
        default=str(datetime.datetime.now().date()),
        help="name of the registered/saved model",
    )

    # model_predict args (that are not in model_train already)
    parser.add_argument(
        "--model_path", type=str, help="path to the folder with saved model"
    )
    parser.add_argument(
        "--label_decoder_path",
        type=str,
        help="path to the file with label_decoder data",
    )
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        required=False,
        default="roberta-base",
        help="tokenizer model used to preprocess data",
    )
    parser.add_argument(
        "--token_max_length",
        type=int,
        required=False,
        default=128,
        help="max number of tokens created by preprocessing a sentence",
    )

    args = parser.parse_args()

    return args


@typeguard.typechecked
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

    Output:
        data_df (pd.DataFrame): result of episode_preprocessing_pipeline,
        pd.DataFrame with sentences from the audio/video.

    Author - Wojciech Stachowiak
    """
    # get segment length in frames
    segment_frames_length = epp.segment_number_to_frames(
        1, sample_rate=args.target_sr, segment_seconds_length=args.segment_length
    )

    # load audio file and set sample rate to the chosen value
    audio = epp.load_audio(file_path=args.input_path, target_sample_rate=args.target_sr)

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
    cut_fragments_frames = epp.get_frame_segments_from_vad_output(
        speech_array=speech_array,
        sample_rate=args.target_sr,
        min_fragment_length_seconds=args.min_fragment_len,
        segment_seconds_length=args.segment_length,
        full_audio_length_frames=full_audio_length_frames,
    )

    # transcribe and translate fragments to get sentences in df
    use_fp16 = args.use_fp16 == "True"
    data_df = epp.transcribe_translate_fragments(
        audio=audio,
        cut_fragments_frames=cut_fragments_frames,
        sample_rate=args.target_sr,
        use_fp16=use_fp16,
        transcription_model_size=args.transcript_model_size,
    )

    if args.save:
        epp.save_data(data_df, args.output_path)

    return data_df


@typeguard.typechecked
def model_training(
    args: argparse.Namespace,
) -> Tuple[TFRobertaForSequenceClassification, Dict[int, str]]:
    """
    A function that follows the model_training module.
    It loads and pre-processes the data for model training, creates a new model,
    and fits the data into the model.

    Input:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line

    Output:
        model: a trained roBERTa transformer model
        label_decoder (dict[int, str]): a dictionary mapping numbers to
            text representation of emotions

    Author - Wojciech Stachowiak
    """
    # load data, and discard label decoder
    train_data, _ = splitting.load_data(args.train_data)

    # if passed, load validation data
    if args.val_data == "":
        train_data, val_data = splitting.get_train_val_data(train_data, args.val_size)
    else:
        val_data, _ = splitting.load_data(args.val_data)
    label_decoder = mt.get_label_decoder(train_data["emotion"])

    # get tf_datasets for model training
    train_dataset, val_dataset = preprocessing.preprocess_training_data(
        train_data,
        val_data,
        label_decoder,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # train newly created model
    model = mt.get_new_model(len(label_decoder))
    model = mt.train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
    )

    return model, label_decoder


@typeguard.typechecked
def evaluate_model(
    args: argparse.Namespace,
    model: TFRobertaForSequenceClassification,
    label_decoder: Dict[int, str],
) -> None:
    """
    A function that evaluates a trained model using a separate dataset.
    The function saves the model if it's accuracy surpasses the threshold.

    Inputs:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line
        model (TFRobertaForSequenceClassification): a trained roBERTa model
        label_decoder (dict[int, str]): python dictionary mapping numbers
        to text emotions

    Outputs: None

    Author - Wojciech Stachowiak
    """
    data = me.load_data(args.test_data)

    # getting token ids and attention masks
    tokens, masks = me.preprocess_prediction_data(data)
    # getting predictions and evaluating the model
    emotions, _ = me.predict(model, tokens, masks, label_decoder)
    accuracy, _ = me.evaluate(emotions, data)
    logger.info(f"Test accuracy: {accuracy * 100:.2f}%")

    # saving the model if the accuracy is high enough
    me.save_model(model, label_decoder, args.model_save_path, accuracy, args.threshold)


@typeguard.typechecked
def predict(args: argparse.Namespace, data: pd.DataFrame) -> Tuple[List[str], np.array]:
    """
    A function that returns model predictions given a model_path command line argument,
    and dataframe with column named "sentence"

    Input:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line
        data (pd.DataFrame): dataframe with sentences in a column

    Output:
        emotions (list[str]): text representation of predicted emotion for
            each sentence
        probabilities (list[float]): model's confidence for
            the most probable emotion in each sentence

    Author - Wojciech Stachowiak
    """
    model, label_decoder = mp.get_model(args.model_path)

    token_array, mask_array = preprocessing.preprocess_prediction_data(
        data, args.tokenizer_model, args.token_max_length
    )

    emotions, probabilities = mp.predict(model, token_array, mask_array, label_decoder)

    return emotions, probabilities


@typeguard.typechecked
def model_output_information(
    predicted_emotions: List[str], confidence_scores: np.array
) -> Figure:
    """
    A function that aggregates prediction results into a total confidence score,
    and a pie chart with predicted emotions distribution.

    Input:
        predicted_emotions (list[str]): text representation of predicted emotion
            for each sentence
        highest_probabilities (list[float]): model's confidence for the most
        probable emotion in each sentence

    Output: None

    Author - Wojciech Stachowiak
    """
    figure = moi.plot_emotion_distribution(predicted_emotions)
    moi.calculate_episode_confidence(confidence_scores)

    return figure


@typeguard.typechecked
def main() -> None:
    """
    A function that handles the correct execution of different modules given
    the command line arguments specified by the user. It calls the higher level
    functions named after modules depending on the specified task (see: --help)

    Input:
        None: the inputs are defined by the get_args function withing main,
            and handled through CLI arguments.

    Output:
        None: the inputs and outputs are defined in other functions and this only
            serves as a way of grouping them, and handling the common logic

    Author - Wojciech Stachowiak
    """

    # get arguments from argparser
    args = get_args()
    logger.info("got command line arguments")

    # Log parameters to MLflow
    mlflow.log_params(vars(args))

    logger.info("the pipeline will run locally")

    # checking for available GPU
    detected_gpu = len(tf_config.list_physical_devices("GPU")) > 0
    if not detected_gpu:
        logger.warning(
            (
                "No GPU detected, using CPU instead. "
                "Preprocessing from audio/video will take significantly longer, "
                "and training and predicting may not be a feasible task. "
                "Consider switching to a GPU device."
            )
        )
    else:
        logger.info("GPU detected")

    # handle episode preprocessing
    if args.task in ["preprocess", "predict"]:
        logger.info("entered task: preprocess")

        data_df = episode_preprocessing(args)

    # handle predicting
    if args.task == "predict":
        logger.info("entered task: predict")

        predicted_emotions, highest_probabilities = predict(args, data_df)
        figure = model_output_information(predicted_emotions, highest_probabilities)

        figure.show()

    # handle training
    if args.task == "train":
        logger.info("entered task: train")

        # Start MLflow run
        # mlflow.start_run()

        model, label_decoder = model_training(args)
        evaluate_model(args, model, label_decoder)

        # End MLflow run
        mlflow.end_run()


if __name__ == "__main__":
    main()
