import argparse
import logging

import episode_preprocessing_pipeline as epp
import model_output_information as moi
import model_training as mt
import pandas as pd
from tensorflow import config as tf_config
from transformers import TFRobertaForSequenceClassification

# setting up logger
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# checking for available GPU
detected_gpu = len(tf_config.list_physical_devices("GPU")) > 0
if not detected_gpu:
    logger.warning(
        """
        No GPU detected, using CPU instead.
        Preprocessing from audio/video will take significantly longer,
        and training and predicting may not be a feasible task.
        Consider switching to a GPU device.
        """
    )
else:
    logger.info("GPU detected")


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
        "--cloud",
        type=bool,
        choices=[True, False],
        help="""
        bool, used to run either cloud flow, or the local flow.
        """,
    )

    # episode_preprocessing args
    parser.add_argument(
        "task",
        type=str,
        choices=["preprocess", "train", "predict"],
        help="""
        string, task that has to be performed (preprocess/train/predict).
        preprocess: from video/audio to sentences,
        train: get new model from existing data,
        predict: get emotions from new episode
        """,
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
        "--train_data",
        required=False,
        type=str,
        help="Path to the training data CSV file.",
    )
    parser.add_argument(
        "--model_path",
        required=False,
        type=str,
        help="Path to the model configuration and weights file.",
    )
    parser.add_argument(
        "--val_size",
        required=False,
        type=float,
        help="size of the validation data for model training",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        help="number of examples in one training batch",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        help="number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        required=False,
        type=float,
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "--early_stopping_patience",
        required=False,
        type=int,
        help="patience parameter in the early stopping callback",
    )
    parser.add_argument(
        "--eval_data",
        required=False,
        type=str,
        help="Path to the evaluation data CSV file.",
    )
    parser.add_argument(
        "--model_save_path",
        required=False,
        type=str,
        help="path to the directory where the trained model will be saved",
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


def model_training(
    args: argparse.Namespace,
) -> tuple[
    TFRobertaForSequenceClassification, tuple[list[str], list[float], float, str]
]:
    """
    A function that follows parts of the model_training module that only train models.
    It loads and pre-processes the data for model training, creates a new model,
    and fits the data into the model.

    Input:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line

    Output:
        model: a trained roBERTa transformer model
        metrics (tuple[]): tuple with evaluation metrics
        (predicted_emotions, confidence_scores, total_accuracy, classification_report)

    Author - Wojciech Stachowiak
    """
    data, label_decoder_data = mt.load_data(args.train_data)

    tokenizer = mt.get_tokenizer()

    if args.model_path == "new":
        label_decoder = label_decoder_data
        model, _ = mt.get_model(args.model_path, num_classes=len(label_decoder))
    else:
        model, label_decoder = mt.get_model(args.model_path)

    train_set, val_set = mt.get_train_val_data(data, val_size=args.val_size)

    train_tokens, train_masks = mt.tokenize_text_data(train_set[0], tokenizer)
    val_tokens, val_masks = mt.tokenize_text_data(val_set[0], tokenizer)

    train_labels = mt.encode_labels(train_set[1], label_decoder)
    val_labels = mt.encode_labels(val_set[1], label_decoder)

    train_dataset = mt.create_tf_dataset(
        train_tokens, train_masks, train_labels, batch_size=args.batch_size
    )
    val_dataset = mt.create_tf_dataset(
        val_tokens, val_masks, val_labels, batch_size=args.batch_size
    )

    model = mt.train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
    )

    predicted_emotions, highest_probabilities, accuracy, report = mt.evaluate(
        model, tokenizer, label_decoder, eval_path=args.eval_data, max_length=128
    )

    metrics = (predicted_emotions, highest_probabilities, accuracy, report)

    mt.save_model(model, label_decoder, model_path=args.model_save_path)

    return model, metrics


def predict(
    args: argparse.Namespace, data: pd.DataFrame
) -> tuple[list[str], list[float]]:
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

    tokenizer = mt.get_tokenizer()

    model, label_decoder = mt.get_model(args.model_path, num_classes=0)

    tokens, masks = mt.tokenize_text_data(data["sentence"], tokenizer)

    emotions, probabilities = mt.predict(model, tokens, masks, label_decoder)

    return emotions, probabilities


def model_output_information(
    predicted_emotions: list[str], confidence_scores: list[float]
) -> None:
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
            serves as a way of grouping them, and handling the common logic

    Author - Wojciech Stachowiak
    """
    # get arguments from argparser
    args = get_args()
    logger.info("got command line arguments")

    cloud = args.cloud == "True"
    if cloud is True:
        logger.info("you are running in the cloud")
    else:
        logger.info("you are NOT running in the cloud")

    # handle episode preprocessing
    if args.task in ["preprocess", "predict"]:
        logger.info("entered task: preprocess")

        data_df = episode_preprocessing(args)

    # handle predicting
    if args.task == "predict":
        logger.info("entered task: predict")
        predicted_emotions, highest_probabilities = predict(args, data_df)
        model_output_information(predicted_emotions, highest_probabilities)

    # handle training
    if args.task == "train":
        logger.info("entered task: train")
        model_training(args)


if __name__ == "__main__":
    main()
