import argparse
import logging
import os
import pickle
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import typeguard
from preprocessing import preprocess_prediction_data

# setting up logger
mt_logger = logging.getLogger(f"{'main.' if __name__ != '__main__' else ''}{__name__}")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("logs.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

mt_logger.addHandler(file_handler)

if len(mt_logger.handlers) == 0:
    mt_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    mt_logger.addHandler(stream_handler)


@typeguard.typechecked
def get_model(
    model_path: str,
) -> Tuple[transformers.TFRobertaForSequenceClassification, Dict[int, str]]:
    """
    Create or load a RoBERTa model with the specified number of output classes.
    Number of classes not needed when loading a previously trained model.

    Input:
        model_path (str): Path to the model directory.

    Output:
        model: RoBERTa model with the specified number of output classes.
        emotion_dict (dict[int, str]): Python dictionary that maps integers to text
            emotions for the model, only returned when loading a trained model,
            otherwise the value is None.

    Author:
        Max Meiners (214936)
    """

    # get config and model file paths
    config_path = os.path.join(model_path, "config.json")
    weights_path = os.path.join(model_path, "tf_model.h5")
    dict_path = os.path.join(model_path, "emotion_dict.joblib")

    mt_logger.info(f"Loading model configuration")

    # load an existing model
    config = transformers.RobertaConfig.from_pretrained(config_path)
    model = transformers.TFRobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=config
    )

    mt_logger.info("Loading model weights")
    model.load_weights(weights_path)

    mt_logger.info("Model loaded")

    # TODO change to pickle
    emotion_dict = joblib.load(dict_path)

    return model, emotion_dict


@typeguard.typechecked
def decode_labels(
    encoded_labels: List[int], emotion_decoder: Dict[int, str]
) -> List[str]:
    """
    A function that decodes label numbers into text representation of labels

    Input:
        encoded_labels (list[int]): List of labels represented as number.
        emotion_decoder (dict[int, str]): Dictionary with number to text mapping loaded
            with get_model function.

    Output:
        decoded_labels (list[str]): List of labels represented as text.

    Author:
        Max Meiners (214936)
    """

    decoded_labels = []
    for label in encoded_labels:
        if str(label) in emotion_decoder:
            decoded_labels.append(emotion_decoder[str(label)])
        else:
            mt_logger.warning(f"Class {label} not found in emotion_decoder.")
            decoded_labels.append("unknown")

    return decoded_labels


@typeguard.typechecked
def predict(
    model: transformers.TFRobertaForSequenceClassification,
    token_array: np.array,
    mask_array: np.array,
    emotion_decoder: Dict[int, str],
) -> Tuple[List[str], List[float]]:
    """
    A function that predicts emotions from preprocessed input using a loaded model.
    It returns text labels decoded using emotion_decoder dictionary loaded
    with the model.

    Input:
        model (transformers.TFRobertaForSequenceClassification): a loaded RoBERTa model.
        token_array (np.array): A token array returned by tokenize_text_data function.
        mask_array (np.array): A mask array returned by tokenize_text_data function.
        emotion_decoder (dict[int, str]): Dictionary with number to text mapping loaded
            with get_model function.

    Output:
        text_labels (list[str]): List of text emotions predicted by the model.
        highest_probabilities (list[float]): List of model's confidence
            that the predicted emotion is correct.

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("Predicting")

    input = {
        "input_ids": token_array,
        "attention_mask": mask_array,
    }

    preds = model(input)
    logits = preds.logits

    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    predicted_classes = np.argmax(probabilities, axis=1)
    highest_probabilities = np.max(probabilities, axis=1)

    text_labels = decode_labels(predicted_classes, emotion_decoder)

    mt_logger.info("Got predictions")

    return text_labels, highest_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        required=False,
        type=str,
        help="Path to the model configuration and weights file.",
    )

    parser.add_argument(
        "--data_path",
        required=False,
        type=str,
        help="Data to be predicted.",
    )

    parser.add_argument(
        "--tokenizer_model",
        required=False,
        type=str,
        default="roberta-base",
        help="Model to use for tokenization.",
    )

    parser.add_argument(
        "--max_length",
        required=False,
        type=int,
        default=128,
        help="Maximum length for tokenized sequences.",
    )

    parser.add_argument(
        "--decoder_path",
        required=False,
        type=str,
        help="Path to the joblib file containing the emotion decoder.",
    )

    parser.add_argument(
        "--output_path",
        required=False,
        type=str,
        help="Path to the output predictions.",
    )

    args = parser.parse_args()

    mt_logger.info("Arguments parsed")

    # Load the data
    data = pd.read_csv(args.data_path)
    mt_logger.info("Data loaded")

    # Call the preprocess_prediction_data function to get the preprocessed data
    tokens, masks = preprocess_prediction_data(
        data, args.tokenizer_model, args.max_length
    )
    mt_logger.info("Data preprocessed")

    # Load the model
    model = transformers.TFRobertaForSequenceClassification.from_pretrained(
        args.model_path
    )
    mt_logger.info("Model loaded")

    if "azureml" in args.decoder_path:
        # Folder URI + filename for Azure ML datastore
        decoder_path = f"{args.decoder_path}/label_decoder"
        # Load the emotion decoder
        with open(decoder_path, "rb") as f:
            emotion_decoder = pickle.load(f)

    else:
        # Load the emotion decoder
        with open(args.decoder_path, "rb") as f:
            emotion_decoder = pickle.load(f)

    mt_logger.info(f"Emotion decoder loaded with keys: {list(emotion_decoder.keys())}")

    # Make predictions
    text_labels, highest_probabilities = predict(model, tokens, masks, emotion_decoder)
    mt_logger.info("Predictions made")

    predictions = pd.DataFrame(
        {"text_labels": text_labels, "highest_probabilities": highest_probabilities}
    )

    predictions.to_csv(args.output_path, index=False)
