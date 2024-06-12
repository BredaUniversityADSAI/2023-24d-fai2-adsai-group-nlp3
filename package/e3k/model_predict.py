import argparse
import logging
import os

import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from preprocessing import preprocess_prediction_data

logging.basicConfig(level=logging.INFO)
mt_logger = logging.getLogger("main.model_predict")


def get_model(
    model_path: str,
) -> tuple[transformers.TFRobertaForSequenceClassification, dict[int, str]]:
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

    emotion_dict = joblib.load(dict_path)

    return model, emotion_dict


def decode_labels(
    encoded_labels: list[int], emotion_decoder: dict[int, str]
) -> list[str]:
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


def predict(
    model: transformers.TFRobertaForSequenceClassification,
    token_array: np.array,
    mask_array: np.array,
    emotion_decoder: dict[int, str],
) -> tuple[list[str], list[float]]:
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

    mt_logger.info(f"Probabilities: {probabilities}")
    mt_logger.info(f"Predicted classes: {predicted_classes}")
    mt_logger.info(f"Highest probabilities: {highest_probabilities}")
    
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

    # Load the emotion decoder
    with open(args.decoder_path, 'r') as f:  # Open file as text
        emotion_decoder = json.load(f)  # Use json.load instead of pickle.load
    mt_logger.info(f"Emotion decoder loaded with keys: {list(emotion_decoder.keys())}")

    # Make predictions
    text_labels, highest_probabilities = predict(model, tokens, masks, emotion_decoder)
    mt_logger.info("Predictions made")

    # Print the predictions
    for label, prob in zip(text_labels, highest_probabilities):
        print(f"Predicted emotion: {label} with confidence: {prob:.2f}")
    mt_logger.info("Results printed")
