import argparse
import logging
import joblib

import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
from Components.preprocessing import preprocess_prediction_data

logging.basicConfig(level=logging.INFO)
mt_logger = logging.getLogger("main.model_predict")


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

    decoded_labels = list(map(lambda x: emotion_decoder[x], encoded_labels))

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

    text_labels = decode_labels(predicted_classes, emotion_decoder)

    mt_logger.info("Got predictions")

    return text_labels, highest_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the model configuration and weights file.",
    )

    parser.add_argument(
        "--data_path",
        required=True,
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
        required=True,
        type=str,
        help="Path to the joblib file containing the emotion decoder.",
    )

    args = parser.parse_args()

    mt_logger.info("Arguments parsed")

    # Load the data
    data = pd.read_csv(args.data_path)
    mt_logger.info("Data loaded")

    # Call the preprocess_prediction_data function to get the preprocessed data
    tokens, masks = preprocess_prediction_data(data, args.tokenizer_model, args.max_length)
    mt_logger.info("Data preprocessed")

    # Load the model
    model = transformers.TFRobertaForSequenceClassification.from_pretrained(args.model_path)
    mt_logger.info("Model loaded")

    # Load the emotion decoder
    emotion_decoder = joblib.load(args.decoder_path)
    mt_logger.info("Emotion decoder loaded")

    # Make predictions
    text_labels, highest_probabilities = predict(model, tokens, masks, emotion_decoder)
    mt_logger.info("Predictions made")

    # Print the predictions
    for label, prob in zip(text_labels, highest_probabilities):
        print(f"Predicted emotion: {label} with confidence: {prob:.2f}")
    mt_logger.info("Results printed")