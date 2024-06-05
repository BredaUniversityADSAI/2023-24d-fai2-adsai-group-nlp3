import argparse
import logging

import numpy as np
import tensorflow as tf
import transformers
from Components.preprocessing import preprocess_prediction_data

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
        required=False,
        type=str,
        help="Path to the model configuration and weights file.",
    )

    args = parser.parse_args()