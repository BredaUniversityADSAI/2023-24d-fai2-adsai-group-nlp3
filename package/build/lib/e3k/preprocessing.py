import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import typeguard

# setting up logger
pre_logger = logging.getLogger(f"{'main.' if __name__ != '__main__' else ''}{__name__}")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    pre_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    pre_logger.addHandler(stream_handler)

file_handler = logging.FileHandler("logs.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

pre_logger.addHandler(file_handler)


# TODO author
@typeguard.typechecked
def get_tokenizer(
    model_name: str = "roberta-base",
) -> transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast:
    """
    Get the tokenizer for a specified model.

    Input:
        model_name (str): Name of the model for the tokenizer.
        Default is 'roberta-base'.

    Output:
        tokenizer: Tokenizer for the specified model.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return tokenizer


# TODO author
@typeguard.typechecked
def tokenize_text_data(
    data: pd.Series,
    tokenizer: transformers.models.roberta.tokenization_roberta_fast.
    RobertaTokenizerFast,
    max_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tokenize text data using the provided tokenizer.

    Input:
        data (pd.Series): Text data to be tokenized.
        tokenizer (transformers.AutoTokenizer): Tokenizer to use.
        max_length (int): Maximum length for the tokenized sequences. Default is 128.

    Output:
        input_ids (np.ndarray): Tokenized input IDs.
        attention_masks (np.ndarray): Attention masks for the tokenized sequences.
    """
    encoding = tokenizer(
        data.tolist(),
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    input_ids = encoding["input_ids"]
    attention_masks = encoding["attention_mask"]
    return input_ids, attention_masks


# TODO author
@typeguard.typechecked
def encode_labels(labels: pd.Series, label_decoder: Dict[int, str]) -> np.ndarray:
    """
    Encode labels using the label decoder.

    Input:
        labels (pd.Series): Labels to be encoded.
        label_decoder (dict[int, str]): Dictionary to map labels to integers.

    Output:
        encoded_labels (np.ndarray): Encoded labels as integers.
    """
    label_encoder = {label: i for i, label in label_decoder.items()}
    encoded_labels = labels.map(label_encoder).values
    return encoded_labels


# TODO author
@typeguard.typechecked
def create_tf_dataset(
    input_ids: np.ndarray,
    attention_masks: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from input IDs, attention masks, and labels.

    Input:
        input_ids (np.ndarray): Tokenized input IDs.
        attention_masks (np.ndarray): Attention masks for the tokenized sequences.
        labels (np.ndarray): Encoded labels.
        batch_size (int): Batch size for the dataset. Default is 32.

    Output:
        dataset (tf.data.Dataset): TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"input_ids": input_ids, "attention_mask": attention_masks}, labels)
    )
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
    return dataset


# TODO author + update type annotations
@typeguard.typechecked
def preprocess_training_data(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    label_decoder: Dict[int, str],
    tokenizer_model: str = "roberta-base",
    max_length: int = 128,
    batch_size: int = 32,
):
    """
    Preprocess training and validation data.

    Input:
        train_data (pd.DataFrame): Training data with 'sentence' and 'emotion' columns.
        val_data (pd.DataFrame): Validation data with 'sentence' and 'emotion' columns.
        label_decoder (dict[int, str]): Dictionary to map labels to integers.
        tokenizer_model (str): Model name for the tokenizer. Default is 'roberta-base'.
        max_length (int): Maximum length for tokenized sequences. Default is 128.
        batch_size (int): Batch size for the dataset. Default is 32.

    Output:
        train_dataset (tf.data.Dataset): Preprocessed training dataset.
        val_dataset (tf.data.Dataset): Preprocessed validation dataset.
    """
    tokenizer = get_tokenizer(tokenizer_model)

    # Tokenize and encode training data
    train_tokens, train_masks = tokenize_text_data(
        train_data["sentence"], tokenizer, max_length
    )
    train_labels = encode_labels(train_data["emotion"], label_decoder)
    train_dataset = create_tf_dataset(
        train_tokens, train_masks, train_labels, batch_size
    )

    # Tokenize and encode validation data
    val_tokens, val_masks = tokenize_text_data(
        val_data["sentence"], tokenizer, max_length
    )
    val_labels = encode_labels(val_data["emotion"], label_decoder)
    val_dataset = create_tf_dataset(val_tokens, val_masks, val_labels, batch_size)

    return train_dataset, val_dataset


# TODO author + update type annotations
@typeguard.typechecked
def preprocess_prediction_data(
    data: pd.DataFrame, tokenizer_model: str = "roberta-base", max_length: int = 128
):
    """
    Preprocess data for prediction.

    Input:
        data (pd.DataFrame): Data with 'sentence' column.
        tokenizer_model (str): Model name for the tokenizer. Default is 'roberta-base'.
        max_length (int): Maximum length for tokenized sequences. Default is 128.

    Output:
        input_ids (np.ndarray): Tokenized input IDs.
        attention_masks (np.ndarray): Attention masks for the tokenized sequences.
    """
    tokenizer = get_tokenizer(tokenizer_model)
    tokens, masks = tokenize_text_data(data["sentence"], tokenizer, max_length)
    return tokens, masks


# TODO author + update type annotations
@typeguard.typechecked
def preprocess_prediction_data_no_tokenizer(
    data: pd.DataFrame, tokenizer, max_length: int = 128
):
    """
    Preprocess data for prediction.

    Input:
        data (pd.DataFrame): Data with 'sentence' column.
        tokenizer_model (str): Model name for the tokenizer. Default is 'roberta-base'.
        max_length (int): Maximum length for tokenized sequences. Default is 128.

    Output:
        input_ids (np.ndarray): Tokenized input IDs.
        attention_masks (np.ndarray): Attention masks for the tokenized sequences.
    """
    tokens, masks = tokenize_text_data(data["sentence"], tokenizer, max_length)
    return tokens, masks
