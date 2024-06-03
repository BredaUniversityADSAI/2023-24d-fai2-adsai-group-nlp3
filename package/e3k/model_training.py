import argparse
import collections
import collections.abc
import logging
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

mt_logger = logging.getLogger("main.model_training")


def load_data(file_path: str) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Load the dataset from a CSV file and return
    the DataFrame and a dictionary with labels.
    CSV needs to have "sentence" and "emotion" columns.

    Input:
        file_path (str): File path to the dataset CSV file.

    Output:
        df (pd.DataFrame): Loaded DataFrame containing the data.
        emotion_decoder (dict[int, str]): Dictionary with the class labels.

    Author:
        Max Meiners (214936)
    """

    df = pd.read_csv(file_path)[["sentence", "emotion"]].dropna()

    # Create a dictionary with class labels
    emotion_decoder = {i: label for i, label in enumerate(df["emotion"].unique())}

    mt_logger.info(f"loaded data: {os.path.basename(file_path)}")

    return df, emotion_decoder


def get_model(
    model_path: str, num_classes: int = 0
) -> tuple[transformers.TFRobertaForSequenceClassification, dict[int, str]]:
    """
    Create or load a RoBERTa model with the specified number of output classes.
    Number of classes not needed when loading a previously trained model.

    Input:
        model_path (str): Path to the model directory.
        num_classes (int): Number of output classes for the model. default: 0
            (handled outside this function using emotion_decoder
            from load_data function)

    Output:
        model: RoBERTa model with the specified number of output classes.
        emotion_dict (dict[int, str]): python dictionary that maps integers to text
            emotions for the model, only returned when loading a trained model,
            otherwise the value is None

    Author:
        Max Meiners (214936)
    """

    # return new instance of the model
    if model_path == "new":
        model_configuration = transformers.RobertaConfig.from_pretrained(
            "roberta-base", num_labels=num_classes
        )
        model = transformers.TFRobertaForSequenceClassification.from_pretrained(
            "roberta-base", config=model_configuration
        )

        return model, None

    # get config and model file paths
    config_path = os.path.join(model_path, "config.json")
    weights_path = os.path.join(model_path, "tf_model.h5")
    dict_path = os.path.join(model_path, "emotion_dict.joblib")

    mt_logger.info(f"loading model configuration")

    # load an existing model
    config = transformers.RobertaConfig.from_pretrained(config_path)
    model = transformers.TFRobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=config
    )

    mt_logger.info("loading model weights")
    model.load_weights(weights_path)

    mt_logger.info("model loaded")

    emotion_dict = joblib.load(dict_path)

    return model, emotion_dict


def get_tokenizer(type: str = "roberta-base") -> transformers.RobertaTokenizer:
    """
    A function that returns a pretrained tokenizer with a specified type.
    As a default, it's "roberta-base" tokenizer, but the type can be changed.

    Input:
        type (str): a tokenizer type from Hugging Faces

    Output:
        tokenizer (transformers.RobertaTokenizer): a tokenizer later used for
            preparing data for the model

    Author:
        Max Meiners (214936)
    """
    tokenizer = transformers.RobertaTokenizer.from_pretrained(type)
    mt_logger.info("tokenizer loaded")

    return tokenizer


def get_train_val_data(
    data_df: pd.DataFrame, val_size: float = 0.2
) -> tuple[tuple[pd.DataFrame], tuple[pd.DataFrame]]:
    """
    A function that splits the data into training and validation sets.
    It returns train set and val set with sentences and labels in both.

    Input:
        data_df (pd.DataFrame): a dataframe with columns named "sentence" and "emotion"
        val_size (float): a portion of data that will be assigned as validation set

    Output:
        train_set (tuple[pd.DataFrame, pd.DataFrame]): tuple with sentences and labels
            for the training set
        val_set (tuple[pd.DataFrame, pd.DataFrame]): tuple with sentences and labels
            for the validation set

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("splitting data into train and validation")

    X_train, X_val, y_train, y_val = train_test_split(
        data_df["sentence"],
        data_df["emotion"],
        test_size=val_size,
        random_state=42,
        stratify=data_df["emotion"],
    )

    train_set = (X_train, y_train)
    val_set = (X_val, y_val)

    mt_logger.info("data split")

    return train_set, val_set


def tokenize_text_data(
    text_data: collections.abc.Iterable,
    tokenizer: transformers.RobertaTokenizer,
    max_length: int = 128,
) -> tuple[np.array, np.array]:
    """
    Tokenizes the input text data using the provided
    tokenizer and returns the token IDs and mask values.

    Input:
        text_data: List of text data to be tokenized.
        tokenizer: Tokenizer used to tokenize the input data.
        max_length: Maximum length of the tokenized sequences.

    Output:
        token_ids_array: Numpy array of token IDs.
        mask_values_array: Numpy array of mask values.

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("tokenizing sentences")

    token_ids = []
    mask_values = []

    for text_piece in text_data:
        tokenized_result = tokenizer.encode_plus(
            text_piece,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="tf",
        )

        token_ids.append(tokenized_result["input_ids"])
        mask_values.append(tokenized_result["attention_mask"])

    token_ids = tf.concat(token_ids, axis=0)
    mask_values = tf.concat(mask_values, axis=0)

    token_ids_array = (
        token_ids.numpy() if isinstance(token_ids, tf.Tensor) else token_ids
    )
    mask_values_array = (
        mask_values.numpy() if isinstance(mask_values, tf.Tensor) else mask_values
    )

    mt_logger.info("tokenized")

    return token_ids_array, mask_values_array


def encode_labels(
    emotion_labels: list[str], label_decoder: dict[int, str]
) -> list[int]:
    """
    Encodes emotion labels using a LabelEncoder and returns the transformed labels.

    Input:
        emotion_labels (list[str]): List of emotional labels to be encoded.
        label_decoder (dict[int, str]):

    Output:
        transformed_labels: Encoded labels.

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("encoding labels")

    label_encoder = reverse_dict(label_decoder)

    encoded_labels = list(map(lambda x: label_encoder[x], emotion_labels))

    return encoded_labels


def create_tf_dataset(
    token_array: np.array,
    mask_array: np.array,
    encoded_labels: list[int],
    batch_size: int = 256,
) -> tf.data.Dataset:
    """
    Creates TensorFlow datasets for training and validation.

    Input:
        token_array (np.array): Numpy array of token IDs.
        mask_array (np.array): Numpy array of mask values.
        encoded_labels (list[int]): Encoded labels.
        batch_size (int): Batch size for the datasets.

    Output:
        dataset (tf.data.Dataset): TensorFlow dataset

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("creating tensorflow dataset")

    # Reformatting numpy arrays back to TensorFlow tensors for model input
    token_tensor = tf.convert_to_tensor(token_array)
    mask_tensor = tf.convert_to_tensor(mask_array)
    label_tensor = tf.convert_to_tensor(encoded_labels)

    # Constructing TensorFlow datasets for training and validation
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"input_ids": token_tensor, "attention_mask": mask_tensor}, label_tensor)
    )
    dataset = (
        dataset.shuffle(len(token_tensor))
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return dataset


def train_model(
    model: transformers.TFRobertaForSequenceClassification,
    training_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    epochs: int = 3,
    learning_rate: float = 1e-5,
    early_stopping_patience: int = 3,
) -> transformers.TFRobertaForSequenceClassification:
    """
    Train the model using tensorflow datasets.

    Input:
        model: Model to be trained.
        training_dataset: Dataset used for training the model
        validation_dataset: Dataset used for validating the model
        epochs (int): Number of epochs to train the model. default: 3
        learning_rate (float): optimizer's learning rate. default: 1e-5
        early_stopping_patience (int): patience parameter for EarlyStopping callback

    Output:
        model: trained model

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("compiling model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        verbose=1,
        restore_best_weights=True,
    )

    mt_logger.info("training model")

    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    mt_logger.info("training finished")

    return model


def save_model(
    model: transformers.TFRobertaForSequenceClassification,
    label_decoder: dict[int, str],
    model_path: str,
) -> None:
    """
    A function that saves trained model and it's emotion mapping to a file.

    Input:
        model (transformers.TFRobertaForSequenceClassification): trained model
        label_encoder (dict[int, str]): python dictionary mapping
            numbers to text emotions
        model_path (str): path to directory where the model will be saved

    Output: None

    Author:
        Max Meiners (214936)
    """

    dict_path = os.path.join(model_path, "emotion_dict.joblib")

    model.save_pretrained(model_path)
    joblib.dump(label_decoder, dict_path)

    mt_logger.info("model saved")


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
        model (transformers.TFRobertaForSequenceClassification): a loaded roBERTa model
        token_array (np.array): a token array returned by tokenize_text_data function
        mask_array (np.array): a mask array returned by tokenize_text_data function
        emotion_decoder (dict[int, str]): dictionary with number to text mapping loaded
            with get_model function

    Output:
        text_labels (list[str]): list of text emotions predicted by the model
        highest_probabilities (list[float]): list of model's confidence
            that the predicted emotion is correct

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("predicting")

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

    mt_logger.info("got predictions")

    return text_labels, highest_probabilities


def evaluate(
    model: transformers.TFRobertaForSequenceClassification,
    tokenizer: transformers.RobertaTokenizer,
    label_decoder: dict[int, str],
    eval_path: str,
    max_length: int,
) -> tuple[list[str], list[float], float, str]:
    """
    A function that evaluates trained model using a separate dataset.
    It returns predicted labels, and their probabilities, total_accuracy,
        and creates a report with different metrics.

    Input:
        model (transformers.TFRobertaForSequenceClassification): a loaded roBERTa model
        tokenizer (transformers.RobertaTokenizer): tokenizer compatible with the model
            architecture returned from the get_tokenizer function
        emotion_decoder (dict[int, str]): dictionary with number to text mapping loaded
            with get_model function
        eval_path (str): path do evaluation dataset CSV file

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("evaluating trained model")

    eval_data, _ = load_data(eval_path)
    token_array, mask_array = tokenize_text_data(
        eval_data["sentence"], tokenizer, max_length
    )

    pred_labels, highest_probabilities = predict(
        model, token_array, mask_array, label_decoder
    )
    true_labels = eval_data["sentence"].to_list()

    accuracy = accuracy_score(true_labels, pred_labels)
    mt_logger.info(f"model accuracy: {accuracy}")

    report = classification_report(true_labels, pred_labels)

    return pred_labels, highest_probabilities, accuracy, report


"""
Util functions (only used in other functions)
"""


def reverse_dict(dict: dict[int, str]) -> dict[str, int]:
    """
    A function that swaps keys and values of a dict so that
    the keys become values and the values become the keys

    Input:
        dict: a python dictionary

    Output:
        reverse_dict: a python dictionary with swapped keys and values

    Author:
        Max Meiners (214936)
    """

    reverse_dict = {}

    for key, value in dict.items():
        reverse_dict[value] = key

    return reverse_dict


def decode_labels(
    encoded_labels: list[int], emotion_decoder: dict[int, str]
) -> list[str]:
    """
    A function that decodes label numbers into text representation of labels

    Input:
        encoded_labels (list[int]): list of labels represented as number
        emotion_decoder (dict[int, str]): dictionary with number to text mapping loaded
            with get_model function

    Output:
        decoded_labels (list[str]): list of labels represented as text

    Author:
        Max Meiners (214936)
    """

    decoded_labels = list(map(lambda x: emotion_decoder[x], encoded_labels))

    return decoded_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "--model_path",
        required=False,
        type=str,
        help="Path to the model configuration and weights file.",
    )

    parser.add_argument(
        "--train_data",
        required=False,
        type=str,
        help="Path to the training data CSV file.",
    )

    parser.add_argument(
        "--eval_data",
        required=False,
        type=str,
        help="Path to the evaluation data CSV file.",
    )

    parser.add_argument(
        "--num_classes",
        required=False,
        type=int,
        help="num_classes",
    )

    parser.add_argument(
        "--val_size",
        required=False,
        type=float,
        help="val_size",
    )

    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        help="batch_size",
    )

    parser.add_argument(
        "--model_save_path",
        required=False,
        type=str,
        help="model_save_path",
    )

    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        help="epochs",
    )

    parser.add_argument(
        "--learning_rate",
        required=False,
        type=float,
        help="learning_rate",
    )

    parser.add_argument(
        "--early_stopping_patience",
        required=False,
        type=int,
        help="early_stopping_patience",
    )

    args = parser.parse_args()

    # training pipeline (commented)
    # to run it, from this script, uncomment the code below
    # and comment prediction pipeline code
    """
    data, label_decoder_data = load_data(args.input_path)

    tokenizer = get_tokenizer()

    if args.model_path == "new":
        label_decoder = label_decoder_data
        model, label_decoder_model = get_model(
            args.model_path,
            num_classes=len(label_decoder)
        )
    else:
        model, label_decoder = get_model(args.model_path, num_classes=args.num_classes)

    train_set, val_set = get_train_val_data(data, val_size=args.val_size)

    train_tokens, train_masks = tokenize_text_data(train_set[0], tokenizer)
    val_tokens, val_masks = tokenize_text_data(val_set[0], tokenizer)

    train_labels = encode_labels(train_set[1], label_decoder)
    val_labels = encode_labels(val_set[1], label_decoder)

    train_dataset = create_tf_dataset(train_tokens, train_masks,
        train_labels, batch_size=args.batch_size)
    val_dataset = create_tf_dataset(val_tokens, val_masks,
        val_labels, batch_size=args.batch_size)


    model = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience
    )

    predicted_emotions, highest_probabilities, accuracy, report = evaluate(
        model,
        tokenizer,
        label_decoder,
        eval_path=args.eval_data,
        max_length=128
    )

    save_model(model, label_decoder, model_path=args.model_save_path)
    """

    # prediction pipeline
    data, label_decoder_data = load_data(args.input_path)
    label_decoder_data = 0
    tokenizer = get_tokenizer()

    if args.model_path == "new":
        label_decoder = label_decoder_data
        model, _ = get_model(args.model_path, num_classes=len(label_decoder))
    else:
        model, label_decoder = get_model(args.model_path, num_classes=0)

    tokens, masks = tokenize_text_data(data["sentence"], tokenizer)

    emotions, probabilities = predict(model, tokens, masks, label_decoder)

    print(emotions[:3])
    print(probabilities[:3])
