import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from transformers import (
    RobertaConfig, 
    TFRobertaForSequenceClassification, 
    RobertaTokenizer
    )

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    accuracy_score
    )

# Setting hyperparameters and constants
MAX_LENGTH = 128
BATCH_SIZE = 256
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(
        file_path: str, 
        dataset: str) -> tuple[pd.DataFrame, int]:
    """
    I figured this is a good way to return the number of classes,
    may be worth to go one step further and just return a dict with the classes
    May be also worth to have a class to keep this organized


    probably some way to chose train/evaluate data
    probably some way to filter training data 
        (no idea how we are saving this, so TBD tomorrow I guess)

    file_path (string): file path to the datasets
    dataset (string): type of dataset to load (train or eval)
    """

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    num_classes = 6

    print(f"Loaded {dataset} dataset successfully.")

    return (df, num_classes)


def get_model(
        config_path: str,
        weights_path: str, 
        num_classes: int):
    """
    Create and return a RoBERTa model with the specified number of output classes.

    Input:
        config_path (str): Path to the model configuration file.
        weights_path (str): Path to the model weights file.
        num_classes (int): Number of output classes for the model.

    Output:
        model: RoBERTa model with the specified number of output classes.
        tokenizer: Tokenizer used to tokenize the input data.
    """
    config = RobertaConfig.from_json_file(config_path)
    config.num_labels = num_classes
    model = TFRobertaForSequenceClassification(config)
    model.load_weights(weights_path)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Model loaded successfully.")
    
    return model, tokenizer


def preprocess_data(
        df: pd.DataFrame, 
        tokenizer):
    """
    A function to preprocess the input data.

    Input:
        df (pd.DataFrame): DataFrame containing the input data.
        tokenizer: Tokenizer used to tokenize the input data.

    Output:
        training_dataset: Training dataset for the model.
        validation_dataset: Validation dataset for the model.
        encoder.classes_: Encoded classes for the model.
        tokenizer: Tokenizer used to tokenize the input data.
    """

    text_data = df['sentence'].values
    emotional_labels = df['emotion'].values

    token_ids = []
    mask_values = []

    for text_piece in text_data:
        tokenized_result = tokenizer.encode_plus(
            text_piece,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='tf',
        )

        token_ids.append(tokenized_result['input_ids'])
        mask_values.append(tokenized_result['attention_mask'])

    token_ids = tf.concat(token_ids, axis=0)
    mask_values = tf.concat(mask_values, axis=0)

    encoder = LabelEncoder()
    transformed_labels = encoder.fit_transform(emotional_labels)
    transformed_labels = tf.convert_to_tensor(transformed_labels)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        token_ids, 
        transformed_labels, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=transformed_labels)
    training_masks, validation_masks = train_test_split(
        mask_values, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=transformed_labels)

    # Constructing TensorFlow datasets for training
    training_dataset = tf.data.Dataset.from_tensor_slices(({
        "input_ids": X_train_split, 
        "attention_mask": training_masks
        }, y_train_split))
    
    training_dataset = training_dataset.shuffle(len(X_train_split))
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Constructing TensorFlow datasets for validation
    validation_dataset = tf.data.Dataset.from_tensor_slices(({
        "input_ids": X_val_split, 
        "attention_mask": validation_masks
        }, y_val_split))
    
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print("Data pre-processed successfully.")

    return training_dataset, validation_dataset, encoder.classes_, tokenizer


def predict(
        model, sentences: 
        pd.DataFrame, 
        tokenizer, 
        label_encoder):
    """
    A function that takes a pre-trained model, a DataFrame of sentences, a tokenizer,
    and a label encoder to predict the emotion labels for the input sentences.
    It returns the predicted emotions and the highest probabilities for each prediction.

    Input:
        model: Model used for prediction.
        sentences (pd.DataFrame): DataFrame containing the input sentences.
        tokenizer: Tokenizer used to tokenize the input sentences.
        label_encoder: Label encoder used to encode the predicted classes.

    Output:
        predicted_emotions: List of predicted emotion labels for the input sentences.
        highest_probabilities: List of highest probabilities for each predicted emotion.
    """

    token_ids = []
    mask_values = []

    for text_piece in sentences['sentence']:
        tokenized_result = tokenizer.encode_plus(
            text_piece,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='tf',
        )

        token_ids.append(tokenized_result['input_ids'])
        mask_values.append(tokenized_result['attention_mask'])

    token_ids = tf.concat(token_ids, axis=0)
    mask_values = tf.concat(mask_values, axis=0)

    inputs = {'input_ids': token_ids, 
              'attention_mask': mask_values}
    outputs = model(inputs)
    logits = outputs.logits

    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    predicted_classes = np.argmax(probabilities, axis=1)
    highest_probabilities = np.max(probabilities, axis=1)

    predicted_emotions = label_encoder.inverse_transform(predicted_classes)

    print("Predictions made successfully.")

    return predicted_emotions, highest_probabilities


def evaluate(
        eval_data: pd.DataFrame, 
        model, tokenizer, 
        label_encoder):
    """
    Evaluate the model using the evaluation data.

    Input:
        eval_data (pd.DataFrame): DataFrame containing the evaluation data.
        model: The trained model to be evaluated.
        tokenizer: Tokenizer used to tokenize the input data.
        label_encoder: Label encoder used to decode the predicted classes.

    Output:
        predicted_emotions: List of predicted emotion labels.
        highest_probabilities: List of highest probabilities for each predicted emotion.
        accuracy: Accuracy score of the model.
        report: Classification report of the model's performance.
    """

    prepared_sentences = eval_data[["sentence"]]
    true_labels = eval_data["emotion"]

    predicted_emotions, highest_probabilities = predict(model, prepared_sentences, tokenizer, label_encoder)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_emotions)
    print(f'Accuracy: {accuracy}')

    # Generate classification report
    report = classification_report(true_labels, predicted_emotions)
    print(report)

    print("Evaluation complete.")

    return predicted_emotions, highest_probabilities, accuracy, report

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    print("Loading data...")

    parser.add_argument(
        "--config_path",
        required=True,
        type=str,
        help="Path to the model configuration file."
    )

    parser.add_argument(
        "--weights_path",
        required=True,
        type=str,
        help="Path to the model weights file."
    )

    print("Data loaded successfully.")

    args = parser.parse_args()