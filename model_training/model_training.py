import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import os

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

    Author: 
        Max Meiners (214936)
    """

    df = pd.read_csv(file_path)
    num_classes = 6

    return (df, num_classes)


def get_model(
        model_path: str,
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

    Author: 
        Max Meiners (214936)
    """

    config_path = os.path.join(model_path, 'config.json')
    weights_path = os.path.join(model_path, 'tf_model.h5')

    print(f"Loading model configuration from {config_path}...")

    config = RobertaConfig.from_pretrained(config_path)
    config.num_labels = num_classes
    model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

    print(f"Loading model weights from {weights_path}...")
    model.load_weights(weights_path)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    print("Model and tokenizer loaded.")
    
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

    Author: 
        Max Meiners (214936)
    """

    if isinstance(df, str):
        df = pd.read_csv(df)

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

    token_ids_array = token_ids.numpy() if isinstance(token_ids, tf.Tensor) else token_ids
    mask_values_array = mask_values.numpy() if isinstance(mask_values, tf.Tensor) else mask_values

    encoder = LabelEncoder()
    transformed_labels = encoder.fit_transform(emotional_labels)
    transformed_labels_array = transformed_labels

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        token_ids_array, 
        transformed_labels_array, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=transformed_labels_array
    )
    training_masks, validation_masks = train_test_split(
        mask_values_array, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=transformed_labels_array
    )

    # Reformatting numpy arrays back to TensorFlow tensors for model input
    X_train_tensors = tf.convert_to_tensor(X_train_split)
    X_val_tensors = tf.convert_to_tensor(X_val_split)
    y_train_tensors = tf.convert_to_tensor(y_train_split)
    y_val_tensors = tf.convert_to_tensor(y_val_split)
    training_masks_tensors = tf.convert_to_tensor(training_masks)
    validation_masks_tensors = tf.convert_to_tensor(validation_masks)


    # Constructing TensorFlow datasets for training and validation
    training_dataset = tf.data.Dataset.from_tensor_slices(
        ({"input_ids": X_train_tensors, "attention_mask": training_masks_tensors}, y_train_tensors)
    )
    training_dataset = training_dataset.shuffle(len(X_train_tensors)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        ({"input_ids": X_val_tensors, "attention_mask": validation_masks_tensors}, y_val_tensors)
    )
    validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    print("Data preprocessing complete.")
    return training_dataset, validation_dataset, encoder.classes_, tokenizer


def predict(
        model, 
        sentences: list, 
        tokenizer, 
        label_encoder,
        batch_size=BATCH_SIZE):
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

    Author: 
        Max Meiners (214936)
    """

    token_ids = []
    mask_values = []

    for text_piece in sentences:
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

    inputs = {'input_ids': token_ids, 'attention_mask': mask_values}
    num_samples = token_ids.shape[0]

    all_predicted_classes = []
    all_highest_probabilities = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_inputs = {
            'input_ids': inputs['input_ids'][start_idx:end_idx],
            'attention_mask': inputs['attention_mask'][start_idx:end_idx]
        }
        outputs = model(batch_inputs)
        logits = outputs.logits

        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        predicted_classes = np.argmax(probabilities, axis=1)
        highest_probabilities = np.max(probabilities, axis=1)

        all_predicted_classes.extend(predicted_classes)
        all_highest_probabilities.extend(highest_probabilities)

    predicted_emotions = label_encoder.inverse_transform(all_predicted_classes)

    return predicted_emotions, all_highest_probabilities


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

    Author: 
        Max Meiners (214936)
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

    return predicted_emotions, highest_probabilities, accuracy, report

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    print("Loading data(1)...")

    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the model configuration and weights file."
    )

    parser.add_argument(
        "--train_data",
        required=True,
        type=str,
        help="Path to the training data CSV file."
    )

    parser.add_argument(
        "--eval_data",
        required=True,
        type=str,
        help="Path to the evaluation data CSV file."
    )

    print("Data loaded successfully.")

    args = parser.parse_args()

    print("Loading data(2)...")
    train_data = pd.read_csv(args.train_data)
    print("Data loaded.")

    print("Preparing model...")
    num_classes = len(train_data['emotion'].unique())
    model, tokenizer = get_model(args.model_path, num_classes)
    print("Model prepared.")

    print("Preprocessing training data...")
    training_dataset, validation_dataset, class_names, tokenizer = preprocess_data(train_data, tokenizer)
    print("Training data preprocessed.")

    print("Loading evaluation data...")
    eval_data = pd.read_csv(args.eval_data)
    print("Evaluation data loaded.")

    print("Starting evaluation...")
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    evaluate(eval_data, model, tokenizer, label_encoder)
    print("Evaluation complete.")