import logging
import config
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

# setting up logger
pre_lstm_logger = logging.getLogger(
    f"{'main.' if __name__ != '__main__' else ''}{__name__}"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    pre_lstm_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    pre_lstm_logger.addHandler(stream_handler)

file_handler = logging.FileHandler("logs.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

pre_lstm_logger.addHandler(file_handler)

# Azure ML Workspace and Model details
MODEL_NAME = "LSTM_model"
SAVE_PATH = ("/Users/maxmeiners/Library/CloudStorage/OneDrive-BUas"
             "/Github/Year 2/Block D/model/updated_lstm_model.h5")

# Function to load Azure ML workspace
def load_workspace():
    """
    Load Azure ML workspace using credentials from the config file.

    Returns:
        workspace (MLClient): Azure ML workspace object.

    Author:
        Max Meiners (214936)
    """
    print("Loading Azure ML workspace...")
    credential = ClientSecretCredential(
        tenant_id=config.config["tenant_id"],
        client_id=config.config["client_id"],
        client_secret=config.config["client_secret"]
    )
    workspace = MLClient(
        subscription_id=config.config["subscription_id"],
        resource_group_name=config.config["resource_group"],
        workspace_name=config.config["workspace_name"],
        credential=credential
    )
    print("Workspace loaded successfully.")
    return workspace

# Function to load the model from Azure ML
def load_model_from_azure(workspace):
    """
    Fetch and load the latest version of a model from Azure ML workspace.

    Input:
        workspace (MLClient): Azure ML workspace object.

    Returns:
        loaded_model (tf.keras.Model): A loaded LSTM model.

    Author:
        Max Meiners (214936)
    """
    print(f"Fetching model '{MODEL_NAME}' from Azure ML workspace...")
    
    # Fetch the latest version of the model
    model_list = workspace.models.list(name=MODEL_NAME)
    latest_model = max(model_list, key=lambda m: m.version)
    
    print(f"Model '{MODEL_NAME}' version '{latest_model.version}' fetched successfully.")
    
    # Download the model locally
    download_path = 'models'
    _ = workspace.models.download(
        name=MODEL_NAME, 
        version=latest_model.version, 
        download_path=download_path
        )
    
    # Locate the actual model file in the download directory
    for root, dirs, files in os.walk(download_path):
        for file in files:
            if file.endswith('.h5'):
                model_file_path = os.path.join(root, file)
                break
    
    print(f"Model downloaded successfully at '{model_file_path}'.")
    
    # Load the model (assuming it's a TensorFlow/Keras model)
    print("Loading the model...")
    loaded_model = tf.keras.models.load_model(model_file_path)
    print("Model loaded successfully.")
    return loaded_model

def preprocess_labels(train_labels):
    """
    Preprocess labels by encoding and one-hot encoding.

    Input:
        train_labels (array-like): Array of training labels.

    Returns:
        train_labels_encoded (np.ndarray): Encoded labels.
        train_labels_one_hot (np.ndarray): One-hot encoded labels.
        label_encoder (LabelEncoder): Fitted label encoder.

    Author:
        Max Meiners (214936)
    """
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels_encoded)
    return train_labels_encoded, train_labels_one_hot, label_encoder

def tokenize_sentences(train_sentences, 
                       test_sentences, 
                       num_words=10000000, 
                       max_length=None):
    """
    Tokenize and pad sequences for training and testing sentences.

    Inout:
        train_sentences (array-like): Array of training sentences.
        test_sentences (array-like): Array of testing sentences.
        num_words (int, optional): Maximum number of words to keep, 
        based on word frequency. Defaults to 10000000.
        max_length (int, optional): Maximum length of sequences. Defaults to None.

    Returns:
        train_padded (np.ndarray): Padded training sequences.
        test_padded (np.ndarray): Padded testing sequences.
        tokenizer (Tokenizer): Fitted tokenizer.
        max_length (int): Maximum length of sequences.

    Author:
        Max Meiners (214936)
    """
    tokenizer = Tokenizer(oov_token="<OOV>", num_words=num_words)
    tokenizer.fit_on_texts(train_sentences)
    tokenizer.fit_on_texts(test_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)

    if max_length is None:
        max_length = max(len(x) for x in train_sequences)

    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post")
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post")

    return train_padded, test_padded, tokenizer, max_length

def split_dataset(train_padded, train_labels_one_hot, test_size=0.2, random_state=42):
    """
    Split dataset into training and validation sets.

    Input:
        train_padded (np.ndarray): Padded training sequences.
        train_labels_one_hot (np.ndarray): One-hot encoded training labels.
        test_size (float, optional): Proportion of the dataset to include 
        in the validation split. Defaults to 0.2.
        random_state (int, optional): Random state for shuffling. Defaults to 42.

    Returns:
        X_train (np.ndarray): Training data.
        X_val (np.ndarray): Validation data.
        y_train (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.

    Author:
        Max Meiners (214936)
    """
    return train_test_split(
        train_padded,
        train_labels_one_hot,
        test_size=test_size,
        random_state=random_state,
    )

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=128):
    """
    Train the model with given training and validation data.

    Input:
        model (tf.keras.Model): The model to train.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        epochs (int, optional): Number of epochs to train. Defaults to 100.
        batch_size (int, optional): Batch size for training. Defaults to 128.

    Returns:
        history (tf.keras.callbacks.History): Training history.

    Author:
        Max Meiners (214936)
    """
    model.compile(optimizer="adam", 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)],
    )
    return history

def predict_and_evaluate(model, X_val, y_val, label_encoder):
    """
    Evaluate the model and predict on validation data.

    Input:
        model (tf.keras.Model): The trained model.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        label_encoder (LabelEncoder): Fitted label encoder.

    Returns:
        predicted_emotions (np.ndarray): Predicted emotions for validation data.

    Author:
        Max Meiners (214936)
    """
    print("Evaluating the model on validation data...")
    results = model.evaluate(X_val, y_val, verbose=0)
    accuracy = results[1]
    print(f'Accuracy: {accuracy}')
    
    y_pred = model.predict(X_val)
    predicted_class_indices = np.argmax(y_pred, axis=1)
    predicted_emotions = label_encoder.inverse_transform(predicted_class_indices)
    return predicted_emotions

def main():
    # Load the workspace
    workspace = load_workspace()

    # Load the model from Azure ML
    loaded_model = load_model_from_azure(workspace)

    # Load your data here
    train = pd.read_csv(
        "/Users/maxmeiners/Library/CloudStorage/OneDrive-BUas"
        "/Github/Year 2/Block D/test_files/test_emotions",
        nrows=500,
    )
    test = pd.read_csv(
        "/Users/maxmeiners/Library/CloudStorage/OneDrive-BUas/"
        "Github/Year 2/Block D/test_files/test_emotions",
        skiprows=range(1, 501),
        nrows=500,
    )

    # Preprocessing steps
    train_sentences = train["sentence"].values
    train_labels = train["emotion"].values
    test_sentences = test["sentence"].values

    _, train_labels_one_hot, label_encoder = preprocess_labels(train_labels)
    train_padded, _, _, _ = tokenize_sentences(train_sentences, test_sentences)
    X_train, X_val, y_train, y_val = split_dataset(train_padded, train_labels_one_hot)

    # Train the model
    train_model(loaded_model, X_train, y_train, X_val, y_val)

    # Predict and evaluate
    _ = predict_and_evaluate(loaded_model, X_val, y_val, label_encoder)


if __name__ == "__main__":
    main()