import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Embedding,
                                     GlobalMaxPool1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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


def preprocess_labels(train_labels):
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels_encoded)
    return train_labels_encoded, train_labels_one_hot, label_encoder


def tokenize_sentences(
    train_sentences, test_sentences, num_words=10000000, max_length=None
):
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
    return train_test_split(
        train_padded,
        train_labels_one_hot,
        test_size=test_size,
        random_state=random_state,
    )


def build_model(vocab_size, max_length, num_emotions, embedding_dim=64):
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=400, input_length=max_length),
            Bidirectional(LSTM(516, return_sequences=True)),
            Dense(256, activation="relu"),
            GlobalMaxPool1D(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(num_emotions, activation="softmax"),
        ]
    )

    model.build(input_shape=(None, max_length))
    model.summary()
    return model


def compile_model(model):
    def f1_score(y_true, y_pred):
        true_positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1))
        )
        possible_positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
        )
        predicted_positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))
        )

        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

        f1_val = (
            2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        )
        return f1_val

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[f1_score])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=128):
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
    y_pred = model.predict(X_val)
    predicted_class_indices = np.argmax(y_pred, axis=1)
    _ = np.argmax(y_val, axis=1)
    predicted_emotions = label_encoder.inverse_transform(predicted_class_indices)

    return predicted_emotions


# def save_predictions(test, predicted_emotions, output_filename):
#     test['emotion'] = predicted_emotions
#     test = test.drop(columns=['sentence'])
#     test.to_csv(output_filename, index=False)


def main():
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
    train_padded, _, tokenizer, max_length = tokenize_sentences(
        train_sentences, test_sentences
    )
    X_train, X_val, y_train, y_val = split_dataset(train_padded, train_labels_one_hot)

    unique_emotions = train["emotion"].unique()
    num_emotions = len(unique_emotions)
    vocab_size = len(tokenizer.word_index) + 1

    # Model creation and training
    model = build_model(vocab_size, max_length, num_emotions)
    model = compile_model(model)
    _ = train_model(model, X_train, y_train, X_val, y_val)

    # Prediction and evaluation
    _ = predict_and_evaluate(model, X_val, y_val, label_encoder)

    # # Save predictions
    # save_predictions(test, predicted_emotions, 'rnn_model_max_4_moredata.csv')


main()
