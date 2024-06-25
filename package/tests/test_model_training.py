import e3k.model_training
import pandas as pd
import pytest
import argparse
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification


@pytest.fixture
def azure_credentials():
    return {
        "subscription_id": "0a94de80-6d3b-49f2-b3e9-ec5818862801",
        "tenant_id": "0a33589b-0036-4fe8-a829-3ed0926af886",
        "client_id": "a2230f31-0fda-428d-8c5c-ec79e91a49f5",
        "client_secret": "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C",
        "resource_group": "buas-y2",
        "workspace_name": "NLP3"
    }

@pytest.mark.usefixtures("azure_credentials")
class TestModelTraining:

    def test_get_ml_client(self, azure_credentials):
        client = e3k.model_training.get_ml_client(
            azure_credentials["subscription_id"],
            azure_credentials["tenant_id"],
            azure_credentials["client_id"],
            azure_credentials["client_secret"],
            azure_credentials["resource_group"],
            azure_credentials["workspace_name"]
        )
        assert client is not None

    def test_get_versioned_datasets(self, azure_credentials):
        # Path to the local JSON dataset info file
        local_dataset_info_file = (
            "/Users/maxmeiners/Library/CloudStorage/OneDrive-BUas"
            "/Github/Year 2/Block D/test_files/pytest_args.json"
            )

        # Mock args
        args = argparse.Namespace(dataset_name_file=local_dataset_info_file)

        # Get ML client
        ml_client = e3k.model_training.get_ml_client(
            azure_credentials["subscription_id"],
            azure_credentials["tenant_id"],
            azure_credentials["client_id"],
            azure_credentials["client_secret"],
            azure_credentials["resource_group"],
            azure_credentials["workspace_name"]
        )
        assert e3k.model_training.get_versioned_datasets(args, ml_client) is not None

    def test_get_data_asset_as_df(self, azure_credentials):
        ml_client = e3k.model_training.get_ml_client(
            azure_credentials["subscription_id"],
            azure_credentials["tenant_id"],
            azure_credentials["client_id"],
            azure_credentials["client_secret"],
            azure_credentials["resource_group"],
            azure_credentials["workspace_name"]
        )
        dataset_name = "val_data"
        dataset_version = "12"
        assert e3k.model_training.get_data_asset_as_df(
            ml_client,
            dataset_name,
            dataset_version) is not None

    def test_get_label_decoder(self):
        series = pd.Series(["happiness", 
                            "sadness", 
                            "anger", 
                            "fear", 
                            "surprise", 
                            "disgust"])
        assert e3k.model_training.get_label_decoder(series) is not None

    def test_get_new_model(self):
        num_classes = 6
        assert e3k.model_training.get_new_model(num_classes) is not None

    def test_train_model(self):
        num_classes = 6
        model = e3k.model_training.get_new_model(num_classes=num_classes)

        # Load datasets (first 50 rows only)
        training_dataset = pd.read_csv(
            "package/tests/test_files/test_emotions",
            nrows=50
        )
        validation_dataset = pd.read_csv(
            "package/tests/test_files/test_emotions_eval",
            nrows=50
        )

        # Encode labels to numeric values
        label_encoder = LabelEncoder()
        training_dataset['label'] = label_encoder.fit_transform(training_dataset['emotion'])
        validation_dataset['label'] = label_encoder.transform(validation_dataset['emotion'])

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Inline tokenization
        train_encodings = tokenizer(
            training_dataset['sentence'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=16,
            return_tensors="tf"
        )
        val_encodings = tokenizer(
            validation_dataset['sentence'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=16,
            return_tensors="tf"
        )

        # Convert labels to NumPy arrays and then to tensors
        train_labels = tf.convert_to_tensor(np.array(training_dataset['label'].values), dtype=tf.int32)
        val_labels = tf.convert_to_tensor(np.array(validation_dataset['label'].values), dtype=tf.int32)

        # Print tensor shapes for debugging
        print("Train encodings shape:", {k: v.shape for k, v in train_encodings.items()})
        print("Validation encodings shape:", {k: v.shape for k, v in val_encodings.items()})
        print("Train labels shape:", train_labels.shape)
        print("Validation labels shape:", val_labels.shape)

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        )).batch(8)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val_labels
        )).batch(8)

        # Training parameters
        epochs = 2
        learning_rate = 1e-3
        early_stopping_patience = 3

        # Train the model
        trained_model = e3k.model_training.train_model(
            model,
            train_dataset,
            val_dataset,
            epochs,
            learning_rate,
            early_stopping_patience
            )
        assert trained_model is not None



if __name__ == "__main__":
    pytest.main()
