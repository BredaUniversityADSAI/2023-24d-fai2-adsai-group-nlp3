import model_training
import pandas as pd
import pytest
import argparse
import tensorflow as tf
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
        client = model_training.get_ml_client(
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
        local_dataset_info_file = ("/Users/maxmeiners/Library/CloudStorage/"
        "OneDrive-BUas/Github/Year 2/Block D/test_files/pytest_args.json")

        # Mock args
        args = argparse.Namespace(dataset_name_file=local_dataset_info_file)

        # Get ML client
        ml_client = model_training.get_ml_client(
            azure_credentials["subscription_id"],
            azure_credentials["tenant_id"],
            azure_credentials["client_id"],
            azure_credentials["client_secret"],
            azure_credentials["resource_group"],
            azure_credentials["workspace_name"]
        )
        assert model_training.get_versioned_datasets(args, ml_client) is not None

    def test_get_data_asset_as_df(self, azure_credentials):
        ml_client = model_training.get_ml_client(
            azure_credentials["subscription_id"],
            azure_credentials["tenant_id"],
            azure_credentials["client_id"],
            azure_credentials["client_secret"],
            azure_credentials["resource_group"],
            azure_credentials["workspace_name"]
        )
        dataset_name = "max_test"
        dataset_version = "1"
        assert model_training.get_data_asset_as_df(
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
        assert model_training.get_label_decoder(series) is not None

    def test_get_new_model(self):
        num_classes = 6
        assert model_training.get_new_model(num_classes) is not None

    def test_train_model(self):
        num_classes = 6
        model = model_training.get_new_model(num_classes=num_classes)
        
        # Load datasets
        training_dataset = pd.read_csv("/Users/maxmeiners/Library/CloudStorage/"
                                       "OneDrive-BUas/Github/Year 2/Block D/"
                                       "test_files/test_emotions")
        validation_dataset = pd.read_csv("/Users/maxmeiners/Library/CloudStorage/"
                                         "OneDrive-BUas/Github/Year 2/Block D/"
                                         "test_files/test_emotions_eval")
        
        # Preprocess data
        label_encoder = LabelEncoder()
        label_encoder.fit(training_dataset['emotion'])
        
        training_dataset['label'] = label_encoder.transform(training_dataset['emotion'])
        validation_dataset['label'] = label_encoder.transform(validation_dataset['emotion'])
        
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        def encode_data(df):
            return tokenizer(
                df['sentence'].tolist(), 
                padding=True, 
                truncation=True, 
                return_tensors="tf"
            )
        
        train_encodings = encode_data(training_dataset)
        val_encodings = encode_data(validation_dataset)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            training_dataset['label'].values
        )).batch(8)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            validation_dataset['label'].values
        )).batch(8)
        
        # Training parameters
        epochs = 2
        learning_rate = 1e-3
        early_stopping_patience = 3
        
        # Train the model
        trained_model = model_training.train_model(
            model,
            train_dataset,
            val_dataset,
            epochs,
            learning_rate,
            early_stopping_patience
        )
    
        # Assertions to validate the model
        assert trained_model is not None


if __name__ == "__main__":
    pytest.main()
