"""
Tests for model_training.py.

Author: Max Meiners (214936)
"""

# import tensorflow as tf
import model_training
import pandas as pd
import pytest
import argparse
import sys
# from transformers import (
# TFRobertaForSequenceClassification, 
# RobertaConfig)
# import sys
# from sklearn.preprocessing import LabelEncoder


# Set recursion limit
# sys.setrecursionlimit(150000)

class TestModelTraining:
    

    @pytest.mark.parametrize(
        "mt_args",
        [
            [
                "--cloud", "True",
                "--dataset_name_file", "dataset_name_file",
                "--epochs", "10",
                "--learning_rate", "0.001",
                "--batch_size", "32",
                "--early_stopping_patience", "5", 
                "--model_output_path", "model_output_path",
                "--decoder_output_path", "decoder_output_path"
            ]
        ],
    )
    def test_get_args(self, monkeypatch, mt_args):
        # Author - Kornelia Flizik
        monkeypatch.setattr(sys, "argv", ["model_training"] + mt_args)

        args = model_training.get_args()

        assert args.cloud == True  # Assuming you want to check for True (not the boolean value)
        assert args.dataset_name_file == "dataset_name_file"
        assert args.epochs == 10
        assert args.learning_rate == 0.001
        assert args.batch_size == 32
        assert args.early_stopping_patience == 5
        assert args.model_output_path == "model_output_path"
        assert args.decoder_output_path == "decoder_output_path"


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

    # def test_train_model(self):
    #     num_classes = 6
    #     config = RobertaConfig.from_pretrained('model/config.json')
    #     model = TFRobertaForSequenceClassification.from_pretrained(
    # 'roberta-base', 
    # config=config
    # )
        
    #     # Load datasets
    #     data = {'sentence': ['I am happy', 'I am sad'], 
    # 'emotion': ['happiness', 'sadness']}
    #     training_dataset = pd.DataFrame(data)
    #     validation_dataset = pd.DataFrame(data)
        
    #     # Preprocess data
    #     label_encoder = LabelEncoder()
    #     label_encoder.fit(training_dataset['emotion'])
        
    #     training_dataset['label'] = label_encoder.transform(training_dataset['emotion'])
    #     validation_dataset['label'] = label_encoder.transform(validation_dataset['emotion'])
        
    #     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
    #     def encode_data(df):
    #         return tokenizer(
    #             df['sentence'].tolist(), 
    #             padding=True, 
    #             truncation=True, 
    #             return_tensors="tf"
    #         )
        
    #     train_encodings = encode_data(training_dataset)
    #     val_encodings = encode_data(validation_dataset)
        
    #     train_dataset = tf.data.Dataset.from_tensor_slices((
    #         dict(train_encodings),
    #         training_dataset['label'].values
    #     )).batch(2)
        
    #     val_dataset = tf.data.Dataset.from_tensor_slices((
    #         dict(val_encodings),
    #         validation_dataset['label'].values
    #     )).batch(2)
        
    #     # Training parameters
    #     epochs = 2
    #     learning_rate = 1e-3
    #     early_stopping_patience = 3

    #     # Train the model
    #     trained_model = model_training.train_model(
    #         model,
    #         train_dataset,
    #         val_dataset,
    #         epochs,
    #         learning_rate,
    #         early_stopping_patience
    #         )
        
    #     # Assertions to validate the model
    #     assert trained_model is not None
            

if __name__ == "__main__":
    pytest.main()
