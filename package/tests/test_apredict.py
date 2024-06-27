"""
Tests for model_training.py.

Author: Max Meiners (214936)
"""

import model_predict
import pandas as pd
import pytest
import argparse
import sys
import tempfile
import os
import pickle
from transformers import TFRobertaForSequenceClassification
import numpy as np


@pytest.fixture
def sample_model_path():
    return "tests/test_data/test_get_model"

@pytest.fixture
def sample_model():
    return "tests/test_data/Test_model"

@pytest.fixture
def sample_test_data():
    return "tests/test_data/dataset_test.csv"

def test_get_model_new_model(sample_model_path):
    """Test creating a new model."""

    model, emotion_dict = model_predict.get_model(sample_model_path)

    assert isinstance(model, TFRobertaForSequenceClassification)
    assert emotion_dict is not None

def test_decode_labels():
    # Define the input and expected output
    encoded_labels = np.array([1, 2, 3, 0])
    emotion_decoder = {
        0: 'happy',
        1: 'sad',
        2: 'angry',
        3: 'surprised'
    }
    expected_output = ['sad', 'angry', 'surprised', 'happy']
    
    # Call the function with the test inputs
    actual_output = model_predict.decode_labels(encoded_labels, emotion_decoder)
    
    # Use assertions to check the output
    assert actual_output == expected_output


def test_predict(sample_test_data, sample_model):
    data = pd.read_csv(sample_test_data)[["sentence", "emotion"]].dropna()

    tokens, masks = model_predict.preprocess_prediction_data(data)
    model = TFRobertaForSequenceClassification.from_pretrained(sample_model)
    #label_decoder = model_evaluate.load_label_decoder(sample_decoder)
    emotion_decoder = {
        0: 'happy',
        1: 'sad',
        2: 'angry',
        3: 'surprised',
        4: 'something',
        5: 'something',
    }
    emotions, probabilities = model_predict.predict(model=model, token_array=tokens,
                            mask_array=masks, emotion_decoder=emotion_decoder)
    
    # Assertions to validate the output
    assert isinstance(emotions, list), "Emotions should be a list"
    assert all(isinstance(emotion, str) for emotion in emotions), "All emotions should be strings"

    assert isinstance(probabilities, np.ndarray), "Probabilities should be a NumPy array"
    assert probabilities.dtype in [np.float32, np.float64], "All probabilities should be floats"

    assert len(emotions) == tokens.shape[0], "Number of predicted emotions should match number of input sequences"
    assert len(probabilities) == tokens.shape[0], "Number of probabilities should match number of input sequences"
    

if __name__ == "__main__":
    pytest.main()
