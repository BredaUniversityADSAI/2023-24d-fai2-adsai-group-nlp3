import model_predict
import pandas as pd
import pytest

from transformers import TFRobertaForSequenceClassification
import numpy as np


# Fixtures to provide sample paths and data
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
    """
    Test creating a new model using `model_predict.get_model`.
    Asserts:
        The model should be an instance of TFRobertaForSequenceClassification.
        The emotion dictionary should not be None.

    Author: Kornelia Flizik
    """

    model, emotion_dict = model_predict.get_model(sample_model_path)

    assert isinstance(model, TFRobertaForSequenceClassification)
    assert emotion_dict is not None

def test_decode_labels():
    """
    Test the `model_predict.decode_labels` function.
    Asserts: The decoded labels should match the expected output.

    Author: Kornelia Flizik
    """

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
    """
    Test the `model_predict.predict` function.
    Asserts:
        The predicted emotions should be a list of strings.
        The predicted probabilities should be a NumPy array of floats.
        The length of the predicted emotions and probabilities should match
        the number of input sequences.

    Author: Kornelia Flizik
    """

    data = pd.read_csv(sample_test_data)[["sentence", "emotion"]].dropna()

    tokens, masks = model_predict.preprocess_prediction_data(data)
    model = TFRobertaForSequenceClassification.from_pretrained(sample_model)

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
    assert all(isinstance(emotion, str) for emotion in emotions), "Should be strings"

    assert isinstance(probabilities, np.ndarray), "Should be a NumPy array"
    assert probabilities.dtype in [np.float32, np.float64], "Should be floats"

    assert len(emotions) == tokens.shape[0], "Should match number of input sequences"
    assert len(probabilities) == tokens.shape[0], "Should match number of input tokens"
    

if __name__ == "__main__":
    pytest.main()
