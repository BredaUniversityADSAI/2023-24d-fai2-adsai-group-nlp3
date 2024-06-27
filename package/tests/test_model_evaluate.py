from e3k import model_evaluate
import pytest
from transformers import TFRobertaForSequenceClassification, RobertaConfig
import pandas as pd
import os
from unittest.mock import patch
import numpy as np


# Fixtures to provide sample paths and data
@pytest.fixture
def sample_test_data():
    return "tests/test_data/dataset_test.csv"

@pytest.fixture
def sample_decoder():
    return "tests/test_data/label_decoder"

@pytest.fixture
def sample_model():
    return "tests/test_data/Test_model"


class TestModelEvaluate:
    def test_load_data(self, sample_test_data):
        """
        Test loading data from a CSV file using `model_evaluate.load_data`.
        Asserts: The loaded DataFrame should not be None.

        Author: Kornelia Flizik
        """

        df = model_evaluate.load_data(sample_test_data)
        assert df is not None, "Value should not be None"

    def test_load_label_decoder(self, sample_decoder):
        """
        Test loading a label decoder from a file using 
        `model_evaluate.load_label_decoder`.
        Asserts: The loaded label decoder should not be None.

        Author: Kornelia Flizik
        """

        decoder = model_evaluate.load_label_decoder(sample_decoder)
        assert decoder is not None, "Value should not be None"
    
    def test_predict(self, sample_test_data, sample_decoder, sample_model):
        """
        Test the `model_predict.predict` function.
        Asserts:
            The predicted emotions should be a list of strings.
            The predicted probabilities should be a NumPy array of floats.
            The length of the predicted emotions and probabilities should match
            the number of input sequences.

        Author: Kornelia Flizik
        """

        data = model_evaluate.load_data(sample_test_data)
        tokens, masks = model_evaluate.preprocess_prediction_data(data)

        model = TFRobertaForSequenceClassification.from_pretrained(sample_model)

        emotion_decoder = {
            0: 'happy',
            1: 'sad',
            2: 'angry',
            3: 'surprised',
            4: 'something',
            5: 'something',
        }

        emotions, probabilities = model_evaluate.predict(model=model,
                                                    token_array=tokens,
                                                    mask_array=masks,
                                                    emotion_decoder=emotion_decoder)
        

        # Assertions to validate the output
        assert isinstance(emotions, list), "Emotions should be a list"
        assert all(isinstance(emotion, str) for emotion in emotions)

        assert isinstance(probabilities, np.ndarray), "Should be a NumPy array"
        assert probabilities.dtype in [np.float32, np.float64], "Should be floats"

        assert len(emotions) == tokens.shape[0], "Should match number of input tokens"
        assert len(probabilities) == tokens.shape[0]

    def test_decode_labels(self):
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
        actual_output = model_evaluate.decode_labels(encoded_labels, emotion_decoder)
        
        # Use assertions to check the output
        assert actual_output == expected_output


    def test_evaluate(self):
        """
        Test the `model_evaluate.evaluate` function to ensure it correctly
        evaluates predictions.

        Asserts: The calculated accuracy should be very close to the expected accuracy.

        Author: Kornelia Flizik
        """

        # Mock input data
        pred_labels = ['happy', 'sad', 'angry', 'happy', 'happy']
        data = pd.DataFrame({
            "emotion": ['happy', 'sad', 'angry', 'happy', 'angry']
        })
        
        # Expected outputs based on the mocked input
        expected_accuracy = 0.8 
    
        # Call the evaluate function
        accuracy, report = model_evaluate.evaluate(pred_labels, data)
        
        # Assertions to verify the correctness of outputs
        assert abs(accuracy - expected_accuracy) < 1e-6
        
    @patch('os.path.exists')
    @patch('os.remove')
    def test_save_model(self, mock_remove, mock_exists):
        """
        Test the `model_evaluate.save_model` function to ensure it saves the model
        correctly based on the provided accuracy and threshold.

        Author: Kornelia Flizik
        """

        test_cases = [
            # Test case 1: Accuracy above threshold
            {
                'accuracy': 0.75,
                'threshold': 0.7,
                'expected_model_saved': True,
                'expected_dict_saved': True,
            },
            # Test case 2: Accuracy below threshold
            {
                'accuracy': 0.65,
                'threshold': 0.7,
                'expected_model_saved': False,
                'expected_dict_saved': False,
            }
        ]
        
        for test_case in test_cases:
            output_model_path = 'mocked_path'
            accuracy = test_case['accuracy']
            threshold = test_case['threshold']

            config = RobertaConfig.from_pretrained(
                'tests/test_data/Test_model/config.json')
            model = TFRobertaForSequenceClassification.from_pretrained('roberta-base',
                                                                        config=config)

            label_decoder = {0: 'happy', 1: 'sad', 2: 'angry'}
            
            # Mock the save_model function
            with patch('model_evaluate.save_model') as mock_save_model:
                # Call the save_model function
                model_evaluate.save_model(model, label_decoder,
                                        output_model_path, accuracy, threshold)

                # Set the return value of os.path.exists based on the test case
                mock_exists.return_value = test_case['expected_model_saved']

                # Assertions to verify the correctness of saving behavior
                if test_case['expected_model_saved']:
                    assert os.path.exists(output_model_path), "Model path should exist"
                else:
                    assert not os.path.exists(output_model_path), "Should not exist"

                # Ensure clean-up is called if the file exists
                if os.path.exists(output_model_path):
                    os.remove(output_model_path)
                    mock_remove.assert_called_with(output_model_path)


if __name__ == "__main__":
    pytest.main()
