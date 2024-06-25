from e3k import model_evaluate
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, RobertaConfig


@pytest.fixture(autouse=True)
class TestModelEvaluate:
    def test_load_data(self):
        file_path = "test_data/dataset_test.csv"
        self.assertIsNotNone(model_evaluate.load_data(file_path))


    def test_load_label_decoder(self):
        encoder_path = "test_data/test_label_decoder"
        self.assertIsNotNone(model_evaluate.load_label_decoder(encoder_path))


    def test_predict():
        data_path = "test_data/dataset_test.csv"
        data = model_evaluate.load_data(data_path)
        tokens, masks = model_evaluate.preprocess_prediction_data(data)

        config = RobertaConfig.from_pretrained('test_data/Test_model/config.json')

        model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
        label_decoder = model_evaluate.load_label_decoder("test_data/label_decoder")
        emotions, probabilities = model_evaluate.predict(model=model, token_array=tokens,
                                mask_array=masks, emotion_decoder=label_decoder)

        # Assertions to validate the output
        assert isinstance(emotions, list), "Emotions should be a list"
        assert isinstance(probabilities, list), "Probabilities should be a list"

        assert all(isinstance(emotion, str) for emotion in emotions), "All emotions should be strings"
        assert all(isinstance(prob, float) for prob in probabilities), "All probabilities should be floats"
        
        assert len(emotions) == tokens.shape[0], "Number of predicted emotions should match number of input sequences"
        assert len(probabilities) == tokens.shape[0], "Number of probabilities should match number of input sequences"
            

    def test_decode_labels():
        # Define the input and expected output
        encoded_labels = [1, 2, 3, 0]
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


    def test_evaluate():
        # Mock input data
        pred_labels = ['happy', 'sad', 'angry', 'happy', 'happy']
        data = {
            "emotion": ['happy', 'sad', 'angry', 'happy', 'angry']
        }
        
        # Expected outputs based on the mocked input
        expected_accuracy = 0.8  # This is an example value; you should calculate the actual expected accuracy
    
        # Call the evaluate function
        accuracy, report = model_evaluate.evaluate(pred_labels, data)
        
        # Assertions to verify the correctness of outputs
        assert abs(accuracy - expected_accuracy) < 1e-6, f"Accuracy calculation is
          incorrect: expected {expected_accuracy}, got {accuracy}"
        

    def test_save_model(self):
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

            config = RobertaConfig.from_pretrained('test_data/Test_model/config.json')
            model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', config=config)

            label_decoder = {0: 'happy', 1: 'sad', 2: 'angry'}
            
            # Call the save_model function
            model_evaluate.save_model(model, label_decoder, output_model_path, accuracy, threshold)
            
            # Assertions to verify the correctness of saving behavior
            if test_case['expected_model_saved']:
                model.save_pretrained.assert_called_once_with(output_model_path)
            else:
                model.save_pretrained.assert_not_called()


if __name__ == "__main__":
    pytest.main()
