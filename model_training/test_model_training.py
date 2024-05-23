import unittest
import pandas as pd
import tensorflow as tf
import numpy as np
from unittest.mock import patch, MagicMock
from script import load_data, get_model, preprocess_data, predict, evaluate

class TestScript(unittest.TestCase):

    @patch("script.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        # Mocking read_csv return value
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df
        
        file_path = 'dummy_path.csv'
        dataset = 'train'
        df, num_classes = load_data(file_path, dataset)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(num_classes, 6)

    @patch("script.RobertaConfig.from_pretrained")
    @patch("script.TFRobertaForSequenceClassification.from_pretrained")
    @patch("script.RobertaTokenizer.from_pretrained")
    @patch("script.tf.keras.models.load_model")
    def test_get_model(self, mock_load_model, mock_tokenizer, mock_roberta_model, mock_roberta_config):
        mock_roberta_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        model_path = 'dummy_model_path'
        num_classes = 6

        model, tokenizer = get_model(model_path, num_classes)

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

    def test_preprocess_data(self):
        # Sample data
        df = pd.DataFrame({
            'sentence': ['I am happy', 'I am sad'],
            'emotion': ['happy', 'sad']
        })
        tokenizer = MagicMock()
        tokenizer.encode_plus = MagicMock(return_value={
            'input_ids': tf.constant([[0]*128]), 
            'attention_mask': tf.constant([[0]*128])
        })

        training_dataset, validation_dataset, classes, tokenizer = preprocess_data(df, tokenizer)

        self.assertIsInstance(training_dataset, tf.data.Dataset)
        self.assertIsInstance(validation_dataset, tf.data.Dataset)
        self.assertEqual(len(classes), 2)

    def test_predict(self):
        model = MagicMock()
        tokenizer = MagicMock()
        label_encoder = MagicMock()
        sentences = pd.DataFrame({
            'sentence': ['I am happy', 'I am sad']
        })
        
        model.return_value = MagicMock(logits=tf.constant([[0.5, 0.5], [0.5, 0.5]]))
        label_encoder.inverse_transform = MagicMock(return_value=['happy', 'sad'])

        predicted_emotions, highest_probabilities = predict(model, sentences, tokenizer, label_encoder)

        self.assertEqual(len(predicted_emotions), 2)
        self.assertEqual(len(highest_probabilities), 2)

    @patch("script.predict")
    def test_evaluate(self, mock_predict):
        eval_data = pd.DataFrame({
            'sentence': ['I am happy', 'I am sad'],
            'emotion': ['happy', 'sad']
        })
        model = MagicMock()
        tokenizer = MagicMock()
        label_encoder = MagicMock()

        mock_predict.return_value = (['happy', 'sad'], [0.9, 0.8])

        predicted_emotions, highest_probabilities, accuracy, report = evaluate(eval_data, model, tokenizer, label_encoder)

        self.assertEqual(len(predicted_emotions), 2)
        self.assertEqual(len(highest_probabilities), 2)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(report, str)

if __name__ == '__main__':
    unittest.main()