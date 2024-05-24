import unittest
import model_training
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd

class TestModelTraining(unittest.TestCase):

    def test_load_data(self):
        file_path = '/Users/maxmeiners/Documents/GitHub/2023-24c-fai2-adsai-MaxMeiners/Datasets_new/emotions_all_V6.csv'
        dataset = 'emotions_all_V6.csv'
        self.assertIsNotNone(model_training.load_data(file_path, dataset))

    def test_get_model(self):
        model_path = '/Users/maxmeiners/Downloads/model'
        num_classes = 6
        self.assertIsNotNone(model_training.get_model(model_path, num_classes))

    def test_preprocess_data(self):
        df = '/Users/maxmeiners/Documents/GitHub/2023-24c-fai2-adsai-MaxMeiners/Datasets_new/emotions_all_V6.csv'
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.assertIsNotNone(model_training.preprocess_data(df, tokenizer))

    def test_predict(self):
        model_path = '/Users/maxmeiners/Downloads/model'
        model = TFRobertaForSequenceClassification.from_pretrained(model_path)

        sentences = '/Users/maxmeiners/Documents/GitHub/2023-24c-fai2-adsai-MaxMeiners/Datasets_new/emotions_all.csv'
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Load the data and fit the LabelEncoder
        df = pd.read_csv(sentences)
        label_encoder = LabelEncoder()
        label_encoder.fit(df['emotion'].values)
        self.assertIsNotNone(model_training.predict(model, sentences, tokenizer, label_encoder))

    def test_evaluate(self):
        eval_data = '/Users/maxmeiners/Documents/GitHub/2023-24c-fai2-adsai-MaxMeiners/Datasets_new/emotions_all.csv'

        model_path = '/Users/maxmeiners/Downloads/model'
        model = TFRobertaForSequenceClassification.from_pretrained(model_path)

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        label_encoder = LabelEncoder()
        self.assertIsNotNone(model_training.evaluate(eval_data, model, tokenizer, label_encoder))

if __name__ == '__main__':
    unittest.main()