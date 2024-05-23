import unittest
import model_training
from transformers import RobertaTokenizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

class TestModelTraining(unittest.TestCase):

    def test_load_data(self):
        file_path = '/Users/maxmeiners/Documents/GitHub/2023-24c-fai2-adsai-MaxMeiners/Datasets_new/emotions_all_V6.csv'
        dataset = 'emotions_all_V6.csv'
        nrows = 10000
        df = model_training.load_data(file_path, dataset, nrows)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 10000)
        
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
        model = tf.keras.models.load_model(model_path)
        sentences = '/Users/maxmeiners/Documents/GitHub/2023-24c-fai2-adsai-MaxMeiners/Datasets_new/emotions_all.csv'
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        label_encoder = LabelEncoder()
        self.assertIsNotNone(model_training.predict(model, sentences, tokenizer, label_encoder))

if __name__ == '__main__':
    unittest.main()