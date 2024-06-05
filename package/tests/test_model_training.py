import model_training
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from ek3.model_training import (load_data)


class TestModelTraining:
    def test_load_data(self):
        file_path = "/Users/maxmeiners/Downloads/model/test_emotions"
        dataset = "emotions_all_V6.csv"
        self.assertIsNotNone(model_training.load_data(file_path, dataset))

    def test_get_model(self):
        model_path = "/Users/maxmeiners/Downloads/model"
        num_classes = 6
        self.assertIsNotNone(model_training.get_model(model_path, num_classes))

    def test_preprocess_data(self):
        file_path = "/Users/maxmeiners/Downloads/model/test_emotions"
        df = pd.read_csv(file_path)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.assertIsNotNone(model_training.preprocess_data(df, tokenizer))

    def test_train_model(self):
        train_data_path = "/Users/maxmeiners/Downloads/model/test_emotions"
        train_data = pd.read_csv(train_data_path)
        model_path = "/Users/maxmeiners/Downloads/model"
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        num_classes = 6
        self.assertIsNotNone(
            model_training.train_model(train_data, 
                                       model_path, 
                                       tokenizer, 
                                       num_classes)
        )

    def test_predict(self):
        model_path = "/Users/maxmeiners/Downloads/model"
        model = TFRobertaForSequenceClassification.from_pretrained(model_path)

        sentences = "/Users/maxmeiners/Downloads/model/test_emotions_eval"
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Load the data and fit the LabelEncoder
        df = pd.read_csv(sentences)
        label_encoder = LabelEncoder()
        label_encoder.fit(df["emotion"].values)
        self.assertIsNotNone(
            model_training.predict(model, sentences, tokenizer, label_encoder)
        )

    def test_evaluate(self):
        eval_data_path = "/Users/maxmeiners/Downloads/model/test_emotions_eval"
        eval_data = pd.read_csv(eval_data_path)

        model_path = "/Users/maxmeiners/Downloads/model"
        model = TFRobertaForSequenceClassification.from_pretrained(model_path)

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        label_encoder = LabelEncoder()
        label_encoder.fit(eval_data["emotion"].values)
        self.assertIsNotNone(
            model_training.evaluate(eval_data, model, tokenizer, label_encoder)
        )


if __name__ == "__main__":
    pytest.main()
