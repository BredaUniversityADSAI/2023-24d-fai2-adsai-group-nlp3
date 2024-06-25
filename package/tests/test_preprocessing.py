"""
Test suite for the preprocessing module.

Author: Juraj Kret, 221439
Date: 25th of June, 2024
"""


import pytest
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from e3k.preprocessing import (
    get_tokenizer,
    tokenize_text_data,
    encode_labels,
    create_tf_dataset,
    preprocess_training_data,
    preprocess_prediction_data,
    preprocess_prediction_data_no_tokenizer,
)

class TestPreprocessing:
    
    # Test the get_tokenizer function
    def test_get_tokenizer(self):
        """
        Test if the get_tokenizer function returns a valid tokenizer
        for the specified model name.
        """
        tokenizer = get_tokenizer("roberta-base")
        assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))

    # Test the tokenize_text_data function
    def test_tokenize_text_data(self):
        """
        Test if the tokenize_text_data function correctly tokenizes
        the input text data using the specified tokenizer.
        """
        data = pd.Series(["This is a test sentence.", "Another test sentence."])
        tokenizer = get_tokenizer("roberta-base")
        input_ids, attention_masks = tokenize_text_data(data, tokenizer)
        assert input_ids.shape == (2, 128)
        assert attention_masks.shape == (2, 128)

    # Test the encode_labels function
    def test_encode_labels(self):
        """
        Test if the encode_labels function correctly encodes
        the input labels using the provided label decoder.
        """
        labels = pd.Series(["happy", "sad", "angry"])
        label_decoder = {0: "happy", 1: "sad", 2: "angry"}
        encoded_labels = encode_labels(labels, label_decoder)
        assert np.array_equal(encoded_labels, np.array([0, 1, 2]))

    # Test the create_tf_dataset function
    def test_create_tf_dataset(self):
        """
        Test if the create_tf_dataset function correctly creates
        a TensorFlow dataset from the tokenized data and labels.
        """
        tokens = np.random.randint(0, 100, (3, 128))
        masks = np.ones((3, 128))
        labels = np.array([0, 1, 2])
        dataset = create_tf_dataset(tokens, masks, labels, batch_size=2)
        assert len(list(dataset)) == 2  # 2 batches due to batch_size=2

    # Test the preprocess_training_data function
    def test_preprocess_training_data(self):
        """
        Test if the preprocess_training_data function correctly preprocesses
        the training and validation data and returns TensorFlow datasets.
        """
        train_data = pd.DataFrame({
            "sentence": ["This is a train sentence.", "Another train sentence."],
            "emotion": ["happy", "sad"]
        })
        val_data = pd.DataFrame({
            "sentence": ["This is a val sentence.", "Another val sentence."],
            "emotion": ["happy", "sad"]
        })
        label_decoder = {0: "happy", 1: "sad"}
        train_dataset, val_dataset = preprocess_training_data(train_data, val_data, label_decoder)
        assert len(list(train_dataset)) > 0
        assert len(list(val_dataset)) > 0

    # Test the preprocess_prediction_data function
    def test_preprocess_prediction_data(self):
        """
        Test if the preprocess_prediction_data function correctly preprocesses
        the input data for prediction, including tokenization.
        """
        data = pd.DataFrame({"sentence": ["This is a test sentence.", "Another test sentence."]})
        tokens, masks = preprocess_prediction_data(data)
        assert tokens.shape == (2, 128)
        assert masks.shape == (2, 128)

    # Test the preprocess_prediction_data_no_tokenizer function
    def test_preprocess_prediction_data_no_tokenizer(self):
        """
        Test if the preprocess_prediction_data_no_tokenizer function correctly
        preprocesses the input data for prediction when a tokenizer is provided.
        """
        data = pd.DataFrame({"sentence": ["This is a test sentence.", "Another test sentence."]})
        tokenizer = get_tokenizer("roberta-base")
        tokens, masks = preprocess_prediction_data_no_tokenizer(data, tokenizer)
        assert tokens.shape == (2, 128)
        assert masks.shape == (2, 128)

if __name__ == "__main__":
    pytest.main()
