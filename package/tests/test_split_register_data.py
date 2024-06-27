import pytest
import logging
from matplotlib import pyplot as plt
from unittest.mock import patch
from split_register_data import load_data, get_train_val_data
import pandas as pd

# To handle the logger's file handlers during tests to avoid creating files
@pytest.fixture
def sample_test_data():
    return "tests/test_data/dataset_test.csv"

class TestSplit_register_data:
    def test_load_data(self, sample_test_data):
        # Author - Kornelia Flizik
        df = load_data(sample_test_data)
        assert df is not None, "Value should not be None"

    def test_get_train_val_data(self, sample_test_data):

        data_df=pd.read_csv(sample_test_data).dropna()

        train_set, val_set = get_train_val_data(data_df, val_size=0.25)

        # Check if the returned types are correct
        assert isinstance(train_set, pd.DataFrame) 
        assert isinstance(val_set, pd.DataFrame)


if __name__ == "__main__":
    pytest.main()

