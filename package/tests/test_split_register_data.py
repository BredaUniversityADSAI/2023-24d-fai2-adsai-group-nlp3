import pytest
from split_register_data import load_data, get_train_val_data
import pandas as pd


# Fixture to provide sample test data
@pytest.fixture
def sample_test_data():
    """
    Fixture to provide the path to the sample test dataset.
    
    Returns:
        str: Path to the sample test dataset CSV file.
    """
    return "tests/test_data/dataset_test.csv"

class TestSplit_register_data:
    def test_load_data(self, sample_test_data):
        """
        Test loading data from a CSV file.
        Asserts: The dataframe loaded from the CSV file should not be None.

        Author - Kornelia Flizik
        """
    
        df = load_data(sample_test_data)
        assert df is not None, "Value should not be None"

    def test_get_train_val_data(self, sample_test_data):
        """
        Test splitting the data into training and validation sets.
        Asserts: The training set and validation set returned should 
        be pandas DataFrames.

        Author - Kornelia Flizik
        """

        data_df=pd.read_csv(sample_test_data).dropna()

        train_set, val_set = get_train_val_data(data_df, val_size=0.25)

        # Check if the returned types are correct
        assert isinstance(train_set, pd.DataFrame) 
        assert isinstance(val_set, pd.DataFrame)


if __name__ == "__main__":
    pytest.main()

