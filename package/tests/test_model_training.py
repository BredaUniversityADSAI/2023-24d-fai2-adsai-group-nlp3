import model_training
import pandas as pd
import pytest
import sys


class TestModelTraining:
    
    @pytest.mark.parametrize(
        "mt_args",
        [
            [
                "--cloud", "True",
                "--dataset_name_file", "dataset_name_file",
                "--epochs", "10",
                "--learning_rate", "0.001",
                "--batch_size", "32",
                "--early_stopping_patience", "5", 
                "--model_output_path", "model_output_path",
                "--decoder_output_path", "decoder_output_path"
            ]
        ],
    )

    def test_get_args(self, monkeypatch, mt_args):
        """
        Test parsing command line arguments using the `model_training.get_args`.
        Asserts: The parsed arguments should match the expected values.

        Author - Kornelia Flizik
        """

        monkeypatch.setattr(sys, "argv", ["model_training"] + mt_args)

        args = model_training.get_args()

        assert args.cloud == True  
        assert args.dataset_name_file == "dataset_name_file"
        assert args.epochs == 10
        assert args.learning_rate == 0.001
        assert args.batch_size == 32
        assert args.early_stopping_patience == 5
        assert args.model_output_path == "model_output_path"
        assert args.decoder_output_path == "decoder_output_path"

    def test_get_label_decoder(self):
        """
        Test the `model_training.get_label_decoder` function to ensure
        it returns a non-null decoder.
        Asserts: The returned decoder should not be None.

        Author: Max Meiners (214936)
        """

        series = pd.Series(["happiness", 
                            "sadness", 
                            "anger", 
                            "fear", 
                            "surprise", 
                            "disgust"])
        assert model_training.get_label_decoder(series) is not None

    def test_get_new_model(self):
        """
        Test the `model_training.get_new_model` to ensure it returns a non-null model.
        Asserts: The returned model should not be None.

        Author: Max Meiners (214936)
        """
        
        num_classes = 6
        assert model_training.get_new_model(num_classes) is not None
            

if __name__ == "__main__":
    pytest.main()
