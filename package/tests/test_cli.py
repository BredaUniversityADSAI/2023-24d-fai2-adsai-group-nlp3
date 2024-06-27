import os
import sys

import pandas as pd
import pytest
from e3k.cli import (episode_preprocessing, get_args, model_output_information,
                     model_training, predict)

@pytest.fixture
def sample_test_data():
    return "tests/test_data/dataset_test.csv"

@pytest.fixture
def sample_video_path():
    return "tests/test_data/video_test.mov"

class TestCLI:
    @pytest.mark.parametrize(
        "cli_args",
        [
            [
                "preprocess", 
                "--input_path", "input_path", 
                "--output_path", "output_path"
            ]
        ],
    )
    def test_get_args(self, monkeypatch, cli_args):

        monkeypatch.setattr(sys, "argv", ["cli"] + cli_args)

        args = get_args()

        # Assert positional argument
        assert args.task == "preprocess"

        # Assert input/output path
        assert args.input_path == "input_path"
        assert args.output_path == "output_path"

        # Assert default values
        assert args.save is False
        assert args.target_sr == 32000
        assert args.segment_length == 0.03
        assert args.min_fragment_len == 300
        assert args.vad_aggressiveness == 0
        assert args.use_fp16 is True
        assert args.transcript_model_size == "large"


    @pytest.mark.parametrize(
        "cli_args",
        [(["preprocess", "--input_path", "", "--transcript_model_size", "tiny"])],
    )
    def test_episode_preprocessing(self, monkeypatch, cli_args, sample_video_path):
        # Author - Wojciech Stachowiak

        cli_args = [
            "preprocess",
            "--input_path",
            sample_video_path,
            "--transcript_model_size",
            "tiny",
        ]

        monkeypatch.setattr(sys, "argv", ["prog_name"] + cli_args)

        args = get_args()

        data_df = episode_preprocessing(args)

        assert isinstance(data_df, pd.DataFrame)
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
        assert all(
            data_df.columns
            == [
                "episode",
                "segment",
                "segment_start_seconds",
                "segment_end_seconds",
                "sentence",
            ]
        )

    """
    @pytest.mark.parametrize(
        "cli_args", [(["train", "--train_data", "tests/test_data/dataset_test.csv", 
                    "--model_path", "", "--val_data", "tests/test_data/dataset_test.csv"])]
    )
    def test_model_training(self, monkeypatch, cli_args):
        # Author - Wojciech Stachowiak

        monkeypatch.setattr(sys, "argv", ["prog_name"] + cli_args)

        args = get_args()

        model, tokenizer, label_encoder = model_training(args)

        assert model
        assert tokenizer
        assert label_encoder

    
    @pytest.mark.parametrize(
        "cli_args",
        [(["predict", "--input_path", "", "--model_path", ""])],
    )
    def test_predict(self, monkeypatch, cli_args, sample_test_data):
        monkeypatch.setattr(sys, "argv", ["prog_name"] + cli_args)

        args = get_args()
        emotions, probabilities = predict(args, sample_test_data)
        assert emotions
        assert probabilities

        assert isinstance(emotions, list), "Emotions should be a list"
        assert isinstance(probabilities, list), "Probabilities should be a list"

        assert all(isinstance(emotion, str) for emotion in emotions)
        assert all(isinstance(prob, float) for prob in probabilities)
    """

if __name__ == "__main__":
    pytest.main()