import sys
import pandas as pd
import pytest
from e3k.cli import (episode_preprocessing, get_args, model_output_information,
                     model_training, predict)


# Fixtures to provide sample paths and data
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
        """
        Test parsing command line arguments using the `cli.get_args`.
        Asserts: The parsed arguments should match the expected values.

        Author - Wojciech Stachowiak
        """

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
        """
        Test the `episode_preprocessing` function to ensure it correctly processes
        the input video and generates a DataFrame.
        Asserts:
            The result should be a DataFrame.
            The DataFrame should not be empty.
            The DataFrame should have the expected columns.

        Author: Wojciech Stachowiak
        """

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


if __name__ == "__main__":
    pytest.main()