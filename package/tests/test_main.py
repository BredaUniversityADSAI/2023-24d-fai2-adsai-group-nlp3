import sys
import os
import pytest
import pandas as pd
from src.main import get_args
from src.main import episode_preprocessing


@pytest.mark.parametrize(
        "cli_args",
        [(["add", "--input_path", "input_path", "--output_path", "output_path"])]
)
def test_get_args(monkeypatch, cli_args):
    # author - Wojciech Stachowiak

    monkeypatch.setattr(sys, "argv", ["prog_name"] + cli_args)

    args = get_args()


    # assert positional argument
    assert args.task == "add"

    # assert input/output path
    assert args.input_path == "input_path"
    assert args.output_path == "output_path"

    # assert defaul values
    assert args.save == False
    assert args.target_sr == 32_000
    assert args.segment_length == 0.03
    assert args.min_fragment_len == 300
    assert args.vad_aggressiveness == 0
    assert args.use_fp16 == True
    assert args.transcript_model_size == "large"


def test_episode_preprocessing(monkeypatch):
    # author - Wojciech Stachowiak

    current_path = os.path.dirname(__file__)
    video_path = os.path.join(current_path, "../ep_1_10min.mp3")

    cli_args = [
        "preprocess",
        "--input_path", video_path,
        "--transcript_model_size", "tiny"
    ]

    monkeypatch.setattr(sys, "argv", ["prog_name"] + cli_args)

    args = get_args()

    data_df = episode_preprocessing(args)

    assert isinstance(data_df, pd.DataFrame)
    assert data_df.shape[0] > 0
    assert data_df.shape[1] > 0
    assert all(data_df.columns == [
        "episode",
        "segment",
        "segment_start_seconds",
        "segment_end_seconds",
        "sentence",
    ])