import os
import sys

import pandas as pd
import pytest
from e3k.cli import (episode_preprocessing, evaluate_model, get_args,
                     model_training, predict)


# model_output_information

@pytest.mark.parametrize(
    "cli_args",
    [(["add", "--input_path", "input_path", "--output_path", "output_path"])],
)
def test_get_args(monkeypatch, cli_args):
    # Author - Wojciech Stachowiak

    monkeypatch.setattr(sys, "argv", ["prog_name"] + cli_args)

    args = get_args()

    # assert positional argument
    assert args.task == "add"

    # assert input/output path
    assert args.input_path == "input_path"
    assert args.output_path == "output_path"

    # assert default values
    assert args.save is False
    assert args.target_sr == 32_000
    assert args.segment_length == 0.03
    assert args.min_fragment_len == 300
    assert args.vad_aggressiveness == 0
    assert args.use_fp16 is True
    assert args.transcript_model_size == "large"


@pytest.mark.parametrize(
    "cli_args",
    [(["preprocess", "--input_path", "", "--transcript_model_size", "tiny"])],
)
def test_episode_preprocessing(monkeypatch):
    # Author - Wojciech Stachowiak

    current_path = os.path.dirname(__file__)
    video_path = os.path.join(current_path, "../ep_1_10min.mp3")

    cli_args = [
        "preprocess",
        "--input_path",
        video_path,
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


@pytest.mark.parametrize(
    "cli_args", [(["train", "--input_path", "", "--model_path", ""])]
)
def test_model_training(monkeypatch, cli_args):
    # Author - Wojciech Stachowiak

    monkeypatch.setattr(sys, "argv", ["prog_name"] + cli_args)

    args = get_args()

    model, tokenizer, label_encoder = model_training(args)

    assert model
    assert tokenizer
    assert label_encoder


@pytest.mark.parametrize(
    "cli_args", [(["train", "--input_path", "", "--model_path", "", "eval_path", ""])]
)
def test_evaluate_model(monkeypatch, cli_args):
    # Author -Wojciech Stachowiak

    monkeypatch.setattr(sys, "argv", ["prog_name"] + cli_args)

    args = get_args()

    model, tokenizer, label_encoder = model_training(args)
    predicted_emotions, confidence_scores, total_accuracy, report = evaluate_model(
        args, model, tokenizer, label_encoder
    )

    assert len(predicted_emotions) > 0
    assert len(confidence_scores) > 0
    assert total_accuracy > 0
    assert report


@pytest.mark.parametrize(
    "cli_args",
    [(["predict", "--input_path", "input_path", "--output_path", "output_path"])],
)
def test_predict():
    predict()


def test_model_output_information():
    pass
