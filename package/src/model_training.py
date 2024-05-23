import argparse

# import numpy as np
import pandas as pd

# from transformers import (RobertaConfig, RobertaTokenizer,
#                           TFRobertaForSequenceClassification)


def load_data(
    file_path: str,
    dataset: str,
) -> tuple[pd.DataFrame, int]:
    """
    I figured this is a good way to return the number of classes,
    may be worth to go one step further and just return a dict with the classes
    May be also worth to have a class to keep this organized


    probably some way to chose train/evaluate data
    probably some way to filter training data
    (no idea how we are saving this, so TBD tomorrow I guess)

    file_path (string): file path to the datasets
    dataset (string): type of dataset to load (train or eval)
    """

    df = pd.read_csv(file_path)
    num_classes = 6

    return (df, num_classes)


def get_model():
    """
    create a model and return it
    look for a way to make this as generic as possible (num of classes is a good start)
    bonus: compatible with different model architectures
    """
    pass


def preprocess_data(sentence_df, tokenizer):
    """
    tokenize, and do whatever needed
    nice is generic, but we can deal with less generic probably
    """
    pass


def predict(model, sentences: pd.DataFrame):
    """
    Probably returns either pd.DataFrame or np.array
    Someone also has the predict func so check if it's done
    We don't need 2 of them
    """
    return model.predict(sentences)


def evaluate(eval_data: pd.DataFrame, model):
    # prepared_sentences = preprocess_data(eval_data["sentence"])
    # preds = predict(model, prepared_sentences)
    # get preds in a reasonable format

    # sklearn eval funcs should be good
    pass


if __name__ == "__main__":
    """
    logic for running this script from CLI
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hi", required=False, default="Lorem Ipsum", help="placeholder argument"
    )

    args = parser.parse_args()

    print(args.hi)
