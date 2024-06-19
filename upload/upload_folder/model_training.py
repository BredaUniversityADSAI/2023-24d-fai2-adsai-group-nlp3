import argparse
import logging
import pickle

import mltable
import pandas as pd
import tensorflow as tf
import transformers
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from preprocessing import preprocess_training_data

# setting up logger
mt_logger = logging.getLogger("model_training")
mt_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("logs.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

mt_logger.addHandler(file_handler)
mt_logger.addHandler(stream_handler)


# const values for Azure connection
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "NLP3"
TENANT_ID = "0a33589b-0036-4fe8-a829-3ed0926af886"
CLIENT_ID = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
CLIENT_SECRET = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"


mt_logger.info(f"devices: {tf.config.list_physical_devices()}")


def get_args() -> argparse.Namespace:
    """
    A function that instantiates argument parser and collects CLI arguments.

    Input: None

    Output:
        args (argparse.Namespace): namespace object with CLI arguments

    Author - Wojciech Stachowiak
    """
    parser = argparse.ArgumentParser()

    mt_logger.debug("collecting CLI args")

    parser.add_argument(
        "--cloud",
        type=bool,
        choices=[True, False],
        help="whether the code will execute on Azure or locally",
    )

    parser.add_argument(
        "--train_dataset_name",
        type=str,
        help="name of registered dataset used to train the model",
    )

    parser.add_argument(
        "--val_dataset_name",
        type=str,
        help="""
        name of registered dataset used as validation data during model training
        """,
    )

    parser.add_argument(
        "--dataset_version", type=str, help="version of the chosen dataset"
    )

    parser.add_argument("--epochs", type=int, help="number of training epochs ")

    parser.add_argument("--learning_rate", type=float, help="optimizer's learning rate")

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="patience of the early stopping callback",
    )

    parser.add_argument(
        "--model_output_path",
        type=str,
        help="path where the trained model will be saved",
    )

    parser.add_argument(
        "--decoder_output_path",
        type=str,
        help="path where the label_decoder will be saved",
    )

    mt_logger.info("collected CLI args")

    mt_logger.debug("parsing args")

    args = parser.parse_args()

    mt_logger.debug("parsed args")

    return args


def get_ml_client(
    subscription_id: str,
    tenant_id: str,
    client_id: str,
    client_secret: str,
    resource_group: str,
    workspace_name: str,
) -> MLClient:
    """
    A function that creates an MLClient object used to interact with AzureML.

    Input:
        subscription_id (str): subscription ID from Azure
        tenant_id (str): tenant ID from Azure
        client_id (str): client ID from Azure
        client_secret (str): client secret from Azure
        resource_group (str): name of the resource group from Azure
        workspace_name (str): name of the workspace name from Azure

    Output:
        ml_client (azure.ai.ml.MLClient): an object that allows manipulating
        objects on Azure

    Author - Wojciech Stachowiak
    """
    mt_logger.debug("getting credentials")
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)

    mt_logger.debug("building MLClient")

    ml_client = MLClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        credential=credential,
        workspace_name=workspace_name,
    )

    mt_logger.info("got MLClient")

    return ml_client


def get_data_asset_as_df(
    ml_client: MLClient, dataset_name: str, dataset_version: str
) -> pd.DataFrame:
    """
    A function that loads Azure dataset and converts it to pandas.DataFrame.

    Input:
        ml_client (azure.ai.ml.MLClient): MLClient used to interact with AzureML
        dataset_name (str): name of the dataset registered on Azure
        dataset_version (str): version of the dataset

    Output:
        df (pd.DataFrame): dataframe created from the Azure dataset

    Author - Wojciech Stachowiak
    """

    # fetching dataset
    mt_logger.debug(f"getting dataset: {dataset_name} version {dataset_version}")
    data_asset = ml_client.data.get(dataset_name, version=dataset_version)
    mt_logger.debug("got dataset")

    path = {"folder": data_asset.path}

    # loading data as mltable
    mt_logger.debug("reading into table")
    table = mltable.from_delimited_files(paths=[path])
    mt_logger.debug("table loaded")

    # converting to pd.DataFrame
    mt_logger.debug("getting dataframe")
    df = table.to_pandas_dataframe()
    mt_logger.debug("got dataframe")

    return df


def get_label_decoder(series: pd.Series) -> dict[int, str]:
    """
    A function that creates label_decoder dictionary from pandas.Series.

    Input:
        series (pd.Series): series with labels

    Output:
        label_decoder (dict[int, str]): dictionary mapping number values
        to text representation

    Author - Wojciech Stachowiak
    """
    mt_logger.info("getting label decoder from training data")
    label_decoder = {i: label for i, label in enumerate(series.unique())}
    mt_logger.debug(f"detected {len(label_decoder)} classes")

    return label_decoder


def get_new_model(num_classes: int) -> transformers.TFRobertaForSequenceClassification:
    """
    Create a RoBERTa model with the specified number of output classes.

    Input:
        model_path (str): Path to the model directory.
        num_classes (int): Number of output classes for the model.

    Output:
        model: RoBERTa model with the specified number of output classes.

    Author:
        Max Meiners (214936)
    """

    mt_logger.debug("loading model")
    model_configuration = transformers.RobertaConfig.from_pretrained(
        "roberta-base", num_labels=num_classes
    )
    model = transformers.TFRobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=model_configuration
    )
    mt_logger.info("loaded model")

    return model


def train_model(
    model: transformers.TFRobertaForSequenceClassification,
    training_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    epochs: int = 3,
    learning_rate: float = 1e-5,
    early_stopping_patience: int = 3,
) -> transformers.TFRobertaForSequenceClassification:
    """
    Train the model using tensorflow datasets.

    Input:
        model: Model to be trained.
        training_dataset: Dataset used for training the model
        validation_dataset: Dataset used for validating the model
        epochs (int): Number of epochs to train the model. default: 3
        learning_rate (float): optimizer's learning rate. default: 1e-5
        early_stopping_patience (int): patience parameter for EarlyStopping callback

    Output:
        model: trained model

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("compiling model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        verbose=1,
        restore_best_weights=True,
    )

    mt_logger.info("training model")

    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    mt_logger.info("training finished")

    return model


def main(args: argparse.Namespace) -> None:
    """
    An aggregate function for the model_training module.
    Intended to be used as an Azure component only. It dumps both the trained model
    and label decoder to files that can be passed in component declaration.

    Input:
        args (argparse.Namespace): namespace object with CLI arguments

    Output: None
        model: saved in a folder under the specified path
        label_decoder: dumped to a file under the specified path using json.dump
    """

    ml_client = get_ml_client(
        SUBSCRIPTION_ID,
        TENANT_ID,
        CLIENT_ID,
        CLIENT_SECRET,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )

    # getting datasets
    train_data = get_data_asset_as_df(
        ml_client, args.train_dataset_name, args.dataset_version
    )
    val_data = get_data_asset_as_df(
        ml_client, args.val_dataset_name, args.dataset_version
    )

    label_decoder = get_label_decoder(train_data["emotion"])

    train_tf_data, val_tf_data = preprocess_training_data(
        train_data, val_data, label_decoder
    )

    model = get_new_model(num_classes=len(label_decoder))
    model = train_model(
        model,
        train_tf_data,
        val_tf_data,
        args.epochs,
        args.learning_rate,
        args.early_stopping_patience,
    )

    model.save_pretrained(args.model_output_path)
    with open(args.decoder_output_path, "wb") as f:
        pickle.dump(label_decoder, f)


if __name__ == "__main__":
    args = get_args()
    main(args)
