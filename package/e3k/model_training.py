import argparse
import json
import logging
import pickle
from functools import reduce
from typing import Dict, Tuple

import config
import matplotlib.pyplot as plt
import mlflow
import mltable
import pandas as pd
import tensorflow as tf
import transformers
import typeguard
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from preprocessing import preprocess_training_data

# setting up logger
mt_logger = logging.getLogger(f"{'main.' if __name__ != '__main__' else ''}{__name__}")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

if len(mt_logger.handlers) == 0:
    mt_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    mt_logger.addHandler(stream_handler)

file_handler = logging.FileHandler("logs.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

mt_logger.addHandler(file_handler)


@typeguard.typechecked
def get_args() -> argparse.Namespace:
    """
    A function that instantiates argument parser and collects CLI arguments.

    Input: None

    Output:
        args (argparse.Namespace): namespace object with CLI arguments

    Author - Wojciech Stachowiak
    """
    parser = argparse.ArgumentParser()

    mt_logger.debug("Collecting CLI args...")

    parser.add_argument(
        "--cloud",
        type=bool,
        choices=[True, False],
        help="whether the code will execute on Azure or locally",
    )

    parser.add_argument(
        "--dataset_name_file",
        type=str,
        help="name of registered dataset used to train the model",
    )

    parser.add_argument("--epochs", type=int, help="number of training epochs ")

    parser.add_argument("--learning_rate", type=float, help="optimizer's learning rate")

    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for training the model",
    )

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

    mt_logger.info("Collected CLI args.")

    mt_logger.debug("Parsing args...")

    args = parser.parse_args()

    mt_logger.debug("Args have been parsed.")

    return args


@typeguard.typechecked
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
    mt_logger.debug("Getting credential...")
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)

    mt_logger.debug("Building MLClient...")

    ml_client = MLClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        credential=credential,
        workspace_name=workspace_name,
    )

    mt_logger.info("Got MLClient!")

    return ml_client


@typeguard.typechecked
def get_versioned_datasets(
    args: argparse.Namespace, 
    ml_client: MLClient
    ) -> Tuple[str, str, str]:
    """
    A function that finds the newest versions of train and validation datasets,
    and returns those values.

    Input:
        args (argparse.Namespace): Namespace object with CLI arguments
        ml_client (azure.ai.ml.MLClient): Azure object used to interact with Azure

    Output:
        train_name (str): name of the train dataset
        val_name (str): name of the validation dataset
        newest_version (str): newest version of the dataset

    Author - Wojciech Stachowiak
    """
    mt_logger.debug("Getting datasets names...")
    with open(args.dataset_name_file) as f:
        dataset_info = json.load(f)

    train_name = dataset_info["train_data"]
    val_name = dataset_info["val_data"]

    mt_logger.info("Got datasets names. Getting dataset list for versioning...")

    # get list of datasets with this name
    dataset_list = ml_client.data.list(name=train_name)

    mt_logger.debug("Got dataset list for versioning.")

    # get highest version frm the list
    newest_version = reduce(
        lambda x, y: max(x, y), map(lambda x: x.version, dataset_list)
    )

    mt_logger.debug(f"Got datasets version {newest_version}.")

    return train_name, val_name, newest_version


@typeguard.typechecked
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
    mt_logger.debug(
        f"Getting dataset: {dataset_name}, with version {dataset_version}..."
    )
    data_asset = ml_client.data.get(dataset_name, version=dataset_version)
    mt_logger.debug("Got dataset.")

    path = {"folder": data_asset.path}

    # loading data as mltable
    mt_logger.debug("Reading table...")
    table = mltable.from_parquet_files(paths=[path])
    mt_logger.debug("Table has been loaded.")

    # converting to pd.DataFrame
    mt_logger.debug("Getting DataFrame...")
    df = table.to_pandas_dataframe()
    mt_logger.debug("Got the DataFrame.")

    return df


@typeguard.typechecked
def get_label_decoder(series: pd.Series) -> Dict[int, str]:
    """
    A function that creates label_decoder dictionary from pandas.Series.

    Input:
        series (pd.Series): series with labels

    Output:
        label_decoder (dict[int, str]): dictionary mapping number values
        to text representation

    Author - Wojciech Stachowiak
    """
    mt_logger.info("Getting the label decoder from the training data...")
    label_decoder = {i: label for i, label in enumerate(series.unique())}
    mt_logger.debug(f"{len(label_decoder)} classes have been detected.")

    return label_decoder


@typeguard.typechecked
def get_new_model(num_classes: int) -> transformers.TFRobertaForSequenceClassification:
    """
    Create or load a RoBERTa model with the specified number of output classes.

    Input:
        num_classes (int): Number of output classes for the model.

    Output:
        model: RoBERTa model with the specified number of output classes.

    Author:
        Max Meiners (214936)
    """

    mt_logger.debug("Loading the model...")
    model_configuration = transformers.RobertaConfig.from_pretrained(
        "roberta-base", num_labels=num_classes
    )
    model = transformers.TFRobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=model_configuration
    )
    mt_logger.info("The model has been loaded!")

    return model


@typeguard.typechecked
def train_model(
    model: transformers.TFRobertaForSequenceClassification,
    training_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
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
        model: A trained model with the specified parameters.

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("Compiling the model...")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    mt_logger.info("Model has been compiled.")

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        verbose=1,
        restore_best_weights=True,
    )

    mt_logger.info("Training the model...")

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    # Log model summary to MLflow
    mlflow.log_text("model_summary.txt", str(model.summary()))

    # Log model metrics to MLflow
    mlflow.log_metric("train_loss", history.history["loss"][-1])
    mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

    # Plot training and validation loss to MLflow
    fig = plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    mlflow.log_figure(fig, "model_metrics.png")

    mt_logger.info("Training has been completed!")

    return model


@typeguard.typechecked
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

    # Start MLflow run
    mlflow.start_run()
    # mlflow.tensorflow.autolog()

    # Log parameters to MLflow
    mlflow.log_params(
        {
            "cloud": args.cloud,
            "dataset_name_file": args.dataset_name_file,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "early_stopping_patience": args.early_stopping_patience,
            "model_output_path": args.model_output_path,
            "decoder_output_path": args.decoder_output_path,
        }
    )

    ml_client = get_ml_client(
        config.config["subscription_id"],
        config.config["tenant_id"],
        config.config["client_id"],
        config.config["client_secret"],
        config.config["resource_group"],
        config.config["workspace_name"],
    )

    train_name, val_name, version = get_versioned_datasets(args, ml_client)

    # getting datasets
    train_data = get_data_asset_as_df(ml_client, train_name, version)
    val_data = get_data_asset_as_df(ml_client, val_name, version)

    label_decoder = get_label_decoder(train_data["emotion"])

    train_tf_data, val_tf_data = preprocess_training_data(
        train_data, val_data, label_decoder
    )

    model = get_new_model(num_classes=len(label_decoder))
    model = train_model(
        model,
        train_tf_data,
        val_tf_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
    )

    model.save_pretrained(args.model_output_path)
    with open(args.decoder_output_path, "wb") as f:
        pickle.dump(label_decoder, f)

    # Log output paths to MLflow
    mlflow.log_artifact(args.model_output_path, "model")
    mlflow.log_artifact(args.decoder_output_path, "label_decoder")

    # End MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    args = get_args()
    main(args)
