import argparse
import logging
import os

import joblib
import mltable
import tensorflow as tf
import transformers
from azure.core import MLClient
from azure.identity import ClientSecretCredential

# from preprocessing import preprocessing_function

mt_logger = logging.getLogger("main.model_training")


# TODO find a different way to get ml_client object (tenant id not available)
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "NLP3"
TENANT_ID = ""
CLIENT_ID = ""
CLIENT_SECRET = ""


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset_name",
        type=str,
        help="name of registered dataset used to train the model",
    )

    parser.add_argument(
        "--dataset_version", type=int, help="version of the chosen dataset"
    )

    parser.add_argument(
        "--val_dataset_name",
        type=str,
        help="""
        name of registered dataset used as validation data during model training
        """,
    )

    parser.add_argument("--epochs", type=int, help="number of training epochs ")

    parser.add_argument("--learning_rate", type=float, help="optimizer's learning rate")

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="patience of the early stopping callback",
    )

    args = parser.parse_args()

    return args


def get_ml_client(tenant_id, client_id, client_secret, subscription_id, resource_group):
    credential = ClientSecretCredential(
        tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
    )

    ml_client = MLClient(
        subscription_id=subscription_id,
        resource_group=resource_group,
        credential=credential,
    )

    return ml_client


def get_data_asset_as_df(ml_client, dataset_name, dataset_version):
    data_asset = ml_client.data.get(dataset_name, version=dataset_version)

    path = {"file": data_asset.path}

    table = mltable.from_delimited_files(paths=[path])
    df = table.to_pandas_dataframe()
    return df


def get_model(
    model_path: str, num_classes: int = 0
) -> tuple[transformers.TFRobertaForSequenceClassification, dict[int, str]]:
    """
    Create or load a RoBERTa model with the specified number of output classes.
    Number of classes not needed when loading a previously trained model.

    Input:
        model_path (str): Path to the model directory.
        num_classes (int): Number of output classes for the model. default: 0
            (handled outside this function using emotion_decoder
            from load_data function)

    Output:
        model: RoBERTa model with the specified number of output classes.
        emotion_dict (dict[int, str]): python dictionary that maps integers to text
            emotions for the model, only returned when loading a trained model,
            otherwise the value is None

    Author:
        Max Meiners (214936)
    """

    # return new instance of the model
    if model_path == "new":
        model_configuration = transformers.RobertaConfig.from_pretrained(
            "roberta-base", num_labels=num_classes
        )
        model = transformers.TFRobertaForSequenceClassification.from_pretrained(
            "roberta-base", config=model_configuration
        )

        return model, None

    # get config and model file paths
    config_path = os.path.join(model_path, "config.json")
    weights_path = os.path.join(model_path, "tf_model.h5")
    dict_path = os.path.join(model_path, "emotion_dict.joblib")

    mt_logger.info(f"loading model configuration")

    # load an existing model
    config = transformers.RobertaConfig.from_pretrained(config_path)
    model = transformers.TFRobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=config
    )

    mt_logger.info("loading model weights")
    model.load_weights(weights_path)

    mt_logger.info("model loaded")

    emotion_dict = joblib.load(dict_path)

    return model, emotion_dict


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


def main(args):
    """
    ml_client = get_ml_client(
        TENANT_ID, CLIENT_ID, CLIENT_SECRET, SUBSCRIPTION_ID, RESOURCE_GROUP
    )
    train_data = get_data_asset_as_df(
        ml_client, args.train_dataset_name, args.dataset_version
    )
    val_data = get_data_asset_as_df(
        ml_client, args.val_dataset_name, args.dataset_version
    )

    train_tf_data = preprocessing_function(train_data)
    val_tf_data = preprocessing_function(val_data)
    model, _ = get_model("new")
    model = train_model(
        model,
        train_tf_data,
        val_tf_data,
        args.epochs,
        args.learning_rate,
        args.early_stopping_patience,
    )
    """


if __name__ == "__main__":
    args = get_args()

    main(args)
