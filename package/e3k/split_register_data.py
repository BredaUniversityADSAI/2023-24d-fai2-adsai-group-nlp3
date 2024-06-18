import argparse
import logging
import os
from typing import Dict, Tuple

# import azureml
# import fsspec
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from sklearn.model_selection import train_test_split

# setting up logger
split_logger = logging.getLogger(f"{'main.' if __name__ != '__main__' else ''}{__name__}")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("logs.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

split_logger.addHandler(file_handler)

if len(split_logger.handlers) == 0:
    split_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    split_logger.addHandler(stream_handler)


subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "NLP3"
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
client_secret = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"


def connect_to_azure_ml(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    tenant_id: str,
    client_id: str,
    client_secret: str,
) -> Tuple[MLClient, Workspace]:
    """
    Connects to the Azure Machine Learning workspace using client secret credentials.

    Args:
    - subscription_id (str): The Azure subscription ID.
    - resource_group (str): The name of the resource group.
    - workspace_name (str): The name of the Azure Machine Learning workspace.
    - tenant_id (str): The Azure AD tenant ID.
    - client_id (str): The Azure AD client ID.
    - client_secret (str): The Azure AD client secret.

    Returns:
    - ml_client (MLClient): The Azure Machine Learning client object.
    - workspace (Workspace): The Azure Machine Learning workspace object.

    Author - Panna Pfandler
    """
    split_logger.info("Connecting to Azure ML workspace.")

    # Use client secret credentials
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)

    # Connect to the Azure Machine Learning client
    ml_client = MLClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        credential=credential,
        workspace_name=workspace_name,
    )

    # Connect to the Azure Machine Learning workspace
    service_principal = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=client_id,
        service_principal_password=client_secret,
    )
    workspace = Workspace(
        subscription_id, resource_group, workspace_name, auth=service_principal
    )

    split_logger.info("Connected to Azure ML workspace.")

    return ml_client, workspace


def load_data(file_path: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Load the dataset from a CSV file and return the DataFrame and
    a dictionary with labels.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - df (pd.DataFrame): The DataFrame containing the data.
    - emotion_decoder (Dict[int, str]): A dictionary mapping each unique emotion
    to a numerical label.

    Author - Max Meiners
    """
    split_logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, header=0)[["sentence", "emotion"]].dropna()

    # Create a dictionary with class labels
    emotion_decoder = {i: label for i, label in enumerate(df["emotion"].unique())}

    # Log message indicating data loaded
    split_logger.info(f"Loaded data: {os.path.basename(file_path)}")

    return df, emotion_decoder


def get_train_val_data(
    data_df: pd.DataFrame, val_size: float = 0.2
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """
    Split the data into training and validation sets.

    Args:
    - data_df (pd.DataFrame): The DataFrame containing the data.
    - val_size (float): The proportion of the data to include in the validation split.

    Returns:
    - train_set (tuple[pd.DataFrame, pd.Series]): The training set
    containing the features and labels.
    - val_set (tuple[pd.DataFrame, pd.Series]): The validation set
    containing the features and labels.

    Author - Panna Pfandler
    """
    split_logger.info("Splitting data into train and validation")

    X_train, X_val, y_train, y_val = train_test_split(
        data_df["sentence"],
        data_df["emotion"],
        test_size=val_size,
        random_state=42,
        stratify=data_df["emotion"],
    )

    local = args.local == "True"
    if local:
        train_set = (X_train, y_train)
        val_set = (X_val, y_val)
    else:
        train_set = pd.DataFrame({"sentence": X_train, "emotion": y_train})
        val_set = pd.DataFrame({"sentence": X_val, "emotion": y_val})

    split_logger.info("Data split")

    return train_set, val_set


def main(args: argparse.Namespace):
    """
    Main function to process data either locally or from Azure,
    and split it into training and validation sets.

    Args:
    - args (argparse.Namespace): The command-line arguments.

    Author - Panna Pfandler
    """
    split_logger.info("Starting main function")
    local = args.local == "True"

    split_logger.info(f"Local mode: {local}")
    split_logger.info(f"Data path: {args.data_path}")
    split_logger.info(f"Validation size: {args.val_size}")

    if local:
        split_logger.info("Processing data locally.")
        # Load data locally
        data_df, _ = load_data(args.data_path)
        train_set, val_set = get_train_val_data(data_df, args.val_size)
    else:
        split_logger.info("Processing data from Azure.")
        # Access data from Azure
        ml_client, workspace = connect_to_azure_ml(
            subscription_id,
            resource_group,
            workspace_name,
            tenant_id,
            client_id,
            client_secret,
        )
        datastore = Datastore.get(workspace, datastore_name="workspaceblobstore")

        # split_logger.info(f"Reading data from Azure Blob Storage URI: {uri}")
        # Read data using pandas

        data_df = pd.read_csv(
            (
                f"azureml://subscriptions/{subscription_id}/resourcegroups/"
                f"{resource_group}/workspaces/{workspace_name}/datastores/"
                f"workspaceblobstore/paths/{args.data_path}"
            )
        )

        # Split data
        train_set, val_set = get_train_val_data(data_df, args.val_size)

        # Register datasets
        # split_logger.info("Registering training dataset in Azure")
        Dataset.Tabular.register_pandas_dataframe(
            dataframe=train_set,
            name="train_data",
            description="training data",
            target=datastore,
        )
        # split_logger.info("Registering validation dataset in Azure")
        Dataset.Tabular.register_pandas_dataframe(
            dataframe=val_set,
            name="val_data",
            description="validation data",
            target=datastore,
        )

        split_logger.info("Data processed and datasets registered in Azure.")

    split_logger.info("Main function completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument(
        "--local",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Load data locally.",
    )
    parser.add_argument("--data_path", type=str, help="Path to the data file.")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    args = parser.parse_args()

    main(args)
