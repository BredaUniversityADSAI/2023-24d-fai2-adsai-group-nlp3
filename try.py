import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from azureml.core import Dataset, Workspace
import logging
import os
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
import mltable
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

def connect_to_azure_ml(subscription_id: str, resource_group: str, workspace_name: str):
    """
    Connects to the Azure Machine Learning workspace using interactive login authentication.
    
    Args:
    - subscription_id (str): The Azure subscription ID.
    - resource_group (str): The name of the resource group.
    - workspace_name (str): The name of the Azure Machine Learning workspace.
    
    Returns:
    - workspace (Workspace): The Azure Machine Learning workspace object.
    - ml_client (AzureMLClient): The Azure Machine Learning client object.
    """
    # Use interactive login authentication
    auth = InteractiveLoginAuthentication()
    
    # Connect to the Azure Machine Learning workspace
    workspace = Workspace(subscription_id=subscription_id,
                          resource_group=resource_group,
                          workspace_name=workspace_name,
                          auth=auth)
    
    # Create a browser credential
    credential = InteractiveBrowserCredential()
    ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name)
    
    return workspace, ml_client

mt_logger = logging.getLogger("main.try")

def load_data(file_path: str) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Load the dataset from a CSV file and return the DataFrame and a dictionary with labels.
    
    Args:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - df (pd.DataFrame): The DataFrame containing the data.
    - emotion_decoder (dict[int, str]): A dictionary mapping each unique emotion to a numerical label.
    """
    df = pd.read_csv(file_path)[["sentence", "emotion"]].dropna()

    # Log message indicating data loaded
    mt_logger.info(f"loaded data: {os.path.basename(file_path)}")

    return df

def get_train_val_data(data_df: pd.DataFrame, val_size: float = 0.2) -> tuple[tuple[pd.DataFrame], tuple[pd.DataFrame]]:
    """
    Split the data into training and validation sets.
    
    Args:
    - data_df (pd.DataFrame): The DataFrame containing the data.
    - val_size (float): The proportion of the data to include in the validation split.
    
    Returns:
    - train_set (tuple[pd.DataFrame]): The training set containing the features and labels.
    - val_set (tuple[pd.DataFrame]): The validation set containing the features and labels.
    """
    mt_logger.info("splitting data into train and validation")

    X_train, X_val, y_train, y_val = train_test_split(
        data_df["sentence"],
        data_df["emotion"],
        test_size=val_size,
        random_state=42,
        stratify=data_df["emotion"],
    )

    train_set = (X_train, y_train)
    val_set = (X_val, y_val)

    mt_logger.info("data split")

    return train_set, val_set

def main(args: argparse.Namespace, subscription_id: str, resource_group: str, workspace_name: str):
    """
    Main function to process data either locally or from Azure, and split it into training and validation sets.
    
    Args:
    - args (argparse.Namespace): The command-line arguments.
    - subscription_id (str): The Azure subscription ID.
    - resource_group (str): The name of the resource group.
    - workspace_name (str): The name of the Azure Machine Learning workspace.
    """
    local = str(args.local) == "True"
    if local:
        # Load data locally
        data_df, _ = load_data(args.data_path)
        train_set, val_set = get_train_val_data(data_df)
    else:
        # Access data from Azure
        workspace, ml_client = connect_to_azure_ml(subscription_id, resource_group, workspace_name)
        datastore = ml_client.datastores.get("workspaceblobstore")

        dataset = Data(path=args.data_path, type=AssetTypes.URI_FOLDER, description="<ADD A DESCRIPTION HERE>", name="<NAME OF DATA ASSET>", version="1")
        
        # Load data as MLTable
        path = {"folder": dataset.path}
        table = mltable.from_delimited_files(paths=[path])

        # Convert table to DataFrame
        data_df = table.to_pandas_dataframe()

        # Split data
        train_set, val_set = get_train_val_data(data_df)

        # Register datasets
        train_reg = train_set.register(workspace=workspace, name='train_data', description='training data', create_new_version=True)
        val_reg = val_set.register(workspace=workspace, name='val_data', description='validation data', create_new_version=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument("--local", type=bool, default=True, choices=[True, False], help="Load data locally.")
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    subscription_id = '0a94de80-6d3b-49f2-b3e9-ec5818862801'
    resource_group = 'buas-y2'
    workspace_name = 'NLP3'

    main(args, subscription_id, resource_group, workspace_name)
