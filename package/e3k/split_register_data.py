import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
import logging
import os
from azureml.core import Dataset, Datastore, Workspace
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
mt_logger = logging.getLogger('split_register_data')

subscription_id = '0a94de80-6d3b-49f2-b3e9-ec5818862801'
resource_group = 'buas-y2'
workspace_name = 'NLP3'
tenant_id = '0a33589b-0036-4fe8-a829-3ed0926af886'
client_id = '27157a5a-3927-4895-8478-9d4554697d25'
client_secret = 'stf8Q~mP2cB923Mvz5K91ITcoYgvRXs4J1lysbfb'

def connect_to_azure_ml(subscription_id: str, resource_group: str, workspace_name: str, tenant_id: str, client_id: str, client_secret: str) -> MLClient:
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

    Author - Panna Pfandler
    """
    mt_logger.info("Connecting to Azure ML workspace.")
    
    # Use client secret credentials
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    
    # Connect to the Azure Machine Learning workspace
    ml_client = MLClient(subscription_id=subscription_id,
        resource_group_name=resource_group,
        credential=credential,
        workspace_name=workspace_name)
    
    mt_logger.info("Connected to Azure ML workspace.")
    
    return ml_client


def load_data(file_path: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Load the dataset from a CSV file and return the DataFrame and a dictionary with labels.
    
    Args:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - df (pd.DataFrame): The DataFrame containing the data.
    - emotion_decoder (Dict[int, str]): A dictionary mapping each unique emotion
    to a numerical label.

    Author - Max Meiners
    """
    mt_logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)[["sentence", "emotion"]].dropna()

    # Create a dictionary with class labels
    emotion_decoder = {i: label for i, label in enumerate(df["emotion"].unique())}

    # Log message indicating data loaded
    mt_logger.info(f"Loaded data: {os.path.basename(file_path)}")

    return df, emotion_decoder

def get_train_val_data(data_df: pd.DataFrame, val_size: float = 0.2) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """
    Split the data into training and validation sets.
    
    Args:
    - data_df (pd.DataFrame): The DataFrame containing the data.
    - val_size (float): The proportion of the data to include in the validation split.
    
    Returns:
    - train_set (tuple[pd.DataFrame, pd.Series]): The training set containing the features and labels.
    - val_set (tuple[pd.DataFrame, pd.Series]): The validation set containing the features and labels.

    Author - Panna Pfandler
    """
    mt_logger.info("Splitting data into train and validation")

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
        train_set = pd.DataFrame({'sentence': X_train, 'emotion': y_train})
        val_set = pd.DataFrame({'sentence': X_val, 'emotion': y_val})

    mt_logger.info("Data split")

    return train_set, val_set

def main(args: argparse.Namespace):
    """
    Main function to process data either locally or from Azure, and split it into training and validation sets.
    
    Args:
    - args (argparse.Namespace): The command-line arguments.

    Author - Panna Pfandler
    """
    local = args.local == "True"
    if local:
        mt_logger.info("Processing data locally.")
        # Load data locally
        data_df, _ = load_data(args.data_path)
        train_set, val_set = get_train_val_data(data_df)
    else:
        mt_logger.info("Processing data from Azure.")
        # Access data from Azure
        ml_client = connect_to_azure_ml(subscription_id, resource_group, workspace_name, tenant_id, client_id, client_secret)
        workspace = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)
        datastore = Datastore.get(workspace, datastore_name='workspaceblobstore')
        uri = f'azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/workspaceblobstore/paths/{args.data_path}'
        
        # Read data using pandas
        data_df = pd.read_csv(uri)

        # Split data
        train_set, val_set = get_train_val_data(data_df)

        # Register datasets
        Dataset.Tabular.register_pandas_dataframe(dataframe=train_set, name='train_data', description='training data', target=datastore)
        Dataset.Tabular.register_pandas_dataframe(dataframe=val_set, name='val_data', description='validation data',target=datastore)
        
        mt_logger.info("Data processed and datasets registered in Azure.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument("--local", type=str, default="True", choices=["True", "False"], help="Load data locally.")
    parser.add_argument("--data_path", type=str, help="Path to the data file.")
    args = parser.parse_args()

    main(args)
