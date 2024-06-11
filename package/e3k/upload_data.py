from azureml.core import Workspace, Datastore
from azureml.core.authentication import InteractiveLoginAuthentication
import os
from typing import List
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
moi_logger = logging.getLogger("main.upload_data")

# Azure subscription and workspace details
subscription_id = '0a94de80-6d3b-49f2-b3e9-ec5818862801'
resource_group = 'buas-y2'
workspace_name = 'NLP3'

# Authentication setup
auth = InteractiveLoginAuthentication()

moi_logger.info("Connecting to Azure Machine Learning Workspace")
workspace = Workspace(subscription_id=subscription_id,
                      resource_group=resource_group,
                      workspace_name=workspace_name,
                      auth=auth)
moi_logger.info("Connected successfully to workspace.")

# Get the default datastore from the Azure Machine Learning workspace
default_datastore = workspace.get_default_datastore()

def upload_files_to_azure(file_paths: List[str], default_datastore: Datastore) -> None:
    """
    Uploads files from specified file paths to target paths in the Azure datastore based on their extensions.
    
    This function iterates over the list of provided file paths and uploads each file to a target path in the 
    Azure datastore. The target path is determined based on the file extension:
    
    - Files with '.mp3' or '.mov' extensions are uploaded to the 'episode/' directory.
    - Files with '.csv' extensions are uploaded to the 'text_dataset/' directory.
    - Files with unsupported extensions are skipped with a warning logged.
    
    Parameters:
    file_paths (List[str]): List of file paths to upload.
    default_datastore (Datastore): The default datastore object that handles the file upload.

    Author - Panna Pfandler
    """
    for file_path in file_paths:
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            
            if file_name.endswith('.mp3') or file_name.endswith('.mov'):
                target_path = 'episode/' 
            elif file_name.endswith('.csv'):
                target_path = 'text_dataset/' 
            else:
                moi_logger.warning(f"Skipping unsupported file type: {file_name}")
                continue
            
            default_datastore.upload_files(files=[file_path],
                                           target_path=target_path,
                                           overwrite=True,
                                           show_progress=True)
            moi_logger.info(f"Uploaded {file_name} to {target_path}")
        else:
            moi_logger.error(f"File not found: {file_path}")

# List of file paths to upload
file_paths = ['dataset_for_test.csv']
upload_files_to_azure(file_paths, default_datastore)
