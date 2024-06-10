from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
import os
from typing import List
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
moi_logger = logging.getLogger("main.upload_data")

subscription_id = '0a94de80-6d3b-49f2-b3e9-ec5818862801'
resource_group = 'buas-y2'
workspace_name = 'NLP3'

auth = InteractiveLoginAuthentication()

moi_logger.info("Connecting to Azure Machine Learning Workspace")
workspace = Workspace(subscription_id=subscription_id,
                      resource_group=resource_group,
                      workspace_name=workspace_name,
                      auth=auth)
moi_logger.info("Connected successfully to workspace.")

# Get the default datastore from the Azure Machine Learning workspace
default_datastore = workspace.get_default_datastore()

def upload_files_to_azure(file_paths: List[str], default_datastore) -> None:
    """
    Uploads files from specified file paths to target paths based on their extensions.
    
    Parameters:
    file_paths (List[str]): List of file paths to upload.
    default_datastore: The default datastore object that handles the file upload.
    """
    # Iterate over the provided file paths
    for file_path in file_paths:
        # Check if the current path is a file
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            # Determine the target path based on the file extension
            if file_name.endswith('.mp3') or file_name.endswith('.mov'):
                target_path = 'episode/' + file_name
            elif file_name.endswith('.csv'):
                target_path = 'text_dataset/' + file_name
            else:
                moi_logger.warning(f"Skipping unsupported file type: {file_name}")
                continue
            
            # Upload the file to the determined target path
            default_datastore.upload_files(files=[file_path],
                                           target_path=target_path,
                                           overwrite=True,
                                           show_progress=True)
            moi_logger.info(f"Uploaded {file_name} to {target_path}")
        else:
            moi_logger.error(f"File not found: {file_path}")

file_paths = ['dataset_for_test.csv']
upload_files_to_azure(file_paths, default_datastore)
