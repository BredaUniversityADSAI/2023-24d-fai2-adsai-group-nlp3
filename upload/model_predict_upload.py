import os

from azure.ai.ml import Input, MLClient, Output, command
from azure.identity import InteractiveBrowserCredential
from azureml.core import Workspace

# Define the workspace
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "NLP3"

# Load the workspace
ws = Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
)

# Authenticate and create an MLClient
credential = InteractiveBrowserCredential()

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

environment = ml_client.environments.get(name="BlockD", version="2")

# Define the compute target
compute_name = "adsai0"
try:
    compute_target = ml_client.compute.get(compute_name)
    print("Found existing compute target.")
except Exception:
    print("Creating new compute target.")

# Define the prediction component
predict_component = command(
    name="prediction_component",
    display_name="Model prediction",
    description="Predict emotions from input data using a pre-trained RoBERTa model",
    inputs={
        "model_path": Input(
            type="uri_folder",
            description="Path to the model configuration and weights file",
        ),
        "data_path": Input(type="uri_file", description="Data to be predicted"),
        "tokenizer_model": Input(
            type="string",
            description="Model to use for tokenization",
            default="roberta-base",
        ),
        "max_length": Input(
            type="integer",
            description="Maximum length for tokenized sequences",
            default=128,
        ),
        "decoder_path": Input(
            type="uri_file",
            description="Path to the joblib file containing the emotion decoder",
        ),
    },
    outputs={
        "predictions": Output(
            type="uri_file",
            mode="upload",
            description="Output predictions with confidence scores",
        )
    },
    code=os.path.abspath("./package/e3k/temp_folder"),
    command="""
    python model_predict.py
    --model_path ${{inputs.model_path}}
    --data_path ${{inputs.data_path}}
    --tokenizer_model ${{inputs.tokenizer_model}}
    --max_length ${{inputs.max_length}}
    --decoder_path ${{inputs.decoder_path}}
    --output_path ${{outputs.predictions}}
    """,
    environment=environment,
    compute=compute_name,
)

# Create or update the prediction component
ml_client.create_or_update(predict_component.component)
