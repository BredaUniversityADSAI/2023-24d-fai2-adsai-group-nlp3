from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
import os

print("Starting the pipeline setup...")

# Define credentials and workspace details
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "NLP3"
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
client_secret = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

print("Credentials and workspace details defined.")

# Authenticate and create an MLClient
credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret
    )

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
    )

# Retrieve the model from the workspace
model_name = "RoBERTa_model"
model_version = "1"
model = ml_client.models.get(
    name=model_name, 
    version=model_version
    )
model_path = model.path

# Define paths for inputs and outputs
data_folder_path = (
    "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
    "resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/"
    "paths/dataset_max/emotions_all_azure_10rows.csv"
)
decoder_folder_path = (
    "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
    "resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/"
    "paths/test_evaluation_pipeline/test_label_encoder.json"
)
#     "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
#     "resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/"
#     "paths/labels_encodings/label_decoder"
# )

# Define the compute target and environment
compute_name = "adsai0"
environment_name = "BlockD"
environment_version = "2"

# Define the inputs and outputs for the job
job_inputs = {
    "model_path": Input(type=AssetTypes.URI_FOLDER, path=model_path, mode="download"),
    "data_path": Input(type=AssetTypes.URI_FILE, path=data_folder_path),
    "decoder_path": Input(type=AssetTypes.URI_FILE, path=decoder_folder_path),
    "tokenizer_model": Input(type="string", default="roberta-base"),
    "max_length": Input(type="integer", default=128),
    }

job_outputs = {
    "predictions": Output(type="uri_file")
    }

# Define the command component
predict_component = command(
    name="prediction_component",
    display_name="Model Prediction Component",
    description="Predict emotions from input data using a pre-trained RoBERTa model",
    inputs=job_inputs,
    outputs=job_outputs,
    code=os.path.abspath("./"),
    command=(
        'python model_predict.py '
        '--model_path ${{inputs.model_path}} '
        '--data_path ${{inputs.data_path}} '
        '--tokenizer_model ${{inputs.tokenizer_model}} '
        '--max_length ${{inputs.max_length}} '
        '--decoder_path ${{inputs.decoder_path}}'
        ),
    environment=f"{environment_name}:{environment_version}",
    compute=compute_name,
    environment_variables={
        "AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true"
        }
    )

# Define the pipeline
@pipeline(description="Pipeline to run model prediction")
def predict_pipeline(tokenizer_model="roberta-base", max_length=128):
    predict_step = predict_component(
        model_path=job_inputs["model_path"],
        data_path=job_inputs["data_path"],
        tokenizer_model=tokenizer_model,
        max_length=max_length,
        decoder_path=job_inputs["decoder_path"]
        )
    return {
        "predictions": predict_step.outputs.predictions
        }

# Create an instance of the pipeline job
try:
    pipeline_job_instance = predict_pipeline()
    print("Pipeline job instance created.")
except Exception as e:
    print("Failed to create pipeline job instance. Details:")
    print(e)

# Create and submit the pipeline job directly
try:
    submitted_job = ml_client.jobs.create_or_update(
        job=pipeline_job_instance,
        experiment_name="predict_experiment"
    )
    print(f"Pipeline job {submitted_job.name} is submitted.")
except Exception as e:
    print("Failed to submit the job. Details:")
    print(e)