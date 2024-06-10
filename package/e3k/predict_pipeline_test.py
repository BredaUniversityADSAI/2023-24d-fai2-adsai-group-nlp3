from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import PipelineJob
import os

# Define credentials and workspace details
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
client_secret = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "NLP3"
model_name = "RoBERTa_model"

# Authenticate and create an MLClient
credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret
)

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name
)

# Define the compute target and environment
compute_name = "adsai0"
environment_name = "BlockD"
environment_version = "1"

# Define the command component
predict_component = command(
    name="prediction_component",
    display_name="Model Prediction Component",
    description="Predict emotions from input data using a pre-trained RoBERTa model",
    inputs={
        "model_path": Input(type="uri_folder"),
        "data_path": Input(type="uri_file"),
        "tokenizer_model": Input(type="string"),
        "max_length": Input(type="integer"),
        "decoder_path": Input(type="uri_file")
    },
    outputs={
        "predictions": Output(type="uri_file")
    },
    code=os.path.abspath("./model_predict.py"),
    command="""
    python model_predict.py 
    --model_path ${{inputs.model_path}} 
    --data_path ${{inputs.data_path}} 
    --tokenizer_model ${{inputs.tokenizer_model}} 
    --max_length ${{inputs.max_length}} 
    --decoder_path ${{inputs.decoder_path}} 
    --output_path ${{outputs.predictions}}
    """,
    environment=f"{environment_name}:{environment_version}",
    compute=compute_name
)

# Define the pipeline
@pipeline(description="Pipeline to run model prediction")
def predict_pipeline(model_name, 
                     data_path, 
                     tokenizer_model, 
                     max_length, 
                     decoder_path):
    predict_step = predict_component(
        model_path=f"azureml://datastores/workspaceblobstore/Models/{model_name}",
        data_path=data_path,
        tokenizer_model=tokenizer_model,
        max_length=max_length,
        decoder_path=decoder_path
    )
    return {
        "predictions": predict_step.outputs.predictions
    }

# Instantiate the pipeline
data_path = "azureml://datastores/workspaceblobstore/dataset_max/emotions_all_azure_10rows.csv"
decoder_path = "azureml://datastores/workspaceblobstore/data/decoder.joblib"  # Update your actual decoder path

# Create an instance of the pipeline job
pipeline_job_instance = predict_pipeline(
    model_name=model_name,
    data_path=data_path,
    tokenizer_model="roberta-base",
    max_length=128,
    decoder_path=decoder_path
)

# Create and submit the pipeline job directly
try:
    submitted_job = ml_client.jobs.create_or_update(
        job=pipeline_job_instance,  # Submit the pipeline instance directly
        experiment_name="predict_experiment"
    )
    print(f"Pipeline job {submitted_job.name} is submitted.")
except Exception as e:
    print("Failed to submit the job. Details:")
    print(e)