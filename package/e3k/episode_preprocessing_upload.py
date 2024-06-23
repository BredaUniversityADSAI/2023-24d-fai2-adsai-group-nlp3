from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import PipelineJob

# Const values for Azure connection
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "NLP3"
TENANT_ID = "0a33589b-0036-4fe8-a829-3ed0926af886"
CLIENT_ID = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
CLIENT_SECRET = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

# Authenticate and create an MLClient
credential = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)

ml_client = MLClient(
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
    credential=credential,
)

# Get the environment and compute
env = ml_client.environments.get("BlockD", version="18")
compute = ml_client.compute.get("adsai0")

# Define the preprocessing component
episode_preprocessing_component = command(
    name="episode_preprocessing_component",
    display_name="Episode Preprocessing",
    description="Preprocess episode for prediction",
    inputs={
        "input_folder": Input(type="uri_folder", description="Input folder containing files"),
        "input_filename": Input(type="string", description="Name of the input file"),
    },
    outputs={
        "transcription": Output(type="uri_folder", mode="rw_mount", description="Transcription output"),
    },
    code="",
    command=(
        "python3 episode_preprocessing_pipeline.py "
        "--input_folder ${{inputs.input_folder}} "
        "--input_filename ${{inputs.input_filename}} "
        "--output_path ${{outputs.transcription}}"
    ),
    environment=env,
    compute_target=compute.name,
)

# Define a pipeline that uses this component
@dsl.pipeline(
    name="episode-preprocessing-pipeline",
    description="Pipeline to preprocess episodes",
    default_compute=ml_client.compute.get("adsai0").name
)
def preprocess_pipeline(input_folder: Input, input_filename: str):
    preprocess_step = episode_preprocessing_component(
        input_folder=input_folder,
        input_filename=input_filename,
    )
    return {
        "transcription_output": preprocess_step.outputs.transcription,
    }

ml_client.create_or_update(episode_preprocessing_component.component)


# Create the pipeline job
pipeline = preprocess_pipeline(
    input_folder=Input(type="uri_folder", path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/paths/test_evaluation_pipeline/"),
    input_filename="trimmed_ER22_AFL01_MXF.mov",
)

# Submit the pipeline job
pipeline_job = ml_client.jobs.create_or_update(pipeline)
print(f"Pipeline run initiated: {pipeline_job.name}")

