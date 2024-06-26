from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import PipelineJob
from azure.ai.ml.constants import AssetTypes

# const values for Azure connection
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
env = ml_client.environments.get("BlockD", version="20")
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
        "transcription": Output(type="uri_file", mode="upload", description="Transcription output"),
    },
    code="./package/e3k/",
    command=(
        "python3 episode_preprocessing_pipeline.py "
        "--cloud True "
        "--input_file ${{inputs.input_folder}} "
        "--input_filename ${{inputs.input_filename}} "
        "--output_path ${{outputs.transcription}}"
    ),
    environment=env,
    compute_target=compute.name,
)

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
        "data_path": Input(
            type="uri_folder", 
            description="Data to be predicted"
        ),
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
    code="./package/e3k/",
    command=(
        "python3 model_predict.py "
        "--model_path ${{inputs.model_path}} "
        "--data_path ${{inputs.data_path}} "
        "--tokenizer_model ${{inputs.tokenizer_model}} "
        "--max_length ${{inputs.max_length}} "
        "--decoder_path ${{inputs.decoder_path}} "
        "--output_path ${{outputs.predictions}}"
    ),
    environment=env,
    compute=compute.name,
)

# Create or update the prediction component
#ml_client.create_or_update(predict_component.component)



# Define a pipeline that uses this component
@dsl.pipeline(
    name="ep&predict-pipeline",
    description="Pipeline to preprocess episodes&predict",
    default_compute=ml_client.compute.get("adsai0").name
)
def preprocess_predict_pipeline(
    input_folder: Input, 
    input_filename: str,
    model_path,
    tokenizer_model,
    max_length,
    decoder_path,

    ):
    preprocess_step = episode_preprocessing_component(
        input_folder=input_folder,
        input_filename=input_filename,

    )

    predict_step = predict_component(
        model_path=model_path,
        data_path=preprocess_step.outputs.transcription,
        tokenizer_model=tokenizer_model,
        max_length=max_length,
        decoder_path=decoder_path,
     )

    return {
        "prediction_output": predict_step.outputs.predictions,
   }

#ml_client.create_or_update(episode_preprocessing_component.component)


# Retrieve the model from the workspace
model_name = "training_test_model"
model_version = "1"
model = ml_client.models.get(
    name=model_name, 
    version=model_version
)

decoder_folder_path = (
    "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
    "resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/"
    f"paths/labels_encodings/{model_name}"
)

model_path = model.path


# Test prediction pipeline on a small dataset
predict_pipeline = preprocess_predict_pipeline(
        input_folder=Input(path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/paths/test_evaluation_pipeline/"),
        input_filename="trimmed_video_test.mov",
        model_path=Input(path=model_path),
        tokenizer_model="roberta-base",
        max_length=128,
        decoder_path= Input(path=decoder_folder_path),
    )

# Submit the pipeline job
pipeline_job = ml_client.jobs.create_or_update(predict_pipeline, experiment_name="EPP")

print(f"Pipeline run initiated: {pipeline_job.name}")

