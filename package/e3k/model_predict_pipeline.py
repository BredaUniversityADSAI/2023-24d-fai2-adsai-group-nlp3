from azure.ai.ml import MLClient, dsl, Input
from azure.identity import ClientSecretCredential

# Const values for Azure connection
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "NLP3"
TENANT_ID = "0a33589b-0036-4fe8-a829-3ed0926af886"
CLIENT_ID = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
CLIENT_SECRET = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

credential = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)

ml_client = MLClient(
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
    credential=credential,
)

env = ml_client.environments.get("BlockD", version="18")
compute = ml_client.compute.get("adsai0")

episode_preprocessing_component = ml_client.components.get(
    name="episode_preprocessing_component",
    version="2024-06-23-09-44-53-9256476"
)
predict_component = ml_client.components.get(
    name="prediction_component",
    version="2024-06-12-13-53-12-7299990"
)

@dsl.pipeline(
    name="model_predict_pipeline",
    description="Pipeline used for making predictions",
    compute="adsai0",
)
def model_predict(
    input_folder: Input(type="uri_folder"),
    input_filename: str,
    model_path: Input(type="uri_folder"),
    tokenizer_model: str,
    max_length: int,
    decoder_path: Input(type="uri_file"),
):
    preprocess_step = episode_preprocessing_component(
        input_folder=input_folder, input_filename=input_filename
    )

    predict_step = predict_component(
        model_path=model_path,
        data_path=preprocess_step.outputs.transcription,
        tokenizer_model=tokenizer_model,
        max_length=max_length,
        decoder_path=decoder_path,
    )
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
    f"paths/labels_encodings/{model_name}/label_decoder"
)

model_path = model.path
if __name__ == "__main__":
    

    # Test prediction pipeline on a small dataset
    predict_pipeline = model_predict(
        input_folder=Input(path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/paths/test_evaluation_pipeline/"),
        input_filename="trimmed_ER22_AFL01_MXF.mov",
        model_path=Input(path=model_path),
        tokenizer_model="roberta-base",
        max_length=128,
        decoder_path=Input(path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/paths/test_evaluation_pipeline/test_label_encoder.json")
    )

    predict_pipeline_run = ml_client.jobs.create_or_update(
        predict_pipeline,
        experiment_name="test_model_predict_pipeline"
    )
