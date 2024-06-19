import config
from azure.ai.ml import Input, MLClient, dsl
from azure.identity import ClientSecretCredential

# Const values for Azure connection
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "NLP3"
TENANT_ID = "0a33589b-0036-4fe8-a829-3ed0926af886"
CLIENT_ID = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
CLIENT_SECRET = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

credential = ClientSecretCredential(
    config.config["tenant_id"],
    config.config["client_id"],
    config.config["client_secret"],
)

ml_client = MLClient(
    subscription_id=config.config["subscription_id"],
    resource_group_name=config.config["resource_group"],
    workspace_name=config.config["workspace_name"],
    credential=credential,
)

env = ml_client.environments.get("BlockD", version="2")
compute = ml_client.compute.get("adsai1")

episode_preprocessing_component = ml_client.components.get(
    name="episode_preprocessing_component", version="1"
)
predict_component = ml_client.components.get(
    name="prediction_component", version="2024-06-12-13-53-12-7299990"
)


@dsl.pipeline(
    name="model_predict_pipeline",
    description="Pipeline used for making predictions",
    compute="adsai0",
)
def model_predict(
    episode_data_path: Input(type="uri_file"),
    model_path: Input(type="uri_folder"),
    tokenizer_model: str,
    max_length: int,
    decoder_path: Input(type="uri_file"),
):
    preprocess_step = episode_preprocessing_component(
        episode_file=episode_data_path,
    )

    _ = predict_component(
        model_path=model_path,
        data_path=preprocess_step.outputs.transcription,
        tokenizer_model=tokenizer_model,
        max_length=max_length,
        decoder_path=decoder_path,
    )


if __name__ == "__main__":
    # Test prediction pipeline on a small dataset
    predict_pipeline = model_predict(
        episode_data_path=Input(
            path=(
                "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
                "resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/"
                "paths/test_evaluation_pipeline/trimmed_ER22_AFL01_MXF.mov"
            )
        ),
        model_path=Input(
            path=(
                "azureml://locations/westeurope/workspaces/"
                "12fa2fcc-0a79-4635-bc8c-8148b17bfac6/models/testModel/versions/3"
            )
        ),
        tokenizer_model="roberta-base",
        max_length=128,
        decoder_path=Input(
            path=(
                "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
                "resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/"
                "paths/labels_encodings/testModel/label_decoder"
            )
        ),
    )

    predict_pipeline_run = ml_client.jobs.create_or_update(
        predict_pipeline, experiment_name="test_model_predict_pipeline"
    )
