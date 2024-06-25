from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.identity import ClientSecretCredential

# const values for Azure connection
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

env = ml_client.environments.get("BlockD", version="2")
compute = ml_client.compute.get("adsai0")

# define the component
episode_preprocessing_component = command(
    name="episode_preprocessing_component",
    display_name="Episode Preprocessing",
    description="Preprocess episode for prediction",
    inputs={
        "episode_file": Input(
            type="uri_file", description="episdode file"),
    },
    outputs={
        "transcription": Output(
            type="uri_folder", mode="rw_mount"),
    },
    code="./package/e3k",
    command=(
        "python episode_preprocessing_pipeline.py "
        "--cloud True "
        "--input_uri ${{inputs.episode_file}} "
        "--output_path ${{outputs.transcription}}"
    ),
    environment=env,
    compute_target=compute.name,
)

# register the component
#ml_client.create_or_update(episode_preprocessing_component.component)


@dsl.pipeline(
    name="test_ep_pipeline",
    description="testing if edp part works",
    compute="adsai0",
)
def test_ep_pipeline(
    episode_file, 
) -> None:
    ep_step = episode_preprocessing_component(
        episode_file=episode_file,
        )

episode = Input(
    path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/buas-y2/workspaces/NLP3/datastores/workspaceblobstore/paths/test_evaluation_pipeline/trimmed_ER22_AFL01_MXF.mov"
)

pipeline_instance = test_ep_pipeline(episode_file=episode)

pipeline_run = ml_client.jobs.create_or_update(pipeline_instance, experiment_name="EPP")