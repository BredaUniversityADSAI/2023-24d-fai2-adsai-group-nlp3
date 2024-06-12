from azure.ai.ml import Input, MLClient, command
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

evaluate_component = command(
    name="evaluation",
    display_name="Model evaluation",
    description="Evaluate model using test data",
    inputs={
        "test_data": Input(type="uri_folder", description="Data asset URI"),
        "model_path": Input(
            type="uri_folder",
            description="Model URI",
        ),
        "label_decoder": Input(
            type="Uri_folder",
            description="URI of label encoding dict",
        ),
    },
    code="./package/e3k/Components",
    command=(
        "python model_evaluate.py "
        "--cloud True "
        "--train_dataset_name ${{inputs.test_data}} "
        "--val_dataset_name ${{inputs.model_path}} "
        "--dataset_version ${{inputs.label_decoder}} "
    ),
    environment=env,
    compute_target=compute.name,
)
