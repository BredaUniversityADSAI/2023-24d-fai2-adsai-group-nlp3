from azure.ai.ml import Input, MLClient, Output, command
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

env = ml_client.environments.get("BlockD", version="12")
compute = ml_client.compute.get("adsai0")

# define the component
train_component = command(
    name="train_component",
    display_name="Model training",
    description="Train model with data from a predefined data asset",
    inputs={
        "dataset_name_file": Input(
            type="uri_file", description="json file with dataset names"
        ),
        "epochs": Input(
            type="integer",
            description="number of epochs to train the model for",
            default=3,
        ),
        "learning_rate": Input(
            type="number",
            description="learning rate of the model's optimizer",
            default=1e-3,
        ),
        "early_stopping_patience": Input(
            type="integer",
            description="patience parameter of the early stopping callback",
            default=3,
        ),
    },
    outputs={
        "model": Output(type="uri_folder", mode="rw_mount"),
        "label_decoder": Output(type="uri_file", mode="rw_mount"),
    },
    code="../package/e3k",
    command=(
        "python model_training.py "
        "--cloud True "
        "--dataset_name_file ${{inputs.dataset_name_file}} "
        "--epochs ${{inputs.epochs}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--early_stopping_patience ${{inputs.early_stopping_patience}} "
        "--model_output_path ${{outputs.model}} "
        "--decoder_output_path ${{outputs.label_decoder}}"
    ),
    environment=env,
    compute_target=compute.name,
)

# register the component
ml_client.create_or_update(train_component.component)

# running the component as a pipeline
"""
@dsl.pipeline(
    name="test_model_training_pipeline",
    description="testing if model_training part works",
    compute="adsai0"
)
def test_training_pipeline(
    train_dataset_name,
    val_dataset_name,
    dataset_version,
    epochs,
    learning_rate,
    early_stopping_patience,
) -> None:
    _ = train_component(
        train_data=train_dataset_name,
        val_data=val_dataset_name,
        dataset_version=dataset_version,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
    )


pipeline_instance = test_training_pipeline(
    train_dataset_name="wojciech_val",
    val_dataset_name="wojciech_val",
    dataset_version="1",
    epochs=1,
    learning_rate=1e-3,
    early_stopping_patience=3,
)

pipeline_run = ml_client.jobs.create_or_update(pipeline_instance)
"""
