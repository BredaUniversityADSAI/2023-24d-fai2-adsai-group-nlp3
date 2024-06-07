from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml import dsl


SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "NLP3"


credential = DefaultAzureCredential()

ml_client = MLClient(
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    credential=credential,
    workspace_name=WORKSPACE_NAME
)

env = ml_client.environments.get("BlockD", version="2")
compute = ml_client.compute.get("adsai0")

train_component = command(
    name="train_env2_check",
    display_name="Model training",
    description="Train model with data from a predefined data asset",
    inputs={
    "train_data": Input(
        type="string",
        description="training data",
        default="wojciech_val"
    ),
    "val_data": Input(
        type="string",
        description="validation dataset for model training",
        default="wojciech_val"
    ),
    "dataset_version": Input(
        type="string",
        description="version of training and validation sets",
        default="1"
    ),
    "epochs": Input(
        type="integer",
        description="number of epochs to train the model for",
        default=3
    ),
    "learning_rate": Input(
        type="number",
        description="learning rate of the model's optimizer",
        default=1e-3
    ),
    "early_stopping_patience": Input(
        type="integer",
        description="patience parameter of the early stopping callback",
        default=3
        )
    },
    outputs={
        "model": Output(type="uri_folder", mode="upload"),
        "label_decoder": Output(type="uri_file", mode="upload")
    },
    code="./package/e3k/Components",
    command=(
    "python model_training.py "
    "--cloud True "
    "--train_dataset_name ${{inputs.train_data}} "
    "--val_dataset_name ${{inputs.val_data}} "
    "--dataset_version ${{inputs.dataset_version}} "
    "--epochs ${{inputs.epochs}} "
    "--learning_rate ${{inputs.learning_rate}} "
    "--early_stopping_patience ${{inputs.early_stopping_patience}} "
    "--model_output_path ${{outputs.model}} "
    "--decoder_output_path ${{outputs.label_decoder}}"
    ),
    environment=env,
    compute_target=compute.name
)

# ml_client.create_or_update(train_component)

@dsl.pipeline(
    name="test_model_training_pipeline",
    description="testing if model_training part works",
    compute="adsai0",
)
def test_training_pipeline(train_dataset_name, val_dataset_name, dataset_version, epochs, learning_rate, early_stopping_patience) -> None:
    train_step = train_component(
        train_data=train_dataset_name,
        val_data=val_dataset_name,
        dataset_version=dataset_version,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience
    )

pipeline_instance = test_training_pipeline(
    train_dataset_name="wojciech_val",
    val_dataset_name="wojciech_val",
    dataset_version="1",
    epochs=3,
    learning_rate=1e-3,
    early_stopping_patience=3
)

pipeline_run = ml_client.jobs.create_or_update(pipeline_instance)