from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import Input, MLClient, Output, command


SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "NLP3"


credential = InteractiveBrowserCredential()

ml_client = MLClient(
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    credential=credential,
    workspace_name=WORKSPACE_NAME
)

env = ml_client.environments.get("BlockD", version="1")
compute = ml_client.compute.get("adsai1")

train_component = command(
    name="train",
    display_name="Model training",
    description="Train model with data from a predefined data asset",
    inputs={
        "train_data": Input(
            type="string",
            description="training data"
        ),
        "val_data": Input(
            type="string",
            description="validation dataset for model training"
        ),
        "dataset_version": Input(
            type="string",
            description="version of training and validation sets"
        ),
        "epochs": Input(
            type="integer",
            description="number of epochs to train the model for"
        ),
        "learning_rate": Input(
            type="number",
            description="learning rate of the model's optimizer"
        ),
        "early_stopping_patience": Input(
            type="integer",
            description="patience parameter of the early stopping callback"
            )
    },
    outputs={
        "model": Output(type="uri_folder", mode="upload"),
        "label_decoder": Output(type="uri_folder", mode="upload")
    },
    code="./package/e3k/new_model_training.py",
    command="""
    python new_model_training.py 
    --train_dataset_name ${{inputs.train_data}} 
    --val_dataset_name ${{inputs.val_data}} 
    --dataset_version ${{inputs.dataset_version}} 
    --epochs ${{input.epochs}} 
    --learning_rate ${{inputs.learning_rate}} 
    --early_stopping_patience ${{inputs.early_stopping_patience}}
    """,
    environment=env,
    compute_target=compute.name
)

ml_client.create_or_update(train_component.component)