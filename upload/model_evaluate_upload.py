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

env = ml_client.environments.get("BlockD", version="2")
compute = ml_client.compute.get("adsai0")

train_component = command(
    name="train_env2_check",
    display_name="Model training",
    description="Train model with data from a predefined data asset",
    inputs={
        "train_data": Input(
            type="string", description="training data", default="wojciech_val"
        ),
        "val_data": Input(
            type="string",
            description="validation dataset for model training",
            default="wojciech_val",
        ),
        "dataset_version": Input(
            type="string",
            description="version of training and validation sets",
            default="1",
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
        "model_path": Output(type="uri_folder", mode="upload"),
        "label_decoder": Output(type="uri_file", mode="upload"),
    },
    code="./package/e3k/",
    command=(
        "python model_training.py "
        "--cloud True "
        "--train_dataset_name ${{inputs.train_data}} "
        "--val_dataset_name ${{inputs.val_data}} "
        "--dataset_version ${{inputs.dataset_version}} "
        "--epochs ${{inputs.epochs}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--early_stopping_patience ${{inputs.early_stopping_patience}} "
        "--model_output_path ${{outputs.model_path}} "
        "--decoder_output_path ${{outputs.label_decoder}}"
    ),
    environment=env,
    compute_target=compute.name,
)

evaluate_component = command(
    name="evaluation",
    display_name="Model evaluation",
    description="Evaluate model using test data",
    inputs={
        "test_data": Input(type="string", description="Data asset URI"),
        "model_path": Input(
            type="uri_folder",
            description="Model URI",
        ),
        "label_decoder": Input(
            type="uri_folder",
            description="URI of label encoding dict",
        ),
        "model_name": Input(type="string", description="Name to register the model"),
        "threshold": Input(
            type="number",
            description="Min accuracy for the model to be considered good",
        ),
    },
    code="./package/e3k/",
    command=(
        "python model_evaluate.py "
        "--cloud True "
        "--test_data_name ${{inputs.test_data}} "
        "--model_path ${{inputs.model_path}} "
        "--label_decoder ${{inputs.label_decoder}} "
        "--model_name ${{inputs.model_name}} "
        "--threshold ${{inputs.threshold}}"
    ),
    environment=env,
    compute_target=compute.name,
)


ml_client.create_or_update(evaluate_component.component)

"""
@dsl.pipeline(
    name="test_model_evaluation_pipeline",
    description="testing if model_evaluation part works",
    compute="adsai0",
)
def test_training_pipeline(
    train_dataset_name,
    val_dataset_name,
    dataset_version,
    epochs,
    learning_rate,
    early_stopping_patience,
    test_data,
    model_name,
) -> None:
    train_step = train_component(
        train_data=train_dataset_name,
        val_data=val_dataset_name,
        dataset_version=dataset_version,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
    )
    eval_step = evaluate_component(
        test_data=test_data,
        model_path=train_step.outputs.model_path,
        label_decoder=train_step.outputs.label_decoder,
        model_name=model_name
    )


pipeline_instance = test_training_pipeline(
    train_dataset_name="wojciech_val",
    val_dataset_name="wojciech_val",
    dataset_version="1",
    epochs=3,
    learning_rate=1e-3,
    early_stopping_patience=3,
    test_data='dataset_wojciech/test_azure_data.csv',
    model_name="testModel",
    )

pipeline_run = ml_client.jobs.create_or_update(
    pipeline_instance, experiment_name="Model_evaluation"
)
"""
