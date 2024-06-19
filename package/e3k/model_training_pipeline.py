import config
from azure.ai.ml import MLClient, dsl
from azure.identity import ClientSecretCredential

# const values for Azure connection
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
compute = ml_client.compute.get("adsai0")

splitting_component = ml_client.components.get(
    name="split_register_component", version="2024-06-18-16-14-15-3695360"
)
train_component = ml_client.components.get(
    name="train_component", version="2024-06-18-16-15-39-1901929"
)
eval_component = ml_client.components.get(
    name="evaluation", version="2024-06-18-16-16-07-7286457"
)


@dsl.pipeline(
    name="model_training_pipeline",
    description="pipeline used to train new models",
    compute="adsai0",
)
def model_training(
    data_path: str,
    val_size: float,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    test_data: str,
    threshold: float,
    model_name: str,
):
    splitting_step = splitting_component(
        data_path=data_path, local=False, val_size=val_size
    )

    train_step = train_component(
        dataset_name_file=splitting_step.outputs.json_path,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
    )

    _ = eval_component(
        model_path=train_step.outputs.model,
        label_decoder=train_step.outputs.label_decoder,
        test_data=test_data,
        threshold=threshold,
        model_name=model_name,
    )


if __name__ == "__main__":
    # test pipeline on small dataset
    training_pipeline = model_training(
        data_path="dataset_panna/dataset_panna.csv",
        val_size=0.2,
        epochs=2,
        learning_rate=1e-3,
        early_stopping_patience=3,
        test_data="dataset_wojciech/test_azure_data.csv",
        threshold=0.0,
        model_name="training_test_model",
    )

    training_pipeline_run = ml_client.jobs.create_or_update(
        training_pipeline, experiment_name="test_model_training_pipeline"
    )
