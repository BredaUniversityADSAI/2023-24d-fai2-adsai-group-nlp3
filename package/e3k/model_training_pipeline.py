import config
import typeguard
from azure.ai.ml import MLClient, dsl
from azure.ai.ml.sweep import Uniform, Choice
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

env = ml_client.environments.get("BlockD", version="21")
compute = ml_client.compute.get("adsai0")

splitting_component = ml_client.components.get(
    name="split_register_component", version="2024-06-18-16-14-15-3695360"
)
train_component = ml_client.components.get(
    name="train_component", version="2024-06-20-14-50-42-8000239"
)
eval_component = ml_client.components.get(
    name="evaluation", version="2024-06-18-16-16-07-7286457"
)


@typeguard.typechecked
@dsl.pipeline(
    name="model_training_pipeline",
    description="pipeline used to train new models",
    compute="adsai0",
)
def model_training(
    data_path: str,
    val_size: float,
    epochs: int,
    test_data: str,
    threshold: float,
    model_name: str,
):
    splitting_step = splitting_component(
        data_path=data_path, local=False, val_size=val_size
    )

    train_step = train_component(
        dataset_name_file=splitting_step.outputs.json_path,
        epochs=epochs,  # Fixed for each trial
        learning_rate=Uniform(min_value=1e-5, max_value=1e-1),
        batch_size=Choice([16, 32, 64, 128]),
        early_stopping_patience=3,
    ).sweep(
        sampling_algorithm="bayesian",
        primary_metric="val_loss",
        goal="minimize"
    )

    train_step.set_limits(
        max_total_trials=3,  # Number of trials for different hyperparameter sets
        max_concurrent_trials=2,  # Number of trials to run concurrently
        timeout=7200  # Maximum allowed time for the sweep
    )

    eval_step = eval_component(
        model_path=train_step.outputs.model,
        label_decoder=train_step.outputs.label_decoder,
        test_data=test_data,
        threshold=threshold,
        model_name=model_name,
    )

    return {
        "model_path": train_step.outputs.model,
        "evaluation_results": eval_step.outputs.results,
    }

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
