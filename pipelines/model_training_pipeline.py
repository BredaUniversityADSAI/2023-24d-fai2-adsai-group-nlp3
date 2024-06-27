from datetime import datetime, timedelta

import config
import typeguard
from azure.ai.ml import MLClient, dsl
from azure.ai.ml.constants import TimeZone
from azure.ai.ml.entities import (JobSchedule, RecurrencePattern,
                                  RecurrenceTrigger)
from azure.ai.ml.sweep import Choice, Uniform
from azure.identity import ClientSecretCredential

# Const values for Azure connection
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
    name="train_component", version="2024-06-25-08-40-02-1931108"
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
    """
    This function creates and runs a machine learning pipeline to train a model.
    It includes data splitting, training with hyperparameter tuning,
    and model evaluation steps.

    Input:
        data_path (str): Path to the training data.
        val_size (float): Data to be used for validation.
        epochs (int): Number of training epochs.
        test_data (str): Path to the test data.
        threshold (float): Accuracy threshold for model evaluation.
        model_name (str): Name of the model to be trained and evaluated.

    Output:
        None

    Author:
        Wojciech Stachowiak & Max Meiners
    """
    splitting_step = splitting_component(
        data_path=data_path, local=False, val_size=val_size
    )

    train_step = train_component(
        dataset_name_file=splitting_step.outputs.json_path,
        epochs=epochs,  # Fixed for each trial
        learning_rate=Uniform(min_value=1e-5, max_value=1e-1),
        batch_size=Choice([16, 32, 64, 128]),
        early_stopping_patience=3,
    ).sweep(sampling_algorithm="bayesian", primary_metric="val_loss", goal="minimize")

    train_step.set_limits(
        max_total_trials=3,  # Number of trials for different hyperparameter sets
        max_concurrent_trials=3,  # Number of trials to run concurrently
        timeout=7200,  # Maximum allowed time for the sweep
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
        test_data="dataset_wojciech/test_azure_data.csv",
        threshold=0.0,
        model_name="training_test_model",
    )

    training_pipeline_run = ml_client.jobs.create_or_update(
        training_pipeline, experiment_name="test_model_training_pipeline"
    )

    # Define the schedule for the pipeline
    schedule_name = "model_training_pipeline_schedule"

    # Setting the start time for 2 minutes from now
    schedule_start_time = datetime.utcnow() + timedelta(minutes=2)

    recurrence_trigger = RecurrenceTrigger(
        frequency="day",
        interval=1,
        schedule=RecurrencePattern(hours=10, minutes=[0, 30]),
        start_time=schedule_start_time,
        time_zone=TimeZone.UTC,
    )

    job_schedule = JobSchedule(
        name=schedule_name,
        trigger=recurrence_trigger,
        create_job=training_pipeline,
    )

    # Create or update the schedule
    ml_client.schedules.begin_create_or_update(job_schedule)
