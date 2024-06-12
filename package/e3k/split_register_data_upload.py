from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import ClientSecretCredential
import logging
from azure.ai.ml import Input, MLClient, Output, command, dsl

# Define the workspace parameters
subscription_id = '0a94de80-6d3b-49f2-b3e9-ec5818862801'
resource_group = 'buas-y2'
workspace_name = 'NLP3'
tenant_id = '0a33589b-0036-4fe8-a829-3ed0926af886'
client_id = '27157a5a-3927-4895-8478-9d4554697d25'
client_secret = 'stf8Q~mP2cB923Mvz5K91ITcoYgvRXs4J1lysbfb'

# Setup logging
logging.basicConfig(level=logging.INFO)
mt_logger = logging.getLogger("azure_ml")

# Authenticate and create an MLClient
credential = ClientSecretCredential(tenant_id, client_id, client_secret)
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Define the compute target
compute_name = "adsai0"
try:
    compute_target = ml_client.compute.get(compute_name)
    mt_logger.info("Found existing compute target.")
except Exception:
    mt_logger.info("Creating new compute target.")

# Get the environment
environment = ml_client.environments.get(name="BlockD", version="1")

# Define the split_register component
split_register_component = command(
    name="split_register_component",
    display_name="Data Split and Register",
    description="Splits the dataset into training and validation sets and registers them in Azure ML",
    inputs={
        "data_path": Input(type="string", description="Path to the data file"),
        "local": Input(type="string", description="Load data locally or from Azure", default="True"),
        "val_size": Input(type="number", description="Validation data size as a proportion of the dataset", default=0.2)
    },
    outputs={
        "train_data": Output(type="uri_file", mode="upload", description="Registered training dataset"),
        "val_data": Output(type="uri_file", mode="upload", description="Registered validation dataset")
    },
    code="./split_register_data.py",
    command="""
    python split_register.py
    --data_path ${{inputs.data_path}}
    --local ${{inputs.local}}
    --val_size ${{inputs.val_size}}
    --train_data ${{outputs.train_data}}
    --val_data ${{outputs.val_data}}
    """,
    environment=environment,
    compute=compute_name,
)

# Create or update the split_register component
# ml_client.create_or_update(split_register_component.component)


@dsl.pipeline(
    name="split_register_data",
    description="testing if the split_register_data works",
    compute="adsai0",
)

def test_split_register_pipeline(
    data_path: str,
    local: str,
    val_size: float
) -> None:
    # Using the split_register_component to split and register the dataset
    split_step = split_register_component(
        data_path=data_path,
        local=local,
        val_size=val_size
    )

# Instantiate the pipeline
pipeline_instance = test_split_register_pipeline(
    data_path="dataset_panna/dataset_panna.csv",
    local="False",
    val_size=0.2
)

# Submit the pipeline job
pipeline_run = ml_client.jobs.create_or_update(pipeline_instance)
print(f"Pipeline run submitted with ID: {pipeline_run.id}")
