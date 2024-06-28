from azureml.core import Dataset, Datastore, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication

# create auth method
auth = InteractiveLoginAuthentication()

# get workspace
workspace = Workspace(
    subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
    resource_group="buas-y2",
    workspace_name="NLP3",
    auth=auth,
)

# get datastore
datastore = Datastore(workspace=workspace, name="workspaceblobstore")

# upload data
datastore.upload(
    "dataset_wojciech",
    target_path="dataset_wojciech",
    overwrite=True,
    show_progress=True,
)

# get data from datastore
data = Dataset.File.from_files(path=(datastore, "dataset_wojciech"))

# create train, test, and validation sets
train_set, val_set = data.random_split(0.8, seed=42)
test_set = Dataset.File.from_files(path=(datastore, "dataset_wojciech"))

# register train, test, and validation sets as data assets on Azure
train_reg = train_set.register(
    workspace=workspace, name="wojciech_train", description="training data"
)
val_reg = val_set.register(
    workspace=workspace, name="wojciech_val", description="validation data"
)
test_reg = test_set.register(
    workspace=workspace, name="wojciech_test", description="test data"
)
