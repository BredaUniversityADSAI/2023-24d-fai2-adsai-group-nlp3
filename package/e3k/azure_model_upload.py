from azureml.core import Workspace
from azureml.core.model import Model

# Define the workspace
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "NLP3"

# Load the workspace
ws = Workspace(subscription_id=subscription_id, 
               resource_group=resource_group, 
               workspace_name=workspace_name)

# Register the model
model = Model.register(
    workspace=ws, 
    model_name="RoBERTa_model", 
    model_path='/Users/maxmeiners/Library/CloudStorage/OneDrive-BUas/Github/Year 2/Block D/model', 
    description="RoBERTa model for emotion recognition", 
    tags={"type": "emotion_recognition"}, 
    model_framework="transformers",
    model_framework_version="1.0"
)

print("Model registered")