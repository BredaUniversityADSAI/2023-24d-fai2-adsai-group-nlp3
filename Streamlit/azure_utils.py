from azureml.core import Workspace, Model
from azureml.core.authentication import ServicePrincipalAuthentication
from azure.identity import ClientSecretCredential
import streamlit as st


def get_azure_workspace():
    # Retrieve secrets from Streamlit
    subscription_id = st.secrets["azure"]["SUBSCRIPTION_ID"]
    resource_group = st.secrets["azure"]["RESOURCE_GROUP"]
    workspace_name = st.secrets["azure"]["WORKSPACE_NAME"]
    tenant_id = st.secrets["azure"]["TENANT_ID"]
    client_id = st.secrets["azure"]["CLIENT_ID"]
    client_secret = st.secrets["azure"]["CLIENT_SECRET"]

    # Authenticate using Service Principal
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=client_id,
        service_principal_password=client_secret
    )

    # Get the Azure ML Workspace
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        auth=svc_pr
    )

    return ws

def list_models(ws):
    models = Model.list(ws)
    model_names = [model.name for model in models]
    return model_names

def get_model(ws, model_name):
    model = Model(ws, name=model_name)
    return model