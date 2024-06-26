import argparse
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import typeguard
from preprocessing import preprocess_prediction_data
from sklearn.metrics import accuracy_score, classification_report

# setting up logger
eval_logger = logging.getLogger(
    f"{'main.' if __name__ != '__main__' else ''}{__name__}"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    eval_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    eval_logger.addHandler(stream_handler)

file_handler = logging.FileHandler("logs.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

eval_logger.addHandler(file_handler)


# Loading data from local device
@typeguard.typechecked
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file and return
    the DataFrame and a dictionary with labels.
    CSV needs to have "sentence" and "emotion" columns.

    Input:
        file_path (str): File path to the dataset CSV file.

    Output:
        df (pd.DataFrame): Loaded DataFrame containing the data.

    Author:
        Max Meiners (214936)
    """

    # get only necessary columns
    df = pd.read_csv(file_path)[["sentence", "emotion"]].dropna()

    eval_logger.info(f"loaded data: {os.path.basename(file_path)}")

    return df


@typeguard.typechecked
def predict(
    model: transformers.TFRobertaForSequenceClassification,
    token_array: np.array,
    mask_array: np.array,
    emotion_decoder: Dict[int, str],
) -> Tuple[List[str], np.array]:
    """
    A function that predicts emotions from preprocessed input using a loaded model.
    It returns text labels decoded using emotion_decoder dictionary loaded
    with the model.

    Input:
        model (transformers.TFRobertaForSequenceClassification): a loaded roBERTa model
        token_array (np.array): a token array returned by tokenize_text_data function
        mask_array (np.array): a mask array returned by tokenize_text_data function
        emotion_decoder (dict[int, str]): dictionary with number to text mapping loaded
            with get_model function

    Output:
        text_labels (list[str]): list of text emotions predicted by the model
        highest_probabilities (np.array): array of model's confidence
            that the predicted emotion is correct

    Author:
        Max Meiners (214936)
    """

    eval_logger.info("predicting")

    # get input from tokens and masks
    input = {
        "input_ids": token_array,
        "attention_mask": mask_array,
    }

    # get predictions
    preds = model(input)
    logits = preds.logits

    # get classes and confidence
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    predicted_classes = np.argmax(probabilities, axis=1)
    highest_probabilities = np.max(probabilities, axis=1)

    text_labels = decode_labels(predicted_classes, emotion_decoder)

    eval_logger.info("got predictions")

    return text_labels, highest_probabilities


@typeguard.typechecked
def evaluate(
    pred_labels: List[str],
    data: pd.DataFrame,
) -> Tuple[float, str]:
    """
    A function that evaluates trained model using a separate dataset.
    It returns total_accuracy and creates a report with different metrics.

    Input:
        pred_labels (list[str]): list of text emotions predicted by the model
        data (pd.DataFrame): test dataset

    Output:
        accuracy (float): The accuracy of the model on test data.
        report (str): The clafication report of the model on test data.

    Author:
        Kornelia Flizik (223643)
    """

    eval_logger.info("evaluating trained model")

    true_labels = data["emotion"].to_list()

    # get model's accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    eval_logger.info(f"model accuracy: {accuracy}")

    # create classification report
    report = classification_report(true_labels, pred_labels)

    return accuracy, report


@typeguard.typechecked
def save_model(
    model: transformers.TFRobertaForSequenceClassification,
    label_decoder: Dict[int, str],
    model_path: str,
    accuracy: float,
    threshold: float,
) -> None:
    """
    A function that saves trained model and it's emotion mapping to a file.

    Input:
        model (transformers.TFRobertaForSequenceClassification): trained model
        label_encoder (dict[int, str]): Python dictionary mapping
            numbers to text emotions.
        model_path (str): Path to directory where the model will be saved.
        accuracy (float):  The accuracy of the model.
        threshold (float): The accuracy threshold for registering the model.

    Output: None

    Author:
        Max Meiners (214936)
    """

    if accuracy >= threshold:
        eval_logger.info("Model accuracy is above threshold, saving model.")

        dict_path = os.path.join(model_path, "emotion_dict")

        model.save_pretrained(model_path)
        with open(dict_path, "wb") as f:
            pickle.dump(label_decoder, f)

        eval_logger.info("Model saved")
    else:
        eval_logger.info("Model accuracy is not above threshold, not saving model.")


"""
Util functions (only used in other functions)
"""


@typeguard.typechecked
def decode_labels(
    encoded_labels: np.array, emotion_decoder: Dict[int, str]
) -> List[str]:
    """
    A function that decodes label numbers into text representation of labels

    Input:
        encoded_labels (list[int]): list of labels represented as number
        emotion_decoder (dict[int, str]): dictionary with number to text mapping loaded
            with get_model function

    Output:
        decoded_labels (list[str]): list of labels represented as text

    Author:
        Max Meiners (214936)
    """

    decoded_labels = list(map(lambda x: emotion_decoder[x.item()], encoded_labels))

    return decoded_labels


@typeguard.typechecked
def load_label_decoder(label_decoder_path: str) -> Dict[int, str]:
    """
    Load a label decoder from a pickle file.

    This function loads a dictionary that maps integer labels to string descriptions
    from a specified pickle file.

    Input:
        label_decoder_path (str): The path to the pickle file containing
        the label decoder.

    Output:
        emotion_decoder (Dict[int, str]): A dictionary mapping integer labels to their
            corresponding string descriptions.

    Author:
        Kornelia Flizik (223643)
    """
    # Load the emotion_decoder using pickle
    with open(label_decoder_path, "rb") as f:
        emotion_decoder = pickle.load(f)
        return emotion_decoder


if __name__ == "__main__":
    import config
    from azureml.core import Dataset, Datastore, Model, Workspace
    from azureml.core.authentication import ServicePrincipalAuthentication

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cloud",
        type=bool,
        choices=[True, False],
        help="whether the code will execute on Azure or locally",
    )

    parser.add_argument(
        "--test_data_path",
        required=False,
        type=str,
        default="",
        help="URI of the test data from Azure ML datastore",
    )

    parser.add_argument(
        "--subscription_id",
        required=False,
        type=str,
        default="0a94de80-6d3b-49f2-b3e9-ec5818862801",
        help="Subscription ID from Azure ML",
    )

    parser.add_argument(
        "--resource_group",
        required=False,
        type=str,
        default="buas-y2",
        help="Resource group from Azure ML",
    )

    parser.add_argument(
        "--workspace_name",
        required=False,
        type=str,
        default="NLP3",
        help="Workspace name from Azure ML",
    )

    parser.add_argument(
        "--datastore_name",
        required=False,
        type=str,
        default="workspaceblobstore",
        help="Datastore name from Azure ML",
    )

    parser.add_argument(
        "--test_data_name",
        required=False,
        type=str,
        help="Data name from Azure ML",
    )

    parser.add_argument(
        "--model_path",
        required=False,
        type=str,
        help="Path to the model file or config.",
    )

    parser.add_argument(
        "--label_decoder",
        required=False,
        type=str,
        help="File containing label decoder dict",
    )

    parser.add_argument(
        "--threshold",
        required=False,
        type=float,
        default=0.8,
        help="Min accuracy for the model to be considered good",
    )

    parser.add_argument(
        "--model_name",
        required=False,
        type=str,
        help="Name to register the model",
    )

    args = parser.parse_args()

    @typeguard.typechecked
    def load_data_from_azure(
        workspace: Workspace,
        datastore_name: str,
        test_data_name: str,
    ) -> pd.DataFrame:
        """
        Loads data from an Azure ML datastore.

        This function connects to an Azure Machine Learning datastore using the provided
        URI and loads the specified test data.

        Input:
            test_data (str): The URI of the test data stored in the Azure ML datastore.

        Output:
            df (pd.DataFrame): Loaded DataFrame containing the data.

        Author:
            Kornelia Flizik (223643)
        """

        # Loading the csv
        # test_set = Dataset.Tabular.from_delimited_files(test_data_uri, validate=False)
        # Get the default datastore
        datastore = Datastore(workspace, name=datastore_name)

        test_set = Dataset.Tabular.from_delimited_files(
            path=(datastore, test_data_name)
        )

        test_data = test_set.to_pandas_dataframe()

        return test_data

    @typeguard.typechecked
    def register_model_and_encoding(
        model_path: str,
        label_decoder: Dict[int, str],
        accuracy: float,
        workspace: Workspace,
        model_name: str,
        threshold: float,
    ) -> None:
        """
        Registers a machine learning model and its label encodings to Azure ML workspace
        if the accuracy exceeds a specified threshold.

        Inputs:
            model_path (str): The path to the model file that needs to be registered.
            label_decoder (str): The path to the label encoder file that needs to be
            uploaded to the datastore.
            accuracy (float): The accuracy of the model.
            workspace (azureml.core.Workspace): The Azure ML workspace where the model
            and label encodings will be registered.
            model_name (str): The name of the model used for Azure ML Models.
            threshold (float): The accuracy threshold for registering the model.

        Output:
            None

        - Author:
            Kornelia Flizik (223643)
        """

        eval_logger.info(f"Registering model if accuracy is above {threshold}.")

        # Only register model if accuracy is above threshold
        if accuracy > threshold:
            eval_logger.info("Model accuracy is above threshold, registering model.")

            # Register the model
            _ = Model.register(
                workspace=workspace,
                model_path=model_path,
                model_name=model_name,
                description="RoBERTa model for emotion recognition",
            )

            eval_logger.info("Model registered")

            # Register label encodings
            datastore = Datastore(workspace, name="workspaceblobstore")

            _ = datastore.upload(
                src_dir=os.path.dirname(label_decoder),
                target_path=f"labels_encodings/{model_name}",
                overwrite=False,
                show_progress=True,
            )

            eval_logger.info("Encodings saved")

        else:
            eval_logger.info(
                "Model accuracy is not above threshold, not registering model."
            )

    model = transformers.TFRobertaForSequenceClassification.from_pretrained(
        args.model_path
    )

    # Load the workspace
    eval_logger.info("cloud path")

    svc_pr = ServicePrincipalAuthentication(
        tenant_id=config.config["tenant_id"],
        service_principal_id=config.config["client_id"],
        service_principal_password=config.config["client_secret"],
    )
    workspace = Workspace(
        subscription_id=config.config["subscription_id"],
        resource_group=config.config["resource_group"],
        workspace_name=config.config["workspace_name"],
        auth=svc_pr,
    )
    # change
    data = load_data_from_azure(workspace, args.datastore_name, args.test_data_name)
    label_decoder = load_label_decoder(args.label_decoder)
    tokens, masks = preprocess_prediction_data(data)
    emotions, probabilities = predict(model, tokens, masks, label_decoder)
    accuracy, _ = evaluate(emotions, data)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    register_model_and_encoding(
        args.model_path,
        args.label_decoder,
        accuracy,
        workspace,
        args.model_name,
        args.threshold,
    )
