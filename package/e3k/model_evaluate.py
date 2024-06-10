import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from azure.ai.ml.entities import Model
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from preprocessing import preprocess_prediction_data
from sklearn.metrics import accuracy_score, classification_report

mt_logger = logging.getLogger("main.model_evaluation")


# Loading data from local device
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

    df = pd.read_csv(file_path)[["sentence", "emotion"]].dropna()

    mt_logger.info(f"loaded data: {os.path.basename(file_path)}")

    return df


# Loading data from Azure ML datastore
def load_data_from_azure(test_data):
    """
    Loads data from an Azure ML datastore.

    This function connects to an Azure Machine Learning datastore using the provided
    URI and loads the specified test data.

    Input:
        test_data (str): The URI of the test data stored in the Azure ML datastore.

    Output:
        df (pd.DataFrame): Loaded DataFrame containing the data.

    -Author: Kornelia Flizik (223643)
    """

    # Loading the csv
    test_set = Dataset.Tabular.from_delimited_files(test_data)
    test_data = test_set.to_pandas_dataframe()

    return test_data


def predict(
    model: transformers.TFRobertaForSequenceClassification,
    token_array: np.array,
    mask_array: np.array,
    emotion_decoder: dict[int, str],
) -> tuple[list[str], list[float]]:
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
        highest_probabilities (list[float]): list of model's confidence
            that the predicted emotion is correct

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("predicting")

    input = {
        "input_ids": token_array,
        "attention_mask": mask_array,
    }

    preds = model(input)
    logits = preds.logits

    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    predicted_classes = np.argmax(probabilities, axis=1)
    highest_probabilities = np.max(probabilities, axis=1)

    text_labels = decode_labels(predicted_classes, emotion_decoder)

    mt_logger.info("got predictions")

    return text_labels, highest_probabilities


def evaluate(
    pred_labels,
    data,
) -> tuple[list[str], list[float], float, str]:
    """
    A function that evaluates trained model using a separate dataset.
    It returns predicted labels, and their probabilities, total_accuracy,
        and creates a report with different metrics.

    Input:
        model (transformers.TFRobertaForSequenceClassification): a loaded roBERTa model
        tokenizer (transformers.RobertaTokenizer): tokenizer compatible with the model
            architecture returned from the get_tokenizer function
        emotion_decoder (dict[int, str]): dictionary with number to text mapping loaded
            with get_model function
        eval_path (str): path do evaluation dataset CSV file

    Author:
        Max Meiners (214936)
    """

    mt_logger.info("evaluating trained model")

    true_labels = data["emotion"].to_list()

    accuracy = accuracy_score(true_labels, pred_labels)
    mt_logger.info(f"model accuracy: {accuracy}")

    report = classification_report(true_labels, pred_labels)

    return accuracy, report


def register_model_and_encoding(
    model_path, label_decoder, accuracy, workspace, threshold=0.5
):
    """
    Registers a machine learning model and its label encodings to Azure ML workspace
    if the accuracy exceeds a specified threshold.

    Inputs:
        model_path (str): The path to the model file that needs to be registered.
        label_encoder (str): The path to the label encoder file that needs to be
        uploaded to the datastore.
        accuracy (float):  The accuracy of the model.
        workspace (azureml.core.Workspace): The Azure ML workspace where the model
        and label encodings will be registered.
        threshold (float, optional): The accuracy threshold for registering the model.
        Default is 0.5.

    Output:
        None

    - Author: Kornelia Flizik
    """
    mt_logger.info(f"Registering model if accuracy is above {threshold}.")

    # Only register model if accuracy is above threshold
    if accuracy > threshold:
        mt_logger.info("Model accuracy is above threshold, registering model.")

        # Register the model
        Model.register(
            workspace=workspace,
            model_name="Emotion Classification",
            model_path=model_path,
            description="RoBERTa model for emotion recognition",
        )

        mt_logger.info("Model registered")

        # Register label encodings
        datastore = Datastore(workspace, name="workspaceblobstore")

        datastore.upload(
            src_dir=os.path.dirname(label_decoder),
            target_path="labels_encodings",
            overwrite=True,
            show_progress=True,
        )

        mt_logger.info("Encodings saved")

    else:
        mt_logger.info("Model accuracy is not above threshold, not registering model.")


def save_model(
    model: transformers.TFRobertaForSequenceClassification,
    label_decoder: dict[int, str],
    model_path: str,
    model_accuracy: float,
    threshold: float = 0.5,
) -> None:
    """
    A function that saves trained model and it's emotion mapping to a file.

    Input:
        model (transformers.TFRobertaForSequenceClassification): trained model
        label_encoder (dict[int, str]): Python dictionary mapping
            numbers to text emotions.
        model_path (str): Path to directory where the model will be saved.

    Output: None

    Author:
        Max Meiners (214936)
    """

    if model_accuracy > threshold:
        mt_logger.info(
            "model's accuracy on the test set surpassed the threshold, saving model"
        )

        dict_path = os.path.join(model_path, "emotion_dict.json")

        model.save_pretrained(model_path)
        with open(dict_path, "w") as f:
            json.dump(label_decoder, f)
        mt_logger.info("Model saved")

    else:
        mt_logger.info(
            "model's accuracy on the test set didn't surpass the threshold,"
            "not saving the model"
        )


"""
Util functions (only used in other functions)
"""


def decode_labels(
    encoded_labels: list[int], emotion_decoder: dict[int, str]
) -> list[str]:
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

    decoded_labels = list(map(lambda x: emotion_decoder[x], encoded_labels))

    return decoded_labels


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a saved TensorFlow model from the specified path.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tf.keras.Model: Loaded model.


    """
    # Check if the model file exists at the specified path and load it if it does.
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        print(f"No model file found at {model_path}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cloud",
        type=bool,
        choices=[True, False],
        help="whether the code will execute on Azure or locally",
    )

    parser.add_argument(
        "--test_data",
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
        "--model_path",
        required=False,
        type=str,
        help="Path to the model file or config.",
    )

    parser.add_argument(
        "--label_decoder",
        required=False,
        type=dict[int, str],
        help="File containing label decoder dict",
    )

    args = parser.parse_args()

    cloud = args.cloud == "True"

    model = load_model(args.model_path)

    if cloud is True:
        # Load the workspace
        auth = InteractiveLoginAuthentication()

        workspace = Workspace(
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            workspace_name=args.workspace_name,
            auth=auth,
        )

        data = load_data_from_azure(args.test_data)
        tokens, masks = preprocess_prediction_data(data)
        emotions, probabilities = predict(model, tokens, masks, args.label_decoder)
        accuracy, _ = evaluate(emotions, data)
        print(f"Test accuracy: {accuracy * 100:.2f}%")
        register_model_and_encoding(
            args.model_path, args.label_decoder, accuracy, workspace
        )

    else:
        data, _ = load_data(args.test_data)
        tokens, masks = preprocess_prediction_data(data)
        emotions, probabilities = predict(model, tokens, masks, args.label_decoder)
        accuracy, _ = evaluate(emotions, data)
        print(f"Test accuracy: {accuracy * 100:.2f}%")
