import os
import pickle
import shutil
import pandas as pd
from typing import Dict, List, Tuple

import package.e3k.episode_preprocessing_pipeline as epp
import package.e3k.model_evaluate as me
import package.e3k.model_output_information as moi
import package.e3k.model_training as mt
import package.e3k.preprocessing as preprocessing
import package.e3k.split_register_data as splitting

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from transformers import TFRobertaForSequenceClassification


def episode_preprocessing(
    input_path,
    sample_rate,
    segment_seconds_length,
    vad_aggressiveness,
    min_fragment_len,
    transcript_model_size,
    use_fp16,
) -> pd.DataFrame:
    """
    Function that follows the episode_preprocessing_pipeline module.
    It uses module's functions to get the final output: pd.DataFrame with sentences.
    For more information on the functions used here, see the
    episode_preprocessing_pipeline module, and check the docstrings
    for separate functions.

    Input:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line

    Output (pd.DataFrame): result of episode_preprocessing_pipeline.
        pd.DataFrame with sentences from the audio/video.

    Author - Wojciech Stachowiak
    """
    # get segment length in frames
    segment_frames_length = epp.segment_number_to_frames(
        1, sample_rate=sample_rate, segment_seconds_length=segment_seconds_length
    )

    # load audio file and set sample rate to the chosen value
    audio = epp.load_audio(file_path=input_path, target_sample_rate=sample_rate)

    # get full audio length in frames
    full_audio_length_frames = len(audio)

    # get segments for vad analysis
    segments = epp.get_segments_for_vad(
        audio=audio,
        sample_rate=sample_rate,
        segment_seconds_length=segment_seconds_length,
    )

    # get vad output per segment
    speech_array = epp.get_vad_per_segment(
        segments=segments,
        vad_aggressiveness=vad_aggressiveness,
        sample_rate=sample_rate,
        segment_frames_length=segment_frames_length,
    )

    # get fragment sizes of chosen length
    cut_fragments_frames = epp.get_frame_segments_from_vad_output(
        speech_array=speech_array,
        sample_rate=sample_rate,
        min_fragment_length_seconds=min_fragment_len,
        segment_seconds_length=segment_seconds_length,
        full_audio_length_frames=full_audio_length_frames,
    )

    # transcribe and translate fragments to get sentences in df
    use_fp16 = use_fp16 == "True"
    data_df = epp.transcribe_translate_fragments(
        audio=audio,
        cut_fragments_frames=cut_fragments_frames,
        sample_rate=sample_rate,
        use_fp16=use_fp16,
        transcription_model_size=transcript_model_size,
    )

    return data_df


def model_training(
    train_data,
    val_data,
    val_size,
    epochs,
    lr,
    early_stopping_patience,
) -> Tuple[
    TFRobertaForSequenceClassification, Tuple[List[str], List[float], float, str]
]:
    """
    A function that follows the model_training module.
    It loads and pre-processes the data for model training, creates a new model,
    and fits the data into the model.

    Input:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line

    Output:
        model: a trained roBERTa transformer model

    Author - Wojciech Stachowiak
    """

    train_data, _ = splitting.load_data(train_data)

    if val_data == "":
        train_data, val_data = splitting.get_train_val_data(train_data, val_size)
    else:
        val_data, _ = splitting.load_data(val_data)
    label_decoder = mt.get_label_decoder(train_data["emotion"])

    train_dataset, val_dataset = preprocessing.preprocess_training_data(
        train_data, val_data, label_decoder, max_length=30
    )

    model = mt.get_new_model(len(label_decoder))
    model = mt.train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=epochs,
        learning_rate=lr,
        early_stopping_patience=early_stopping_patience,
    )

    return model, label_decoder


def evaluate_model(
    test_data,
    model: TFRobertaForSequenceClassification,
    label_decoder: Dict[int, str],
) -> None:
    """
    A function that evaluates a trained model using a separate dataset.

    Inputs:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line
        model (TFRobertaForSequenceClassification): a trained roBERTa model
        label_decoder (dict[int, str]): python dictionary mapping numbers
        to text emotions

    Outputs: None

    Author - Wojciech Stachowiak
    """
    data = me.load_data(test_data)
    tokens, masks = preprocessing.preprocess_prediction_data(data, max_length=30)
    emotions, _ = me.predict(model, tokens, masks, label_decoder)
    accuracy, _ = me.evaluate(emotions, data)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    # me.save_model(model, label_decoder, model_save_path, accuracy, threshold)
    return accuracy


def predict(
    model: TFRobertaForSequenceClassification,
    label_decoder: Dict[int, str],
    data,
) -> Tuple[List[str], List[float]]:
    """
    A function that returns model predictions given a model_path command line argument,
    and dataframe with column named "sentence"

    Input:
        args (argparse.Namespace): Namespace object returned by get_args function.
            holds the information about positional and optional arguments
            from command line
        data (pd.DataFrame): dataframe with sentences in a column

    Output:
        emotions (list[str]): text representation of predicted emotion for
            each sentence
        probabilities (list[float]): model's confidence for
            the most probable emotion in each sentence

    Author - Wojciech Stachowiak
    """

    tokens, masks = preprocessing.preprocess_prediction_data(data, max_length=30)
    emotions, probabilities = me.predict(model, tokens, masks, label_decoder)

    return emotions, probabilities


def model_output_information(
    predicted_emotions: List[str], confidence_scores: List[float]
) -> None:
    """
    A function that aggregates prediction results into a total confidence score,
    and a pie chart with predicted emotions distribution.

    Input:
        predicted_emotions (list[str]): text representation of predicted emotion
            for each sentence
        highest_probabilities (list[float]): model's confidence for the most
        probable emotion in each sentence

    Output: None

    Author - Wojciech Stachowiak
    """
    plt = moi.plot_emotion_distribution(predicted_emotions)
    ep_confidence = moi.calculate_episode_confidence(confidence_scores)

    return plt, ep_confidence


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/train/{val_size}/{epochs}/{lr}/{early_stopping_patience}")
def train_endpoint(
    val_size: float,
    epochs: int,
    lr: float,
    early_stopping_patience: int,
    model_name: str,
    train_data: UploadFile = File(...),
):
    """
    Endpoint to train a sequence classification model.

    Args:
        val_size (float): Size of validation data split.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for training.
        early_stopping_patience (int): Patience for early stopping.
        model_name (str): Name to save the trained model.
        train_data (UploadFile): Uploaded training data file.

    Returns:
        dict: A message indicating the success of model training.

    Author: Kornelia Flizik
    """

    input_path = f"temp_{train_data.filename}"
    with open(input_path, "wb") as f:
        f.write(train_data.file.read())

    model, label_decoder = model_training(
        train_data=input_path,
        val_size=val_size,
        epochs=epochs,
        lr=lr,
        early_stopping_patience=early_stopping_patience,
        val_data="",
    )

    # Directory to store the temporary model files
    MODEL_DIR = f"models/{model_name}"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_save_path = os.path.join(MODEL_DIR, "model")
    model.save_pretrained(model_save_path)
    # Save label decoder as well (you can use JSON or pickle)
    label_decoder_path = os.path.join(MODEL_DIR, "label_decoder.pkl")
    with open(label_decoder_path, "wb") as f:
        pickle.dump(label_decoder, f)

    os.remove(input_path)  # Clean up the temporary file

    return {"message": "Model trained successfully"}


@app.post("/evaluate")
def evaluate_endpoint(model_name: str, test_data: UploadFile = File(...)):
    """
    Endpoint to evaluate a trained model using test data.

    Args:
        model_name (str): Name of the trained model to evaluate.
        test_data (UploadFile): Uploaded test data file.

    Returns:
        dict: Evaluation results including metrics.

    Author: Kornelia Flizik
    """

    MODEL_DIR = f"models/{model_name}"

    model_save_path = os.path.join(MODEL_DIR, "model")
    label_decoder_path = os.path.join(MODEL_DIR, "label_decoder.pkl")
    if not os.path.exists(model_save_path) or not os.path.exists(label_decoder_path):
        return {"error": "Wrong model name"}

    input_path = f"temp_{test_data.filename}"
    with open(input_path, "wb") as f:
        f.write(test_data.file.read())

    # Load the model and label decoder
    model = TFRobertaForSequenceClassification.from_pretrained(model_save_path)
    with open(label_decoder_path, "rb") as f:
        label_decoder = pickle.load(f)

    evaluation_result = evaluate_model(
        test_data=input_path, model=model, label_decoder=label_decoder
    )
    os.remove(input_path)
    return evaluation_result


@app.post("/predict")
def predict_emotions(
    model_name: str,
    target_sr: int = 32000,
    segment_length: float = 0.03,
    min_fragment_len: int = 300,
    vad_aggressiveness: int = 0,
    use_fp16: bool = True,
    transcript_model_size: str = "large",
    audio_file: UploadFile = File(...),
):
    """
    Endpoint to predict emotions from audio data.

    Args:
        model_name (str): Name of the trained model for prediction.
        target_sr (int): Target sample rate for audio data.
        segment_length (float): Length of audio segments in seconds.
        min_fragment_len (int): Minimum fragment length for audio segmentation.
        vad_aggressiveness (int): VAD (Voice Activity Detection) aggressiveness level.
        use_fp16 (bool): Flag indicating whether to use FP16 precision.
        transcript_model_size (str): Size of the transcription model.
        audio_file (UploadFile): Uploaded audio file to predict emotions from.

    Returns:
        dict: Predicted emotions, probabilities, and episode confidence along
        with a link to the generated pie chart.

    Author: Kornelia Flizik
    """

    # Save uploaded file to a temporary location
    input_path = f"temp_{audio_file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    data_df = episode_preprocessing(
        input_path=input_path,
        sample_rate=target_sr,
        segment_seconds_length=segment_length,
        min_fragment_len=min_fragment_len,
        vad_aggressiveness=vad_aggressiveness,
        use_fp16=use_fp16,
        transcript_model_size=transcript_model_size,
    )

    MODEL_DIR = f"models/{model_name}"

    model_save_path = os.path.join(MODEL_DIR, "model")
    label_decoder_path = os.path.join(MODEL_DIR, "label_decoder.pkl")
    if not os.path.exists(model_save_path) or not os.path.exists(label_decoder_path):
        return {"error": "Wrong model name"}

    # Load the model and label decoder
    model = TFRobertaForSequenceClassification.from_pretrained(model_save_path)
    with open(label_decoder_path, "rb") as f:
        label_decoder = pickle.load(f)

    tokens, masks = preprocessing.preprocess_prediction_data(data_df, max_length=30)
    emotions, probabilities = me.predict(model, tokens, masks, label_decoder)

    # Generate model output information
    plt, ep_confidence = model_output_information(emotions, probabilities)
    print(type(plt))
    print(plt)
    plt.savefig("emotion_distribution.png")

    # Clean up temporary file
    os.remove(input_path)
    probabilities = list(map(lambda x: float(x), probabilities))

    return {
        "emotions": emotions,
        "probabilities": probabilities,
        "episode confidence": ep_confidence,
    }


@app.get("/pie-chart")
def get_pie_chart():
    """
    Endpoint to retrieve the generated pie chart image.

    Returns:
        FileResponse: Response with the pie chart image file.

    Author: Kornelia Flizik
    """
    pie_chart_path = "emotion_distribution.png"
    return FileResponse(
        pie_chart_path, media_type="image/png", filename="pie_chart.png"
    )


if __name__ == "__main__":
    """
    Run the API server.
    """

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
