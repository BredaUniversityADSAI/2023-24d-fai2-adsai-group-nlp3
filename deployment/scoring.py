import os 
import tensorflow as tf
from package.e3k.preprocessing import get_tokenizer, tokenize_text_data, preprocess_prediction_data_no_tokenizer
from package.e3k.model_predict import decode_labels, get_model, predict, predict_scoring
import json
import transformers
import joblib
import pandas as pd
import logging

# Configure logging and setup on debug level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def init():
    # Define global variables
    global model
    global tokenizer
    global emotion_decoder

    # Get the path where the model is saved, set environment variable AZUREML_MODEL_DIR
    base_path = os.getenv('AZUREML_MODEL_DIR')
    print(f"base_path: {base_path}")


    # Add the model file name to the base path
    model_path = os.path.join(base_path, "model")
    # Print the model path
    print(f"model_path: {model_path}")

    # Load the Model
    model, emotion_decoder = get_model(model_path)
    print("Model loaded successfully.")

    # Get the tokenizer
    tokenizer = get_tokenizer(model_name="roberta-base")
    print(f"tokenizer: {tokenizer}")

def run(raw_data):
    
    # Load the JSON data from the POST reques, print the data to see the structure and content
    print(f"raw_data: {raw_data}")
    data = json.loads(raw_data)
    print(f"data: {data}")

    # Convert the JSON object (Python dictionary) to a DataFrame and then to a Series
    df = pd.DataFrame(data)
    print(f"DataFrame: {df}")
    
    # Log the content of emotion_decoder
    logger.info(f"Emotion decoder content: {emotion_decoder}")
    print(f"Emotion decoder content: {emotion_decoder}")
    
    # Preprocess the data
    tokens, masks = preprocess_prediction_data_no_tokenizer(df, tokenizer, max_length=128)
    print("Data preprocessed - Tokens and Masks created.")

    # Make predictions
    predictions, probabilities = predict(model, tokens, masks, emotion_decoder)
    print(f"prediction: {predictions, probabilities}")

    return json.dumps(predictions)


    
