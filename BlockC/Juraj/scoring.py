import os 
import tensorflow as tf
from preprocessing import get_tokenizer, tokenize_text_data, preprocess_prediction_data
from model_predict import decode_labels, get_model, predict
import json
import transformers
import joblib



def init():
    # Define global variables
    global model
    global tokenizer
    global label_decoder

    # Get the path where the model is saved, set environment variable AZUREML_MODEL_DIR
    base_path = os.getenv('AZUREML_MODEL_DIR')
    print(f"base_path: {base_path}")


    # Add the model file name to the base path
    model_path = os.path.join(base_path, "INPUT_model", 'RoBERTa_model')
    # Print the model path
    print(f"model_path: {model_path}")

    # Load the Model
    model = get_model(model_path)
    print("Model loaded successfully.")
'''
    # Get the tokenizer
    tokenizer = get_tokenizer(model_path)
    print(f"tokenizer: {tokenizer}")

    # Decode Labels from model path from file named test_label_encoder.json
    label_decoder = decode_labels(model_path)
    print(f"label_decoder: {label_decoder}")
'''

def run(raw_data):
    # Load the JSON data from the POST reques, print the data to see the structure and content
    print(f"raw_data: {raw_data}")
    data = json.loads(raw_data)
    print(f"data: {data}")

    # Preprocess the data
    tokens, masks = preprocess_prediction_data(data, tokenizer, max_length=128)
    print("Data preprocessed - Tokens and Masks created.")

    # Make predictions
    prediction = predict(model, tokens, masks)
    print(f"prediction: {prediction}")
    # Get the predicted label
    predicted_label = label_decoder[prediction]

    # Print the predicted label
    print(f"Predicted label: {predicted_label}")
    print(f"Output Format: {json.dumps(predicted_label.tolist())}")

    return json.dumps(predicted_label.tolist())


    
