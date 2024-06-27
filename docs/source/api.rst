.. _api:

===============
API Documentation
===============

This API provides functionalities for training machine learning models, evaluating them,
preprocessing data, making predictions, and displaying output. It is built using FastAPI.

Base URL
========

All endpoints are relative to the base URL: ``http://your-api-domain.com``

Endpoints
=========

Train Model
-----------

Endpoint: ``/train/{val_size}/{epochs}/{lr}/{early_stopping_patience}``

Method: POST

Description: Trains a sequence classification model using uploaded training data.

Parameters:
   - ``val_size``: Size of validation data split.
   - ``epochs``: Number of training epochs.
   - ``lr``: Learning rate for training.
   - ``early_stopping_patience``: Patience for early stopping.
   - ``model_name``: Name to save the trained model.
   - ``train_data``: Uploaded training data file.

Returns: A message indicating the success of model training.

Example Usage:
   ::
   
       curl -X POST "http://your-api-domain.com/train/0.2/50/0.01/5/my_model" -F "train_data=@/path/to/train_data.csv"

Evaluate Model
--------------

Endpoint: ``/evaluate``

Method: POST

Description: Evaluates a trained model using uploaded test data.

Parameters:
   - ``model_name``: Name of the trained model to evaluate.
   - ``test_data``: Uploaded test data file.

Returns: Evaluation results including metrics.

Example Usage:
   ::
   
       curl -X POST "http://your-api-domain.com/evaluate" -F "model_name=my_model" -F "test_data=@/path/to/test_data.csv"

Predict Labels
--------------

Endpoint: ``/predict``

Method: POST

Description: Predicts emotions from uploaded audio data.

Parameters:
   - ``model_name``: Name of the trained model for prediction.
   - Various audio preprocessing parameters.
   - ``audio_file``: Uploaded audio file to predict emotions from.

Returns: Predicted emotions, probabilities, and episode confidence.

Example Usage:
   ::
   
       curl -X POST "http://your-api-domain.com/predict" -F "model_name=my_model" -F "audio_file=@/path/to/audio_file.wav"

Display Output
--------------

Endpoint: ``/pie-chart``

Method: GET

Description: Retrieves the generated pie chart image showing emotion distribution.

Returns: Pie chart image file.

Example Usage:
   ::
   
       curl -X GET "http://your-api-domain.com/pie-chart" -o emotion_distribution.png

Notes
=====

- **Model Storage:** Models and related files are stored in the ``models/{model_name}`` directory.
- **Cleanup:** Temporary files (``temp_{train_data/test_data/audio_file}``) are removed after processing.
- **Dependencies:** Ensure all necessary Python packages (``FastAPI``, ``uvicorn``, etc.) are installed and accessible.

Running the API
===============

To run the API server, execute the script:

   ::
   
       python app.py