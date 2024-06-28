.. _usage:

Usage
=====

This section provides information on how to use the project both through the Command Line Interface (CLI) and the API.

.. contents:: Table of Contents
   :depth: 2
   :local:

CLI Usage
---------

The Command Line Interface (CLI) allows you to interact with the project from the terminal.

1. Installation

First, ensure the project is installed. Check installation page.

2. General Arguments
The tool accepts several command line arguments that define the behavior of the program:

- **task**: Specifies the specific operation to be performed by the CLI tool. This is a mandatory parameter, and users must choose from the following valid options:

  - Preprocess: Converts audio/video files into structured data.
  - Train: Trains a new emotion detection model using provided data.
  - Predict: Predicts emotions from processed data using a trained model.
  - Add: Adds processed data to a database.

- cloud: Specifies whether to run the process in the cloud or locally. This is a mandatory parameter and it accepts True for cloud and False for local.

3. Preprocessing Arguments

- input_path: Path to the input file (audio or video).

- output_path: Path to save the output file. Defaults to output.csv.

- save: Indicates whether to save the processed data to disk (True or False). Defaults to False.

- target_sr: Target sample rate for the audio file to be processed. Accepts 8000, 16000, or 32000 Hz. Defaults to 32000 Hz.

- segment_length: Length of each audio segment for processing in seconds. Choices are 0.01, 0.02, or 0.03. Defaults to 0.03.

- min_fragment_len: Minimum length of the audio fragments considered for processing, in seconds. Defaults to 300.

- vad_aggressiveness: Aggressiveness level of the voice activity detection. Ranges from 0 (less aggressive) to 3 (more aggressive). Defaults to 0.

- use_fp16: Whether to use FP16 precision for model predictions, which can be faster on GPUs but is not suitable for CPUs. Defaults to True.

- transcript_model_size: Size of the Whisper model used for transcription. Choices are tiny, base, small, medium, and large. Defaults to large.

4. Training Arguments

- train_data: Path to the training dataset file (used with train task).

- val_data: Path to the validation dataset file (used with train task).

- val_size: Proportion of train data used as validation data to train the model. Defaults to 0.2.

- batch_size: Number of examples in one training batch. Defaults to 32.

- max_length: Max number of tokens used from one input sentence. Defaults to 128.

- epochs: Number of training epochs. Defaults to 5.

- learning_rate: Learning rate for the optimizer. Defaults to 1e-5.

- early_stopping_patience: Patience parameter in the early stopping callback. Defaults to 3.

- model_save_path: Path to the directory where the trained model will be saved. Defaults to new_model.

- threshold: Accuracy threshold for the model to be saved or registered. Defaults to 0.8.

- test_data: Path to test data for model evaluation.

- model_name: Name of the registered/saved model. Defaults to the current date.
  
5. Prediction Arguments 

- model_path: Path to the model file for training or prediction.

- label_decoder_path: Path to the file with label decoder data.

- tokenizer_model: Tokenizer model used to preprocess data. Defaults to roberta-base.

- token_max_length: Max number of tokens created by preprocessing a sentence. Defaults to 128.

Example Commands
-----------------

.. code-block:: shell

   # Preprocess an audio or video file
   cli --task Preprocess --cloud False --input_path /path/to/input/file --output_path /path/to/output/file --save True --target_sr 16000 --segment_length 0.02 --min_fragment_len 300 --vad_aggressiveness 1 --use_fp16 False --transcript_model_size large

   # Train a new emotion detection model
   cli --task Train --cloud True --train_data /path/to/train/data --val_data /path/to/val/data --val_size 0.2 --batch_size 64 --max_length 128 --epochs 10 --learning_rate 3e-5 --early_stopping_patience 5 --model_save_path /path/to/save/model --threshold 0.85 --test_data /path/to/test/data --model_name my_emotion_model

   # Predict emotions from processed data
   cli --task Predict --cloud False --model_path /path/to/model/file --label_decoder_path /path/to/label/decoder --tokenizer_model bert-base-uncased --token_max_length 128

   # Add processed data to a database
   cli --task Add --cloud True --input_path /path/to/processed/data

API Usage
-----------

To run API file locally relocate the file API/ `app.py` to package/e3k folder.
To see the documentation of the API refer to /docs url.

You can access the API using IP: http://57.153.92.125:8001.