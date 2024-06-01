# Natural Language Processing / TV Show Emotion Classifier

## Description / Project Overview 

This project focuses on the development and deployment of a production-ready Natural Language Processing (NLP) model for classifying emotions in TV shows. The aim is to provide Banijay Benelux with insights into viewer engagement and preferences within their TV series, particularly focusing on the popular show Expiditie Robinson.

## Installation


## Usage 

Our Command Line Interface (CLI) tool is designed to detect emotion from audio/video inputs. Below you'll find detailed instructions on how to use this tool, including all available options and commands.

### Command Line Arguments
The tool accepts several command line arguments that define the behavior of the program:

- **task**: Specifies the specific operation to be performed by the CLI tool. This is a mandatory parameter, and users must choose from the following valid options:

  -  Preprocess: Converts audio/video files into structured data.
  - Train: Trains a new emotion detection model using provided data.
  - Predict: Predicts emotions from processed data using a trained model.
  - Add: Adds processed data to a database.

- input_path: Path to the input file (audio or video).

- output_path: Path to save the output file. Defaults to output.csv.

- model_path: Path to the model file for training or prediction.

- train_data: Path to the training dataset file (used with train task).

- eval_data: Path to the evaluation dataset file (used with train task).

- save: Indicates whether to save the processed data to disk (true/false). Defaults to false.

Additional arguments specific to tasks like preprocessing or training include settings like sample rates, fragment lengths, and model parameters.

Additional Arguments
- target_sr: Target sample rate for the audio file to be processed. Accepts 8000, 16000, or 32000 Hz. Defaults to 32000 Hz.
- segment_length: Length of each audio segment for processing in seconds. Choices are 0.01, 0.02, or 0.03. Defaults to 0.03.
- min_fragment_len: Minimum length of the audio fragments considered for processing, in seconds. Defaults to 300.
- vad_aggressiveness: Aggressiveness level of the voice activity detection. Ranges from 0 (less aggressive) to 3 (more aggressive). Defaults to 0.
- use_fp16: Whether to use FP16 precision for model predictions, which can be faster on GPUs but is not suitable for CPUs. Defaults to True.
- transcript_model_size: Size of the Whisper model used for transcription. Choices are tiny, base, small, medium, and large. Defaults to large.

## License
This project is licensed under the terms of the MIT license.