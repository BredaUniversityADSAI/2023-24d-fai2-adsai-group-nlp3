# Natural Language Processing / TV Show Emotion Classifier

## Description / Project Overview 

This project focuses on the development and deployment of a production-ready Natural Language Processing (NLP) model for classifying emotions in TV shows. The aim is to provide Banijay Benelux with insights into viewer engagement and preferences within their TV series, particularly focusing on the popular show Expiditie Robinson.

## Installation

To install the project, follow these steps:

### For Conda Users

1. Download the `env.yaml` file from this project’s GitHub repository.
2. Navigate to the downloaded file.
3. Create the environment using the `env.yaml` file:
    ```bash
    conda env create -f env.yaml
    ```
4. Activate the newly created environment:
    ```bash
    conda activate <env_name>
    ```
5. Install the wheel package using pip:
    ```bash
    pip install wheel
    ```

### For Non-Conda Users

For non-conda users, having Python 3.10 is necessary.

1. Download the `requirements.txt` file from this project’s GitHub repository.
2. Navigate to the downloaded file.
3. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
4. Install the wheel package using pip:
    ```bash
    pip install wheel
    ```


### Verify the Installation

To verify the installation:
```bash
e3k --version
```

## Usage 

Our Command Line Interface (CLI) tool is designed to detect emotion from audio/video inputs. Below you'll find detailed instructions on how to use this tool, including all available options and commands.

### Command Line Arguments
#### General Arguments
The tool accepts several command line arguments that define the behavior of the program:

- **task**: Specifies the specific operation to be performed by the CLI tool. This is a mandatory parameter, and users must choose from the following valid options:

  - Preprocess: Converts audio/video files into structured data.
  - Train: Trains a new emotion detection model using provided data.
  - Predict: Predicts emotions from processed data using a trained model.
  - Add: Adds processed data to a database.

- cloud: Specifies whether to run the process in the cloud or locally. This is a mandatory parameter and it accepts True for cloud and False for local.

#### Preprocessing Arguments

- input_path: Path to the input file (audio or video).

- output_path: Path to save the output file. Defaults to output.csv.

- save: Indicates whether to save the processed data to disk (True or False). Defaults to False.

- target_sr: Target sample rate for the audio file to be processed. Accepts 8000, 16000, or 32000 Hz. Defaults to 32000 Hz.

- segment_length: Length of each audio segment for processing in seconds. Choices are 0.01, 0.02, or 0.03. Defaults to 0.03.

- min_fragment_len: Minimum length of the audio fragments considered for processing, in seconds. Defaults to 300.

- vad_aggressiveness: Aggressiveness level of the voice activity detection. Ranges from 0 (less aggressive) to 3 (more aggressive). Defaults to 0.

- use_fp16: Whether to use FP16 precision for model predictions, which can be faster on GPUs but is not suitable for CPUs. Defaults to True.

- transcript_model_size: Size of the Whisper model used for transcription. Choices are tiny, base, small, medium, and large. Defaults to large.

#### Training Arguments

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
  
#### Prediction Arguments 

- model_path: Path to the model file for training or prediction.

- label_decoder_path: Path to the file with label decoder data.

- tokenizer_model: Tokenizer model used to preprocess data. Defaults to roberta-base.

- token_max_length: Max number of tokens created by preprocessing a sentence. Defaults to 128.


## License
This project is licensed under the terms of the MIT license.