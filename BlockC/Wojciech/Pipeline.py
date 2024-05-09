import torch
import os
import shutil
import whisper
try:
    import deepspeech
except:
    print("Couldn't load deepspeech")
import wave

import re
import numpy as np
import pandas as pd
import keras

# To release GPU memmory after execution
from numba import cuda 
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_extract_audio
from tqdm import tqdm

class Video2EmotionClassifier:
    """
    A class for processing video fragments, transcribing audio, and predicting emotions from text.
    """
    def __init__(self, pattern = r'(?<=\.|\?|\!)\s', fp16=True):
         """
        Initialize the Video2EmotionClassifier object.

        Args:
            pattern (str): Regular expression pattern used to split text into sentences. Defaults to r'(?<=\.|\?|\!)\s'.
            fp16 (bool): Whether to use FP16 format for model prediction, needs to be False for CPU. Defaults to True.
        """
        self.pattern = pattern
        self.fp16 = fp16
        self.id2label = {
            0: 'happiness',
            1: 'sadness',
            2: 'anger',
            3: 'surprise',
            4: 'fear',
            5: 'disgust'
        }
        
        try:
            self.deepspeech_model = deepspeech.model('deepspeech-0.9.3-models.pbmm')
            self.deepspeech = True
        except:
            self.deepspeech = False
        
    def InitializeModels(self, model_name='fine_tuned_bert_model_05_04_2024'):
        """
        Initialize the Transformer model for emotion prediction and tokenization model.

        Args:
            model_name (str): Name of the pre-trained model to be used for classification. Defaults to 'fine_tuned_bert_model_05_04_2024'.
        """
        self.classifier = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def Video2FragmentTranscript(self, full_video_path, segment_df, deepspeech=False, transcription_model_type = 'base'):
        """
        Transcribe audio from video fragments and return the text transcript.

        Args:
            full_video_path (str): Path to the full video file.
            segment_df (DataFrame): DataFrame containing start and end times of video segments.
            deepspeech (bool): Whether to use DeepSpeech for transcription. Defaults to False.
            transcription_model_type (str): Type of transcription model to use if DeepSpeech is not used. Defaults to 'base'.

        Returns:
            List of text transcripts for each video segment.
        """
        try:
            os.mkdir("temp_clips")
            os.mkdir("temp_audio")
        except:
            pass

        #model = whisper.load_model(transcription_model_type)

        #segment_df = segment_df[['start_seconds', 'end_seconds']]
        segment_df = segment_df[['Start Time (seconds)', 'End Time (seconds)']]
        
        start_seconds_col_index = 0
        end_seconds_col_index = 1
        if not deepspeech:
            model = whisper.load_model(transcription_model_type)
        results = []

        for row_index in tqdm(range(segment_df.shape[0])):
            start = segment_df.iloc[row_index, start_seconds_col_index]
            end = segment_df.iloc[row_index, end_seconds_col_index]

            clip_path = f"temp_clips/{start}_{end}.mov"
            audio_path = f"temp_audio/{os.path.basename(clip_path[:-4])}.mp3"

            ffmpeg_extract_subclip(full_video_path, start, end, clip_path)
            ffmpeg_extract_audio(clip_path, audio_path)
            if deepspeech:
                self.transcribe_audio_with_deepspeech(audio_path)
            else:
                result = model.transcribe(audio_path, fp16=self.fp16, language='en')

            results.append(result['text'])


        #shutil.rmtree("data/client/temp_clips")
        #shutil.rmtree("data/client/temp_audio")

        return results
        
    def transcribe_audio_with_deepspeech(self, audio_path):
        """
        Transcribe audio using the DeepSpeech model.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            Transcribed text.
        """
        fin = wave.open(audio_path, 'rb')
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        fin.close()

        return self.deepspeech_model.stt(audio)
    
    
    def Fragment2Sentences(self, fragments, returning=True):
        """
        Split fragments into sentences using a specified pattern.

        Args:
            fragments (list): List of text fragments to be split into sentences.
            returning (bool): Whether to return the list of sentences. Defaults to True.

        Returns:
            List of sentences if returning is True.
        """
        self.sentences = []
        for fragment in fragments:
            self.sentences.append(re.split(self.pattern, fragment), )
        if returning:
            return self.sentences        
    
    def Sentence2Emotion(self, sentences, release=True, threshold=0.8):
        """
        Predict emotions for given sentences.

        Args:
            sentences (list): List of sentences for emotion prediction.
            release (bool): Whether to release memory after prediction. Defaults to True.
            threshold (float): Confidence threshold for emotion prediction. Defaults to 0.8.

        Returns:
            List of predicted emotions for each sentence.
        """
        # Initialize models
        
        # Preprocess input
        x_sentences = self.tokenizer(list(sentences), truncation=True, padding=True)
        
        x_sentences_input = {key: np.array(value) for key, value in x_sentences.items()}
        # Predict
        preds = self.classifier.predict(x_sentences_input).logits
        preds = keras.layers.Softmax()(preds)
        
        highest_pred = 0
        labels = []
        for i, pred in enumerate(preds):
            max_value = np.max(pred)
            if max_value > highest_pred:
                highest_pred = max_value

            if max_value < threshold:
                labels.append('neutral')
            else:
                max_index = np.argmax(pred)
                labels.append(self.id2label[max_index])
        print(highest_pred)

        #labels = pd.DataFrame(labels)
        #labels.columns = ['emotion', 'sentence']
        #labels.index.name = 'id'    
        
        
        if release:
            del x_sentences
            # Release GPU memmory
            device = cuda.get_current_device()
            device.reset()
            
        return labels
    def PostProcessing(self, fragment_array, emotions_array):
        """
        Combine fragment transcripts with predicted emotions.

        Args:
            fragment_array (list): List of fragment transcripts.
            emotions_array (list): List of predicted emotions for each fragment.

        Returns:
            DataFrame containing fragment transcripts and predicted emotions.
        """
        emotions = []
        for inner in emotions_array:
            emotions.append(list(set(inner)))
        combined_data = list(zip(fragment_array, emotions))
        results = pd.DataFrame(combined_data)
        results.columns = ['fragment transcript', 'emotions']
        results.index.name = 'id'
        return results