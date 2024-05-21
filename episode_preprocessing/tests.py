import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from episode_preprocessing_pipeline import (
    load_audio_from_video,
    get_segments_for_vad,
    get_vad_per_segment,
    transcribe_translate_fragments,
    save_data
)  

@pytest.fixture
def mock_audio_data():
    sample_rate = 32000
    return np.random.rand(sample_rate * 10)  # Mock 10 seconds of audio data

@pytest.fixture
def audio_file_path():
    # Replace with the actual path to a sample audio file
    return 'path_to_sample_audio_file'

def test_load_audio_from_video(audio_file_path):
    target_sample_rate = 32000
    assert os.path.exists(audio_file_path), "Audio file path does not exist"
    audio = load_audio_from_video(audio_file_path, target_sample_rate)
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0, "Loaded audio should not be empty"

def test_get_segments_for_vad(mock_audio_data):
    sample_rate = 32000
    segment_length = 0.03
    segments = get_segments_for_vad(mock_audio_data, sample_rate, segment_length)
    assert isinstance(segments, list)
    assert len(segments) > 0, "Segments list should not be empty"
    for segment in segments:
        assert isinstance(segment, np.ndarray)

def test_get_vad_per_segment(mock_audio_data):
    sample_rate = 32000
    segment_length = 0.03
    segments = get_segments_for_vad(mock_audio_data, sample_rate, segment_length)
    vad_aggressiveness = 0
    segment_frames_length = int(segment_length * sample_rate)
    vad_results = get_vad_per_segment(segments, vad_aggressiveness, sample_rate, segment_frames_length)
    assert isinstance(vad_results, np.ndarray)
    assert len(vad_results) == len(segments)

def test_transcribe_translate_fragments(mock_audio_data):
    sample_rate = 32000
    segment_length = 0.03
    segments = get_segments_for_vad(mock_audio_data, sample_rate, segment_length)
    vad_aggressiveness = 0
    segment_frames_length = int(segment_length * sample_rate)
    vad_results = get_vad_per_segment(segments, vad_aggressiveness, sample_rate, segment_frames_length)
    cut_fragments_frames = [(0, len(mock_audio_data))]  # Assuming entire audio as one fragment for simplicity
    transcription_df = transcribe_translate_fragments(mock_audio_data, cut_fragments_frames, sample_rate, use_fp16=True, transcription_model_size="base")
    assert isinstance(transcription_df, pd.DataFrame)
    assert len(transcription_df) > 0, "Transcription dataframe should not be empty"

def test_save_data():
    df = pd.DataFrame({
        'episode': [1],
        'segment': [1],
        'segment_start_seconds': [0.0],
        'segment_end_seconds': [10.0],
        'sentence': ['This is a test sentence.']
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmpfile:
        save_data(df, tmpfile.name)
        tmpfile.close()
        with open(tmpfile.name, 'r') as file:
            content = file.read()
            assert len(content) > 0, "Saved file should not be empty"
        os.remove(tmpfile.name)

if __name__ == '__main__':
    pytest.main()
