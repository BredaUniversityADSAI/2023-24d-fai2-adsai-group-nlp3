import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from episode_preprocessing_pipeline import (
    load_audio_from_video,
    get_segments_for_vad,
    get_vad_per_segment,
    get_frame_segments_from_vad_output,
    transcribe_translate_fragments,
    save_data
)

@pytest.fixture

@pytest.mark.parametrize("file_path, sample_rate, segment_seconds_length, vad_aggressiveness, min_fragment_len", [
    ("path/to/video.mov", 32_000, 0.03, 0, 300),  # Sample test case
])


def test_load_audio_from_video(file_path, sample_rate):
    # Call the function to load audio
    audio = load_audio_from_video(file_path, sample_rate)
    
    # Assertions
    assert isinstance(audio, np.ndarray)  # Check if output is numpy array
    assert audio.ndim == 1  # Check if output is mono
    assert len(audio) > 0  # Check if audio data is loaded
    assert np.amin(audio) >= -1.0 and np.amax(audio) <= 1.0  # Check audio range


def test_get_segments_for_vad(file_path, sample_rate, segment_seconds_length):
    # Call the function to load audio
    audio = load_audio_from_video(file_path, sample_rate)
    # Call the function to get segments
    segments = get_segments_for_vad(audio, sample_rate, segment_seconds_length)
    
    # Assertions
    assert isinstance(segments, list)  # Check if output is a list
    assert all(isinstance(segment, np.ndarray) for segment in segments)  # Check if segments are numpy arrays
    assert len(segments) > 0  # Check if segments are generated
    assert all(len(segment) == int(sample_rate * segment_seconds_length) for segment in segments)  # Check segment length


def test_get_vad_per_segment(file_path, sample_rate, vad_aggressiveness, segment_seconds_length):
    # Call the function to load audio
    audio = load_audio_from_video(file_path, sample_rate)
    # Call the function to get segments
    segments = get_segments_for_vad(audio, sample_rate, segment_seconds_length)
    # Converting segment number to the number of frames
    segment_frames_length = int(segment_seconds_length * sample_rate)
    # Call the function to get VAD results
    segments_is_speech = get_vad_per_segment(segments, vad_aggressiveness, sample_rate, segment_frames_length)

    # Assertions
    assert isinstance(segments_is_speech, np.ndarray)
    assert segments_is_speech.dtype == bool
    assert len(segments_is_speech) == len(segments)
    assert np.array_equal(segments_is_speech, [True if i % 2 == 0 else False for i in range(len(segments))])


def test_get_frame_segments_from_vad_output(file_path, sample_rate, vad_aggressiveness, segment_seconds_length, min_fragment_len):
    # Call the function to load audio
    audio = load_audio_from_video(file_path, sample_rate)
    # Call the function to get segments
    segments = get_segments_for_vad(audio, sample_rate, segment_seconds_length)
    # Converting segment number to the number of frames
    segment_frames_length = int(segment_seconds_length * sample_rate)
    # Call the function to get VAD results
    speech_array = get_vad_per_segment(segments, vad_aggressiveness, sample_rate, segment_frames_length)
    # get full audio length in frames
    full_audio_length_frames = len(audio)

    result = get_frame_segments_from_vad_output(
        speech_array,
        sample_rate,
        min_fragment_len,
        segment_seconds_length,
        full_audio_length_frames
    )

    # Assertions
    assert isinstance(result, list)  # Check if the result is a list
    assert all(isinstance(segment, tuple) for segment in result)  # Check if all elements are tuples
    assert all(len(segment) == 2 for segment in result)  # Check if all tuples have length 2

@pytest.mark.parametrize("use_fp16, transcript_model_size" [
    (True, "large"),  
])

def test_transcribe_translate_fragments(file_path, sample_rate, vad_aggressiveness, segment_seconds_length, min_fragment_len, use_fp16, transcript_model_size):
    # Call the function to load audio
    audio = load_audio_from_video(file_path, sample_rate)
    # Call the function to get segments
    segments = get_segments_for_vad(audio, sample_rate, segment_seconds_length)
    # Converting segment number to the number of frames
    segment_frames_length = int(segment_seconds_length * sample_rate)
    # Call the function to get VAD results
    speech_array = get_vad_per_segment(segments, vad_aggressiveness, sample_rate, segment_frames_length)
    # get full audio length in frames
    full_audio_length_frames = len(audio)

    cut_fragments_frames = get_frame_segments_from_vad_output(
        speech_array,
        sample_rate,
        min_fragment_len,
        segment_seconds_length,
        full_audio_length_frames
    )

    result = transcribe_translate_fragments(
        audio=audio,
        cut_fragments_frames=cut_fragments_frames,
        sample_rate=sample_rate,
        use_fp16=use_fp16,
        transcription_model_size=transcript_model_size
    )

    expected_columns = [
        "episode",
        "segment",
        "segment_start_seconds",
        "segment_end_seconds",
        "sentence"
    ]

    # Assertions
    assert isinstance(result, pd.DataFrame)  # Check if the result is a DataFrame
    assert result.shape[1] == len(expected_columns)  # Check if the number of columns matches
    assert all(col in result.columns for col in expected_columns)  # Check if all expected columns are present

  
if __name__ == '__main__':
    pytest.main()
