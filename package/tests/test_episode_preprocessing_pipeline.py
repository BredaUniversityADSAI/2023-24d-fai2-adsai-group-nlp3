import pytest
import numpy as np
import pandas as pd
import os

# Loading main functions
from e3k.episode_preprocessing_pipeline import (
    load_audio,
    get_segments_for_vad,
    get_vad_per_segment,
    get_frame_segments_from_vad_output,
    transcribe_translate_fragments,
    save_data
)

# Loading utils functions
from e3k.episode_preprocessing_pipeline import (
    segment_number_to_frames,
    get_target_length_frames,
    adjust_fragment_start_frame,
    adjust_fragment_end_frame,
    clean_transcript_df
)

# Fixture to provide sample data
@pytest.fixture
def sample_file_path():
    return "tests/test_data/video_test.mov"

class TestMain:
    @pytest.mark.parametrize("sample_rate", [32000])
    def test_load_audio(self, sample_file_path, sample_rate):
        """
        Test the `load_audio` function to ensure it correctly loads audio data.
        Asserts:
            The loaded audio should be a numpy array.
            The loaded audio should be mono (1-dimensional).
            The loaded audio data should be non-empty.

        Author: Kornelia Flizik
        """

        # Call the function to load audio
        audio = load_audio(sample_file_path, sample_rate)
        
        # Assertions
        # Check if output is numpy array
        assert isinstance(audio, np.ndarray)
        # Check if output is mono
        assert audio.ndim == 1
        # Check if audio data is loaded
        assert len(audio) > 0

    @pytest.mark.parametrize("sample_rate, segment_seconds_length", [(32000, 0.03)])
    def test_get_segments_for_vad(self, sample_file_path,
                                  sample_rate, segment_seconds_length):
        
        """
        Test the `get_segments_for_vad` function to ensure it correctly
        segments audio data.
        Asserts:
            The segments should be a list.
            Each segment should be a numpy array.
            Segments should be generated (non-empty list).
            Each segment should have the correct length.

        Author: Kornelia Flizik
        """

        # Call the function to load audio
        audio = load_audio(sample_file_path, sample_rate)
        # Call the function to get segments
        segments = get_segments_for_vad(audio, sample_rate, segment_seconds_length)
        
        # Assertions
        # Check if output is a list
        assert isinstance(segments, list)
        # Check if segments are numpy arrays
        assert all(isinstance(segment, np.ndarray) for segment in segments)
        # Check if segments are generated
        assert len(segments) > 0
        # Check segment length
        assert all(len(segment) == int(sample_rate * segment_seconds_length)
                   for segment in segments)

    @pytest.mark.parametrize("sample_rate, vad_aggressiveness, segment_seconds_length",
                             [(32000, 0, 0.03)])
    def test_get_vad_per_segment(self, sample_file_path, sample_rate,
                                 vad_aggressiveness, segment_seconds_length):
        
        """
        Test the `get_vad_per_segment` function to ensure it correctly
        applies VAD to segments.
        Asserts:
            The VAD output should be a numpy array.
            The VAD output should be of boolean type.
            The length of the VAD output should match the number of segments.

        Author: Kornelia Flizik
        """

        # Call the function to load audio
        audio = load_audio(sample_file_path, sample_rate)
        # Call the function to get segments
        segments = get_segments_for_vad(audio, sample_rate, segment_seconds_length)
        # Converting segment number to the number of frames
        segment_frames_length = int(segment_seconds_length * sample_rate)
        # Call the function to get VAD results
        segments_is_speech = get_vad_per_segment(segments, vad_aggressiveness,
                                                sample_rate, segment_frames_length)

        # Assertions
        # Check if output is numpy array
        assert isinstance(segments_is_speech, np.ndarray)
        # Check if output is bool
        assert segments_is_speech.dtype == bool 
        # Output array should have the same length as input segments
        assert len(segments_is_speech) == len(segments) 

    @pytest.mark.parametrize("sample_rate, vad_aggressiveness, segment_seconds_length,\
                              min_fragment_len", [(32000, 0, 0.03, 300)])
    def test_get_frame_segments_from_vad_output(self, sample_file_path,
                                                sample_rate,
                                                vad_aggressiveness,
                                                segment_seconds_length, 
                                                min_fragment_len):
        
        """
        Test the `get_frame_segments_from_vad_output` function to ensure it correctly
        processes VAD output into frame segments.
        Asserts:
            The result should be a list.
            Each element of the list should be a tuple.
            Each tuple should have a length of 2.

        Author: Kornelia Flizik
        """

        # Call the function to load audio
        audio = load_audio(sample_file_path, sample_rate)
        # Call the function to get segments
        segments = get_segments_for_vad(audio, sample_rate, segment_seconds_length)
        # Converting segment number to the number of frames
        segment_frames_length = int(segment_seconds_length * sample_rate)
        # Call the function to get VAD results
        speech_array = get_vad_per_segment(segments, vad_aggressiveness, sample_rate,
                                           segment_frames_length)
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
        # Check if the result is a list
        assert isinstance(result, list)
        # Check if all elements are tuples
        assert all(isinstance(segment, tuple) for segment in result)
        # Check if all tuples have length 2
        assert all(len(segment) == 2 for segment in result)

    @pytest.mark.parametrize("sample_rate, vad_aggressiveness, segment_seconds_length,\
                            min_fragment_len, use_fp16, transcript_model_size,\
                            output_path", 
                            [(32000, 0, 0.03, 300, True, "tiny", "output.csv")])
    def test_transcribe_translate_fragments_and_save(self, sample_file_path,
                                                    sample_rate, vad_aggressiveness,
                                                    segment_seconds_length,
                                                    min_fragment_len, use_fp16,
                                                    transcript_model_size, 
                                                    output_path):
        
        """
        Test the `transcribe_translate_fragments_and_save` function to ensure it
        correctly processes audio and saves the transcription.
        Asserts:
            The result should be a DataFrame.
            The DataFrame should have the expected columns.
            The output file should exist after saving.

        Author: Kornelia Flizik
        """

        # Call the function to load audio
        audio = load_audio(sample_file_path, sample_rate)
        # Call the function to get segments
        segments = get_segments_for_vad(audio, sample_rate, segment_seconds_length)
        # Converting segment number to the number of frames
        segment_frames_length = int(segment_seconds_length * sample_rate)
        # Call the function to get VAD results
        speech_array = get_vad_per_segment(segments, vad_aggressiveness, sample_rate,
                                           segment_frames_length)
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
        # Check if the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Check if the number of columns matches
        assert result.shape[1] == len(expected_columns)
        # Check if all expected columns are present
        assert all(col in result.columns for col in expected_columns)

        # Save the result DataFrame using save_data
        save_data(result, output_path)
        # Check if the output file exists
        assert os.path.exists(output_path)
        # Clean up the test output files
        #  os.remove(output_path)

    @pytest.mark.parametrize("segment_number, sample_rate, segment_seconds_length",
                              [(1, 32000, 0.03)])
    def test_segment_number_to_frames(self, segment_number, sample_rate,
                                      segment_seconds_length):
        """
        Test the `segment_number_to_frames` function to ensure it correctly
        converts segment number to frames.
        Asserts: The result should match the expected frame count.

        Author: Kornelia Flizik
        """

        # Call the function to segment number to frames
        result = segment_number_to_frames(segment_number, sample_rate,
                                          segment_seconds_length)
        # Assertions
        assert result == 960

    @pytest.mark.parametrize("sample_rate, min_fragment_len", [(32000, 300)])
    def test_get_target_length_frames(self, sample_rate, min_fragment_len):
        """
        Test the `get_target_length_frames` function to ensure it correctly
        computes target length in frames.
        Asserts: The result should match the expected target length in frames.

        Author: Kornelia Flizik
        """

        # Call the function to get frames length
        result = get_target_length_frames(min_fragment_len, sample_rate)
        # Assertions
        assert result == 96_000_00

    @pytest.mark.parametrize("sample_rate, start_fragment_frame", [(32000, 4000)])
    def test_adjust_fragment_start_frame(self, sample_rate, start_fragment_frame):
        """
        Test the `adjust_fragment_start_frame` function to ensure it correctly
        adjusts the start frame of a fragment.
        Asserts: The result should match the expected adjusted start frame.

        Author: Kornelia Flizik
        """
 
        result = adjust_fragment_start_frame(start_fragment_frame, sample_rate)
        # Assertions
        assert result == 0

    @pytest.mark.parametrize("sample_rate, end_fragment_frame, \
                             full_audio_length_frames", [(32000, 32000, 64000)])
    def test_adjust_fragment_end_frame(self, sample_rate, end_fragment_frame,
                                       full_audio_length_frames):
        """
        Test the `adjust_fragment_end_frame` function to ensure it correctly
        adjusts the end frame of a fragment.
        Asserts: The result should match the expected adjusted end frame.

        Author: Kornelia Flizik
        """

        result = adjust_fragment_end_frame(end_fragment_frame, sample_rate,
                                           full_audio_length_frames)
        # Assertions
        assert result == 36000

    @pytest.mark.parametrize("sample_rate", [32000])
    def test_clean_transcript_df(self, sample_rate):
        sample_df = pd.DataFrame({"index": [1, 2, 3], 
                                  "start": [312, 485, 645], 
                                  "end": [484, 644, 823], 
                                  "text": ["hello", "darkness", "my old friend"]})
        """
        Test the `clean_transcript_df` function to ensure it correctly cleans
        the transcript DataFrame.
        Asserts:
            The cleaned DataFrame should have the expected columns.
            The number of columns should match the expected number.
            All expected columns should be present in the cleaned DataFrame.

        Author: Kornelia Flizik
        """

        result = clean_transcript_df(df=sample_df, sample_rate=sample_rate)

        expected_columns = [
            "episode",
            "segment",
            "segment_start_seconds",
            "segment_end_seconds",
            "sentence"
        ]

        # Assertions
        # Check if the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Check if the number of columns matches
        assert result.shape[1] == len(expected_columns)
        # Check if all expected columns are present
        assert all

if __name__ == '__main__':
    pytest.main()
