import os
import whisper
from moviepy.editor import *
from moviepy.editor import VideoFileClip


# Function to extract audio from a video file
def extract_audio(video_path, audio_path):
    # Load the video file
    video = VideoFileClip(video_path)
    # Extract the audio
    audio = video.audio
    # Save the audio to the specified path
    audio.write_audiofile(audio_path)
    # Close the video and audio files
    video.close()
    audio.close()

import re
def episode_number(file_path):
    pattern = r'ER\d+_AFL(\d+)_MXF\.mov'

    # Use re.search() to find the matching pattern
    match = re.search(pattern, file_path)

    # If match is found, extract the number
    if match:
        number = int(match.group(1))
        return number
    else:
        return None

def extract_audio_segments(audio_path, start_times, end_times, output_directory):
    # Loop through the start and end times to slice the audio
    for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        # Define the output audio file path for this segment
        output_path = f"{output_directory}/segment_{i+1}.mp3"
        # Load the audio file
        audio = AudioFileClip(audio_path)
        
        # Slice the audio
        audio_segment = audio.subclip(start_time, end_time)
        
        # Save the audio segment
        audio_segment.write_audiofile(output_path)
        
        # Close the audio segment
        audio_segment.close()

    print(f"Audio segments saved to {output_directory} successfully.")

# Function to transcribe audio segments and save the results to text files
def transcribe_segments(folder_path, model, language='en'):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            # Construct the full path to the audio file
            audio_path = os.path.join(folder_path, filename)
            
            # Transcribe the audio segment
            result = model.transcribe(audio_path, language=language)
            
            # Extract the text from the result
            text = result["text"]
            
            # Save the text to a text file
            output_file = f"{os.path.splitext(audio_path)[0]}.txt"
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(text)

def video_to_segments(file_path, df_structure):
    last_slash_index = file_path.rfind('/')
    file_directory = file_path[:last_slash_index + 1]
    #print(file_directory)

    audio_path= file_path.replace('MXF.mov', 'audio.mp3')
    extract_audio(file_path, audio_path)

    ep_number=episode_number(file_path)

    episode_df = df_structure[df_structure['Episode name'] == ep_number]
 
    # Get unique values in the specified columns
    start_times = episode_df['Start Time (seconds)'].unique()
    end_times = episode_df['End Time (seconds)'].unique()

    segment_times = list(start_times[1:]) + [end_times[-1]]

    output_directory=f'{file_directory}segments'
    #print(output_directory)
    extract_audio_segments(audio_path, start_times, segment_times, output_directory)



