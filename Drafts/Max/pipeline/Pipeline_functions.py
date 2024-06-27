import subprocess
import os
from transformers import TFRobertaForSequenceClassification
import pandas as pd
from collections import Counter

def convert_mov_to_mp3(input_directory):

    # Path to the ffmpeg executable
    ffmpeg_path = r"C:\Users\daraz\Desktop\BUas\Year 2 Block C\ffmpeg-master-latest-win64-lgpl\ffmpeg-master-latest-win64-lgpl\bin\ffmpeg.exe"
    
    # Output directory for the MP3 files
    output_directory = r"C:\Users\daraz\Desktop\BUas\Year 2 Block C\Tasks\Pipeline\mp3s"

    # List all files in the directory
    for file_name in os.listdir(input_directory):

        # Check if the file is a .mov file
        if file_name.endswith('.mov'):

            def renamer(filename):
                parts = filename.split('_')
                episode_part = parts[1]
                episode_number = int(episode_part[3:])
                new_episode_part = episode_part[:3] + str(episode_number)
                parts[1] = new_episode_part
                filename = "_".join(parts)
                return filename

            # Construct the full path to the input and output files
            input_file = os.path.join(input_directory, file_name)

            file_name = renamer(file_name)

            output_file = os.path.join(output_directory, file_name.replace('.mov', '.mp3'))
            
            # Construct the command to convert the file
            command = [ffmpeg_path, '-i', input_file, '-vn', '-ab', '128k', '-ar', '44100', '-y', output_file]
            
            # Run the command
            try:
                subprocess.run(command, check=True)
                print(f'Successfully converted {input_file} to {output_file}')
            except subprocess.CalledProcessError as e:
                print(f'Error occurred: {e}')


def predicting(input_ids, attention_masks):

        # Loading the model
        model_directory = r"C:\Users\daraz\Desktop\BUas\Year 2 Block C\Tasks\Pipeline\model"

        # Load the fine-tuned model
        model = TFRobertaForSequenceClassification.from_pretrained(model_directory)

        # Make predictions
        predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_masks})

        # Returning the predictions
        return predictions


def mp3_to_segments(audio_full, row):

    # Extracting the start and end times of the audio segment
    start = row['Start Time (seconds)']
    end = row['End Time (seconds)']
    
    # Converting to milliseconds
    start = start * 1000
    end = end * 1000

    # Extracting the audio segment
    segment = audio_full[int(start):int(end)]

    return segment


def transcription(input_directory, model):

    def translate(file_path: str) -> str:
    
        # Load and transcribe the audio file
        result = model.transcribe(file_path, task="translate")
        
        # The translated text is in the 'text' field of the result
        translated_text = result["text"]
        return translated_text
    
    def sort_key(filename):

        # Split the filename into parts
        parts = filename.split('_')

        # Extract the episode number and segment number
        episode_number = int(parts[1][3:])  # Adjust this based on your filename format
        segment_number = int(parts[3].split('.')[0])

        # Return the episode number and segment number
        return episode_number, segment_number
    
    

    # Creating 17 seperate dictionaries for each episode
    ep1 = {}
    ep2 = {}
    ep3 = {}
    ep4 = {}
    ep5 = {}
    ep6 = {}
    ep7 = {}
    ep8 = {}
    ep9 = {}
    ep10 = {}
    ep11 = {}
    ep12 = {}
    ep13 = {}
    ep14 = {}
    ep15 = {}
    ep16 = {}
    ep17 = {}

    # Get all MP3 filenames
    filenames = [f for f in os.listdir(input_directory) if f.endswith(".mp3")]

    # Sort the filenames based on the episode number and segment number
    sorted_filenames = sorted(filenames, key=sort_key)

    # Initialize a dictionary for the current episode's segments
    current_episode_segments = {}
    current_episode_number = None

    # Loop through each file in the directory
    for filename in sorted_filenames:

        # Check if the file is an MP3 file
        if filename.endswith(".mp3"):

            # Construct the full path to the file
            file_path = os.path.join(input_directory, filename)

            # Getting the episode and segment name
            parts = filename.split('_')
            episode_number = parts[1]  # This gets the {i} part
            segment_index = parts[3].split('.')[0]  # This gets the {index} part before '.mp3'
            episode_number_str = episode_number[3:]
            episode_number = int(episode_number_str)

            #if episode_number < 10:
            #    continue
            
            # Translating
            translation = translate(file_path)

            print(f"Translated {filename}")

            # Check if we've moved to a new episode
            if current_episode_number is None:
                current_episode_number = episode_number

            # New episode detected, save the current episode's data
            elif episode_number != current_episode_number:
                
                # Save the current episode's data to a CSV file
                transc_df = pd.DataFrame(list(current_episode_segments.items()), columns=['Segment', 'Translation'])
                transc_df.to_csv(f"C:/Users/daraz/Desktop/BUas/Year 2 Block C/Tasks/Pipeline/transcription_dfs/ep{current_episode_number}_translations.csv", index=False)
                print(f"Saved ep{current_episode_number}_translations.csv")
                
                # Reset for the new episode
                current_episode_segments = {}
                current_episode_number = episode_number
        
            # Add the translation to the current episode's dictionary
            current_episode_segments[segment_index] = translation

    # After the loop, save the last episode's data
    if current_episode_segments:
        transc_df = pd.DataFrame(list(current_episode_segments.items()), columns=['Segment', 'Translation'])
        transc_df.to_csv(f"C:/Users/daraz/Desktop/BUas/Year 2 Block C/Tasks/Pipeline/transcription_dfs/ep{current_episode_number}_translations.csv", index=False)
        print(f"Saved ep{current_episode_number}_translations.csv")


def post_transcription(structure, transcript, bigmerged):

    # Adding a new column to the structure dataframe that is just the row number
    structure['row_number'] = structure.index

    # Renaming the "Segment column" to "row_number" so that we can merge the two dataframes
    transcript = transcript.rename(columns={'Segment': 'row_number'})

    # Merging the two dataframes on the "row_number" column
    merged = pd.merge(structure, transcript, on='row_number')

    # Returning the merged dataframe
    return merged

    

# Creating a function to map the emotions to the basic emotions
def map_emotions_and_find_common(row):

    # Mapping the emotions to the six basic emotions
    emotion_mapping = {
        'Admiration': 'happiness',
        'Amusement': 'happiness',
        'Anger': 'anger',
        'Annoyance': 'anger',
        'Anticipation': 'surprise',
        'Approval': 'happiness',
        'Caring': 'happiness',
        'Confusion': 'surprise',
        'Curiosity': 'surprise',
        'Desire': 'happiness',
        'Disappointment': 'sadness',
        'Disapproval': 'anger',
        'Disgust': 'disgust',
        'Embarrassment': 'sadness',
        'Excitement': 'happiness',
        'Fear': 'fear',
        'Gratitude': 'happiness',
        'Grief': 'sadness',
        'Hunger': 'anger',
        'Joy': 'happiness',
        'Love': 'happiness',
        'Nervousness': 'fear',
        'Optimism': 'happiness',
        'Pride': 'happiness',
        'Realization': 'surprise',
        'Relief': 'happiness',
        'Remorse': 'sadness',
        'Sadness': 'sadness',
        'Shame': 'sadness',
        'Surprise': 'surprise'
    }

    # Split the string into a list of emotions
    emotions_list = row.split(", ")

    # Map each emotion to a basic emotion
    basic_emotions = [emotion_mapping[emotion] for emotion in emotions_list if emotion in emotion_mapping]

    # Find the most common basic emotion
    if basic_emotions:
        most_common_basic_emotion = Counter(basic_emotions).most_common(1)[0][0]
    else:
        most_common_basic_emotion = None

    # Return the most common basic emotion
    return most_common_basic_emotion