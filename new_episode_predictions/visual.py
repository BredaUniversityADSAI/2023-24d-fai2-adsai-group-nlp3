import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# Configure logging to save logs to a file as well as print them to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("emotion_distribution.log"),
                        logging.StreamHandler()
                    ])

def plot_emotion_distribution(predictions: list[str]) -> None:
    """
    Plot a pie chart of the overall emotion distribution based on a list of predicted
    emotions for each sentence in an episode.

    Parameters:
    predictions (list of str): A list of predicted emotions for each sentence within an episode
                               (e.g., ['happiness', 'sadness', ...]).

    Returns:
    None: The function plots a pie chart that illustrates the overall emotion distribution in
          the episode and does not return any value.
    """
    logging.info("Starting to plot emotion distribution.")
    
    # Count each predicted emotion
    emotion_counts = Counter(predictions)
    if not emotion_counts:
        logging.warning("No emotions to plot, the input list is empty.")
        return

    # Identify the most dominant emotion
    dominant_emotion, dominant_count = emotion_counts.most_common(1)[0]

    # Prepare data for the pie chart
    labels = list(emotion_counts.keys())
    sizes = [count / sum(emotion_counts.values()) for count in emotion_counts.values()]

    # Define a range of blue colors
    blues = plt.get_cmap('Blues')
    
    # Generate color intensities based on the number of emotions
    color_shades = blues(np.linspace(0.2, 0.8, len(emotion_counts)))
    
    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       startangle=90, colors=color_shades, 
                                       textprops={'color':"black"})

    # Highlight the most dominant emotion
    for text, auto_text in zip(texts, autotexts):
        if text.get_text() == dominant_emotion:
            text.set_color('red')
            text.set_fontweight('bold')
            text.set_fontsize(12)
            auto_text.set_color('red')
            auto_text.set_fontweight('bold')
            auto_text.set_fontsize(12)

    plt.axis('equal')  # Ensures the pie chart is a circle.
    plt.title('Overall Emotion Distribution in the Episode')
    plt.show()

    logging.info("Successfully plotted the emotion distribution.")
