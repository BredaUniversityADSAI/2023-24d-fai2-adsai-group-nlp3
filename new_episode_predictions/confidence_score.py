import logging

# Configure logging to save logs to a file as well as print them to the console
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("episode_confidence.log"),
                        logging.StreamHandler()
                    ])

def calculate_episode_confidence(highest_probabilities: list[float]) -> float:
    """
    Calculate the overall confidence score for the episode by averaging the highest probabilities 
    for each sentence.
    
    Parameters:
    scores (List[float]): A list of the highest probabilities for each sentence in an episode.
    
    Returns:
    float: The overall confidence score, calculated as the average of the highest probabilities 
    across all sentences.
    """
    if not highest_probabilities:
        logging.warning("Received an empty list of probabilities.")
        return 0.0

    average_probability = sum(highest_probabilities) / len(highest_probabilities)
    logging.info(f"Calculated average probability: {average_probability}")
    return average_probability
