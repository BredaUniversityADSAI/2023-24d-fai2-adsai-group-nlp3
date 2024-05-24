import pytest
import logging
from matplotlib import pyplot as plt
from unittest.mock import patch
from model_output_information import plot_emotion_distribution, calculate_episode_confidence

# To handle the logger's file handlers during tests to avoid creating files
logging.getLogger().handlers = []

@pytest.fixture(autouse=True)
def run_around_tests():
    """
    Fixture to run before and after each test.
    Ensures that all plots are closed before each test.
    Author - Panna Pfandler
    """
    plt.close('all')  # Close all plots before running a test
    yield
    # Code that will run after each test

def test_plot_emotion_distribution_empty_list():
    """
    Test that plot_emotion_distribution logs a warning
    when given an empty list.
    Author - Panna Pfandler
    """
    with patch("model_output_information.logging") as mock_logging:
        plot_emotion_distribution([])
        mock_logging.warning.assert_called_with("No emotions to plot, the input list is empty.")

def test_plot_emotion_distribution():
    """
    Test that plot_emotion_distribution creates a plot
    when given a non-empty list of emotions.
    Author - Panna Pfandler
    """
    emotions = ['happiness', 'sadness', 'anger', 'happiness', 'happiness', 'sadness']
    with patch("model_output_information.plt.show"):
        plot_emotion_distribution(emotions)
    # Check if the plot is created
    assert plt.gcf().number > 0

def test_calculate_episode_confidence_empty_list():
    """
    Test that calculate_episode_confidence logs a warning
    and returns 0.0 when given an empty list of scores.
    Author - Panna Pfandler
    """
    with patch("model_output_information.logging") as mock_logging:
        result = calculate_episode_confidence([])
        mock_logging.warning.assert_called_with("Received an empty list of scores.")
        assert result == 0.0

def test_calculate_episode_confidence():
    """
    Test that calculate_episode_confidence returns
    the correct average confidence score.
    Author - Panna Pfandler
    """
    scores = [0.9, 0.8, 0.95, 0.85]
    result = calculate_episode_confidence(scores)
    assert result == pytest.approx(0.875, 0.01)

def test_calculate_episode_confidence_single_value():
    """
    Test that calculate_episode_confidence returns
    the score itself when given a single score.
    Author - Panna Pfandler
    """
    scores = [0.8]
    result = calculate_episode_confidence(scores)
    assert result == 0.8

if __name__ == "__main__":
    pytest.main()
