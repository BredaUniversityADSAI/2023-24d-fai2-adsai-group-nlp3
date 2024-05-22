import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from visual import plot_emotion_distribution

class TestPlotEmotionDistribution(unittest.TestCase):
    """
    Unit tests for the 'plot_emotion_distribution' function from the 'visual' module.
    
    This class includes tests to ensure that the function handles various inputs correctly and
    produces the expected visual output indications via matplotlib's plotting functions.
    """

    @patch('matplotlib.pyplot.show')
    def test_empty_input(self, mock_show):
        """
        Test that 'plot_emotion_distribution' handles an empty list input correctly.

        An empty list should trigger a warning log and prevent any plot from being displayed.
        This test ensures that the function logs the appropriate warning and that no plot is shown.
        """
        with self.assertLogs(level='WARNING') as log:
            plot_emotion_distribution([])
            # Verify that the appropriate warning was logged
            self.assertIn('No emotions to plot, the input list is empty.', log.output[0])
            # Ensure that matplotlib's show function was not called, as there's nothing to plot
            mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_single_emotion(self, mock_show):
        """
        Test that 'plot_emotion_distribution' handles a single emotion correctly.

        This test provides a list of identical emotion strings to see if the function can correctly
        generate a plot with a single category. The matplotlib show function being called is checked
        to ensure that a plot is attempted.
        """
        predictions = ['happiness'] * 5
        plot_emotion_distribution(predictions)
        # Check that matplotlib's show function was called, indicating that a plot was made
        self.assertTrue(mock_show.called)

    @patch('matplotlib.pyplot.show')
    def test_multiple_emotions(self, mock_show):
        """
        Test that 'plot_emotion_distribution' correctly plots multiple different emotions.

        This test uses a list of various emotions to verify if the function can handle and
        correctly plot a pie chart with multiple categories. The test ensures that matplotlib's
        show function is called, indicating that the function operates correctly across diverse inputs.
        """
        predictions = ['happiness', 'sadness', 'anger', 'happiness', 'sadness']
        plot_emotion_distribution(predictions)
        # Check that matplotlib's show function was called
        self.assertTrue(mock_show.called)

if __name__ == '__main__':
    unittest.main()

