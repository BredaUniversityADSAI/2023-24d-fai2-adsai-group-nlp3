import unittest

from confidence_score import calculate_episode_confidence


class TestCalculateEpisodeConfidence(unittest.TestCase):
    def test_empty_scores(self):
        """Test that the function returns 0.0 when input list is empty."""
        self.assertEqual(
            calculate_episode_confidence([]),
            0.0,
            "Should return 0.0 for empty score list",
        )

    def test_correct_average_calculation(self):
        """Test that the function returns correct average for a list of scores."""
        scores = [0.5, 0.75, 0.9]
        expected_average = (0.5 + 0.75 + 0.9) / 3
        self.assertAlmostEqual(
            calculate_episode_confidence(scores),
            expected_average,
            msg="Should return correct average",
        )


if __name__ == "__main__":
    unittest.main()
