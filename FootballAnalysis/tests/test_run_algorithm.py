import unittest
from unittest import mock

from run_algorithm import *


class TestRunAlgorithmModule(unittest.TestCase):

    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_get_players_team_None(self):
        team1_color_index = 1
        team2_color_index = 3

        result = get_players_team(self.frame, team1_color_index, team2_color_index)

        self.assertIsNone(result)


class TestGameAnalyzer(unittest.TestCase):
    def setUp(self):
        self.model = mock.Mock()
        self.args = mock.Mock()
        self.analyzer = GameAnalyzer(self.model, self.args)

    def test_compute_possession(self):
        scheduler = mock.Mock()

        self.analyzer.close_calculator = False
        self.analyzer.current_possession_team = None
        self.analyzer.total_time_secs = 10
        self.analyzer.team1_secs = 5
        self.analyzer.team2_secs = 5

        self.analyzer.compute_possession(scheduler)

        self.assertEqual(scheduler.enter.call_count, 1)
        self.assertEqual(scheduler.enter.call_args[0][0], 1)
        self.assertEqual(scheduler.enter.call_args[0][1], 1)
        self.assertEqual(scheduler.enter.call_args[0][2], self.analyzer.compute_possession)
        self.assertEqual(scheduler.enter.call_args[0][3], (scheduler,))

        self.assertEqual(self.analyzer.team1_poss_str, "50%")
        self.assertEqual(self.analyzer.team2_poss_str, "50%")

        self.analyzer.current_possession_team = 1
        self.analyzer.compute_possession(scheduler)

        self.assertEqual(self.analyzer.total_time_secs, 11)
        self.assertEqual(self.analyzer.team1_secs, 6)
        self.assertEqual(self.analyzer.team2_secs, 5)
        self.assertEqual(self.analyzer.team1_poss_str, "55%")
        self.assertEqual(self.analyzer.team2_poss_str, "45%")

        self.analyzer.current_possession_team = 2
        self.analyzer.compute_possession(scheduler)

        self.assertEqual(self.analyzer.total_time_secs, 12)
        self.assertEqual(self.analyzer.team1_secs, 6)
        self.assertEqual(self.analyzer.team2_secs, 6)
        self.assertEqual(self.analyzer.team1_poss_str, "50%")
        self.assertEqual(self.analyzer.team2_poss_str, "50%")

    def test_get_most_dominant_colors_for_players_in_last_frames(self):
        players_detections = [
            mock.Mock(data={"id": 1, "color_index": 3}),
            mock.Mock(data={"id": 2, "color_index": 2}),
            mock.Mock(data={"id": 1, "color_index": 3}),
            mock.Mock(data={"id": 3, "color_index": 1})
        ]

        self.analyzer.players_colors = {1: [0, 1], 2: [2, 3], 3: [4, 5]}

        self.analyzer.get_most_dominant_colors_for_players_in_last_frames(players_detections)

        self.assertEqual(self.analyzer.players_colors, {1: [0, 1, 3, 3], 2: [2, 3, 2], 3: [4, 5, 1]})


if __name__ == '__main__':
    unittest.main()
