import unittest
from unittest.mock import MagicMock

from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtWidgets import QApplication, QMessageBox

from football_analysis import MainWindow


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a QApplication instance before running the tests
        cls.app = QApplication([])

    def setUp(self):
        # Create an instance of the MainWindow for each test
        self.window = MainWindow()

    def tearDown(self):
        # Close and destroy the MainWindow instance after each test
        self.window.close()
        self.window = None

    @classmethod
    def tearDownClass(cls):
        # Destroy the QApplication instance after running the tests
        del cls.app

    def test_center_window(self):
        # Simulate calling the center_window method
        self.window.center_window()

        # Check if the window is correctly centered
        screen_geometry = QApplication.desktop().screenGeometry()
        window_geometry = self.window.geometry()
        expected_x = (screen_geometry.width() - window_geometry.width()) // 2
        expected_y = (screen_geometry.height() - window_geometry.height()) // 2
        self.assertEqual(expected_x, self.window.x())
        self.assertEqual(expected_y, self.window.y())

    def test_slider_moved(self):
        # Simulate calling the slider_moved method with a position value
        position = 500
        self.window.video_slider.setRange(0, 1000)
        self.window.slider_moved(position)

        # Check if the video slider value is correctly set
        self.assertEqual(position, self.window.video_slider.value())
        self.window.video_slider.setRange(0, 0)

    def test_duration_changed(self):
        # Simulate calling the duration_changed method with a duration value
        duration = 10000
        self.window.duration_changed(duration)

        # Check if the video slider range is correctly set
        self.assertEqual(0, self.window.video_slider.minimum())
        self.assertEqual(duration, self.window.video_slider.maximum())

    def test_set_slider_pos(self):
        # Simulate calling the set_slider_pos method with a position value
        position = 5000
        self.window.set_slider_pos(position)

        # Check if the media player position is correctly set
        self.assertEqual(position, self.window.media_player.position())

    def test_handle_error(self):
        error_code = QMediaPlayer.Error.FormatError
        error_string = "Format error occurred."

        # Mock the QMessageBox.critical method
        QMessageBox.critical = MagicMock()

        # Call the handle_error method
        self.window.handle_error(error_code)

        # Assert that the QMessageBox.critical method was called with the expected arguments
        QMessageBox.critical.assert_called_with(self.window, "Error", "Failed to play the video: " + error_string)
        self.assertFalse(self.window.play_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
