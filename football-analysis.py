import sys

from threading import Thread

import FootAndBall.network.footandball as footandball

import torch.cuda
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QPushButton, QSlider, QStyle, QVBoxLayout, \
    QWidget, QMessageBox, QProgressDialog, QLabel, QProgressBar, QComboBox

from run_algorithm import *


def run_algorithm(video_path, output_path, device_type, team1_color, team2_color):
    print('Running FootballAnalysis algorithm on input video')
    args = ModelArgs(
        path=video_path,
        model='fb1',
        weights=f"models/model_20230310_1416_final.pth",
        ball_threshold=0.01,
        player_threshold=0.7,
        out_video=output_path,
        device=device_type,
        team1_color=team1_color,
        team2_color=team2_color
    )

    model = footandball.model_factory(args.model, 'detect', ball_threshold=args.ball_threshold,
                                      player_threshold=args.player_threshold)

    analyzer = GameAnalyzer(model, args)
    analyzer.run()


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        upload_button = QPushButton("Upload Video")
        upload_button.setStyleSheet("font-size: 12px;")
        upload_button.clicked.connect(self.upload_file)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)

        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setRange(0, 0)
        self.video_slider.sliderMoved.connect(self.set_slider_pos)

        video_play_layout = QHBoxLayout()
        video_play_layout.setContentsMargins(0, 0, 0, 0)
        video_play_layout.addWidget(self.play_button)
        video_play_layout.addWidget(self.video_slider)

        self.team1_combo_box = QComboBox()
        self.team1_combo_box.addItem("White")
        self.team1_combo_box.addItem("Dark Blue")
        self.team1_combo_box.addItem("Light Blue")
        self.team1_combo_box.addItem("Red")
        self.team1_combo_box.addItem("Black")
        self.team1_combo_box.addItem("Orange")
        self.team1_combo_box.addItem("Yellow")
        self.team1_combo_box.addItem("Purple")

        self.team2_combo_box = QComboBox()
        self.team2_combo_box.addItem("White")
        self.team2_combo_box.addItem("Dark Blue")
        self.team2_combo_box.addItem("Light Blue")
        self.team2_combo_box.addItem("Red")
        self.team2_combo_box.addItem("Black")
        self.team2_combo_box.addItem("Orange")
        self.team2_combo_box.addItem("Yellow")
        self.team2_combo_box.addItem("Purple")
        self.team2_combo_box.setCurrentIndex(1)

        combo_box_layout = QHBoxLayout()
        combo_box_layout.setContentsMargins(0, 0, 0, 0)
        combo_box_layout.addWidget(self.team1_combo_box)
        combo_box_layout.addWidget(self.team2_combo_box)

        videoWidget = QVideoWidget()

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(videoWidget)
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.slider_moved)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.error.connect(self.handle_error)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(video_play_layout)
        layout.addWidget(upload_button)
        layout.addLayout(combo_box_layout)

        self.setLayout(layout)

        self.setWindowTitle("Football Analysis")
        self.resize(1280, 800)
        self.center_window()

    def center_window(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        window_geometry = self.geometry()

        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2

        self.move(x, y)

    def upload_file(self):
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Choose video",
                                                  ".",
                                                  "Video Files (*.mp4 *.ogg *.avi *.mkv)")

        if fileName != '':
            output_path = fileName.rsplit(".", 1)[0] + "_out.avi"

            process_dialog_label = QLabel("Processing video...")
            process_dialog_label.setAlignment(Qt.AlignCenter)

            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)
            progress_bar.setValue(0)
            progress_bar.setStyleSheet("margin-right: 0px;")

            process_dialog_layout = QVBoxLayout()
            process_dialog_layout.addWidget(process_dialog_label)
            process_dialog_layout.addWidget(progress_bar)

            progress_dialog = QProgressDialog(self)
            progress_dialog.setWindowFlags(
                Qt.Window |
                Qt.CustomizeWindowHint |
                Qt.WindowTitleHint |
                Qt.WindowStaysOnTopHint)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setLayout(process_dialog_layout)
            progress_dialog.setFixedSize(200, 100)
            progress_dialog.setWindowTitle("Processing video")
            progress_dialog.setCancelButton(None)
            progress_dialog.show()

            device_used = "cuda" if torch.cuda.is_available() else "cpu"
            team1_color = self.team1_combo_box.currentText().replace(" ", "_").lower()
            team2_color = self.team2_combo_box.currentText().replace(" ", "_").lower()

            run_algorithm_thread = Thread(target=run_algorithm, args=(fileName, output_path, device_used,
                                                                      team1_color, team2_color))
            try:
                run_algorithm_thread.start()

                while run_algorithm_thread.is_alive():
                    QApplication.processEvents()
                    time.sleep(0.1)

                progress_dialog.close()
            except Exception as err:
                QMessageBox.critical(self, "Algorithm error", str(err))

            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(output_path)))
            self.play_button.setEnabled(True)
            self.play_video()

    def play_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def media_state_changed(self, state):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def slider_moved(self, position):
        self.video_slider.setValue(position)

    def duration_changed(self, duration):
        self.video_slider.setRange(0, duration)

    def set_slider_pos(self, position):
        self.media_player.setPosition(position)

    def handle_error(self, error):
        self.play_button.setEnabled(False)
        if error == QMediaPlayer.Error.ResourceError:
            error_string = "Resource error occurred."
        elif error == QMediaPlayer.Error.FormatError:
            error_string = "Format error occurred."
        elif error == QMediaPlayer.Error.NetworkError:
            error_string = "Network error occurred."
        elif error == QMediaPlayer.Error.AccessDeniedError:
            error_string = "Access denied error occurred."
        else:
            error_string = "An unknown error occurred."

        QMessageBox.critical(self, "Error", "Failed to play the video: " + error_string)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
