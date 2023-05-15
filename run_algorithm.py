import argparse
import os
import sched
import time
import cv2
import numpy as np
import torch
import tqdm

from threading import Thread
from typing import List
from dataclasses import dataclass

from norfair import mean_euclidean
from norfair.camera_motion import MotionEstimator
from norfair.tracker import Detection, Tracker
from norfair.tracker import TrackedObject

from run_algorithm_utils import *

import FootAndBall.data.augmentation as augmentations
import FootAndBall.network.footandball as footandball
from FootAndBall.data.augmentation import PLAYER_LABEL, BALL_LABEL

IDENTIFICATION_COLOR_LIMIT = 1500
NR_OF_DETECTIONS = 10
NR_OF_TOUCHINGS = 30


def get_players_team(frame, team1_color_index, team2_color_index):
    frame = frame.copy()
    frame = cv2.resize(frame, (250, 400))

    height, width, _ = frame.shape

    y_start = int(height * 0.275)
    y_end = int(height * 0.65)
    x_start = int(width * 0.225)
    x_end = int(width * 0.775)

    img = frame[y_start:y_end, x_start:x_end]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # WHITE Range
    lower_white = np.array([0, 0, 150], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)

    # GREEN Range
    lower_green = np.array([40, 60, 70], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)

    # DARK BLUE Range
    lower_blue_strong = np.array([110, 60, 60], dtype=np.uint8)
    upper_blue_strong = np.array([130, 255, 255], dtype=np.uint8)

    # LIGHT BLUE Range
    lower_blue_light = np.array([100, 60, 80], dtype=np.uint8)
    upper_blue_light = np.array([110, 255, 255], dtype=np.uint8)

    # RED Range
    lower_red_low = np.array([0, 60, 70], dtype=np.uint8)
    upper_red_low = np.array([10, 255, 255], dtype=np.uint8)

    # PINK Range
    lower_red_high = np.array([175, 60, 70], dtype=np.uint8)
    upper_red_high = np.array([180, 255, 255], dtype=np.uint8)

    # BLACK Range
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 40], dtype=np.uint8)

    # ORANGE Range
    lower_orange = np.array([10, 70, 70], dtype=np.uint8)
    upper_orange = np.array([20, 255, 255], dtype=np.uint8)

    # Yellow Range
    lower_yellow = np.array([25, 60, 70], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

    # PURPLE Range
    lower_purple = np.array([130, 80, 80], dtype=np.uint8)
    upper_purple = np.array([160, 255, 255], dtype=np.uint8)

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue_strong = cv2.inRange(hsv, lower_blue_strong, upper_blue_strong)
    mask_blue_light = cv2.inRange(hsv, lower_blue_light, upper_blue_light)
    mask_red_low = cv2.inRange(hsv, lower_red_low, upper_red_low)
    mask_red_high = cv2.inRange(hsv, lower_red_high, upper_red_high)
    mask_red = cv2.bitwise_or(mask_red_low, mask_red_high)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    nr_of_zeros = [0, 0, 0, 0, 0, 0, 0, 0]

    res_green = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_green))

    nr_of_zeros[0] = get_nr_of_zeros(mask_white, res_green)
    nr_of_zeros[1] = get_nr_of_zeros(mask_blue_strong, res_green)
    nr_of_zeros[2] = get_nr_of_zeros(mask_blue_light, res_green)
    nr_of_zeros[3] = get_nr_of_zeros(mask_red, res_green)
    nr_of_zeros[4] = get_nr_of_zeros(mask_black, res_green)
    nr_of_zeros[5] = get_nr_of_zeros(mask_orange, res_green)
    nr_of_zeros[6] = get_nr_of_zeros(mask_yellow, res_green)
    nr_of_zeros[7] = get_nr_of_zeros(mask_purple, res_green)

    max_zeroes = 0
    max_index = 0

    for i in range(len(nr_of_zeros)):
        if nr_of_zeros[i] > max_zeroes and i in [team1_color_index, team2_color_index]:
            max_zeroes = nr_of_zeros[i]
            max_index = i

    if max_zeroes > IDENTIFICATION_COLOR_LIMIT:
        return max_index

    return None


@dataclass
class Ball:
    x: int
    y: int
    radius: int
    color: (int, int, int)
    prediction_score: int


class GameAnalyzer:
    def __init__(self, model, args):
        self.team1_secs = 0
        self.team2_secs = 0
        self.total_time_secs = 0
        self.team1_poss_str = ""
        self.team2_poss_str = ""
        self.players_colors = {}
        self.current_possession_player = None
        self.current_possession_team = None
        self.close_calculator = False
        self.last_player_touching = []
        self.color_index_mappings = {"white": 0, "dark_blue": 1, "light_blue": 2, "red": 3, "black": 4,
                                     "orange": 5, "yellow": 6, "purple": 7}
        self.players_tracker = Tracker(distance_function=mean_euclidean,
                                       distance_threshold=250,
                                       initialization_delay=3,
                                       hit_counter_max=90)
        self.colors = [(255, 255, 255), (255, 0, 0), (128, 0, 0), (0, 0, 255), (0, 0, 0), (0, 165, 255), (0, 128, 128),
                       (128, 0, 128)]
        self.__model = model
        self.__args = args
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2

    def compute_possession(self, scheduler):
        if self.close_calculator:
            return
        scheduler.enter(1, 1, self.compute_possession, (scheduler,))
        if self.current_possession_team is None:
            if self.total_time_secs > 0:
                self.team1_poss_str = "{}%".format(round(100 * (self.team1_secs / self.total_time_secs)))
                self.team2_poss_str = "{}%".format(round(100 * (self.team2_secs / self.total_time_secs)))
        elif self.current_possession_team == 1:
            self.total_time_secs += 1
            self.team1_secs += 1
            self.team1_poss_str = "{}%".format(round(100 * (self.team1_secs / self.total_time_secs)))
            self.team2_poss_str = "{}%".format(round(100 * (self.team2_secs / self.total_time_secs)))
        else:
            self.total_time_secs += 1
            self.team2_secs += 1
            self.team1_poss_str = "{}%".format(round(100 * (self.team1_secs / self.total_time_secs)))
            self.team2_poss_str = "{}%".format(round(100 * (self.team2_secs / self.total_time_secs)))

    def get_most_dominant_colors_for_players_in_last_frames(self, players_detections):
        for detection in players_detections:
            if detection.data["id"] in players_detections and len(
                    self.players_colors[detection.data["id"]]) > NR_OF_DETECTIONS:
                self.players_colors[detection.data["id"]].pop(0)

            if detection.data["id"] not in self.players_colors:
                self.players_colors[detection.data["id"]] = []

            self.players_colors[detection.data["id"]].append(detection.data["color_index"])

    def print_possession(self, frame, id, x_left, y_top, x_right, y_bottom, team1_color, team2_color, color_index):
        possession_text = self.team1_poss_str + " - " + self.team2_poss_str
        (possession_text_width, possession_text_height), _ = cv2.getTextSize(possession_text, self.font,
                                                                             self.font_scale,
                                                                             self.font_thickness)
        possession_text_x = int((frame.shape[1] - possession_text_width) / 2)
        possession_text_y = possession_text_height + 10

        cv2.putText(frame, possession_text, (possession_text_x, possession_text_y), self.font, self.font_scale,
                    (255, 255, 255), self.font_thickness)

        # cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), self.colors[color_index], 2)
        # cv2.putText(frame, self.team1_poss_str, (x_left, max(0, y_top - 70)), self.font, self.font_scale,
        #             self.colors[self.color_index_mappings[team1_color]], self.font_thickness)
        # cv2.putText(frame, self.team2_poss_str, (int(x_left), max(0, int(y_top) - 110)), self.font, self.font_scale,
        #             self.colors[self.color_index_mappings[team2_color]], self.font_thickness)
        # cv2.putText(frame,
        #             "YES" if id == self.current_possession_player else "NO",
        #             (int(x_left),
        #              max(0, int(y_top) - 30)),
        #             self.font,
        #             self.font_scale,
        #             self.colors[color_index],
        #             self.font_thickness)

    def check_team_in_possession(self, id, ball, x_left, y_top, x_right, y_bottom, color_index, team1_color,
                                 team2_color):
        if ball is not None:
            has_ball = check_intersection(ball.x, ball.y, x_left, x_right, y_top, y_bottom)
            if has_ball and id != self.current_possession_player:
                if len(self.last_player_touching) < NR_OF_TOUCHINGS:
                    self.last_player_touching.append(id)
                    self.current_possession_player = id
                    self.current_possession_team = 1 if color_index == self.color_index_mappings[team1_color] \
                        else 2
                else:
                    self.last_player_touching.pop(0)
                    self.last_player_touching.append(id)
                    if self.current_possession_player != id and check_last_touchings_equal(
                            self.last_player_touching) and self.last_player_touching[0] == id:
                        self.current_possession_player = id
                        self.current_possession_team = 1 if color_index == self.color_index_mappings[
                            team1_color] else 2

    def print_ball(self, frame, ball):
        if ball is not None:
            cv2.circle(frame, (int(ball.x), int(ball.y)), ball.radius, ball.color, 2)
            cv2.putText(frame, '{:0.2f}'.format(ball.prediction_score),
                        (max(0, int(ball.x - ball.radius)), max(0, (ball.y - ball.radius - 10))), self.font,
                        self.font_scale,
                        ball.color, self.font_thickness)

    def get_players_detections_using_the_tracker(self, frame, boxes_detections):
        motion_estimator = MotionEstimator()
        coord_transformations = update_motion_estimator(motion_estimator=motion_estimator,
                                                        detections=boxes_detections,
                                                        frame=frame)

        players_track_objects = self.players_tracker.update(detections=boxes_detections,
                                                            coord_transformations=coord_transformations)

        return tracked_objects_to_detections(players_track_objects)

    def analyse_frame(self, frame, detections, team1_color, team2_color):
        ball = None
        boxes_detections = []
        dominant_colors = {}

        for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
            if label == PLAYER_LABEL:
                x_left, y_top, x_right, y_bottom = box
                img = frame[int(y_top):int(y_bottom), int(x_left):int(x_right)]

                if img.size <= 0:
                    continue

                color_index = get_players_team(img, self.color_index_mappings[team1_color],
                                               self.color_index_mappings[team2_color])

                if color_index is None:
                    continue

                box = np.array([[int(x_left), int(y_top)],
                                [int(x_right), int(y_bottom)]])

                data = {label: "player", "color_index": color_index, "score": score}
                boxes_detections.append(Detection(points=box, data=data))

            elif label == BALL_LABEL:
                x_left, y_top, x_right, y_bottom = box
                x = int((x_left + x_right) / 2)
                y = int((y_top + y_bottom) / 2)

                ball = Ball(x, y, 25, (0, 0, 255), score)

        if len(boxes_detections) > 0:
            players_detections = self.get_players_detections_using_the_tracker(frame, boxes_detections)
            self.get_most_dominant_colors_for_players_in_last_frames(players_detections)

            for key in self.players_colors:
                dominant_colors[key] = get_most_dominant_color(self.players_colors[key])

            for detection in players_detections:
                x_left = detection.points[0][0]
                x_right = detection.points[1][0]
                y_top = detection.points[0][1]
                y_bottom = detection.points[1][1]
                id = detection.data["id"]
                color_index = dominant_colors[id]
                score = detection.data["score"]

                self.check_team_in_possession(id, ball, x_left, y_top, x_right, y_bottom, color_index, team1_color,
                                              team2_color)
                self.print_possession(frame, id, x_left, y_top, x_right, y_bottom, team1_color, team2_color,
                                      color_index)

            # self.print_ball(frame, ball)

        return frame

    def run(self):
        self.__model.print_summary(show_architecture=False)
        self.__model = self.__model.to(self.__args.device)

        _, file_name = os.path.split(self.__args.path)

        if self.__args.device == 'cpu':
            print('Loading CPU weights...')
            state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
        else:
            print('Loading GPU weights...')
            state_dict = torch.load(args.weights)

        self.__model.load_state_dict(state_dict)
        self.__model.eval()

        video = cv2.VideoCapture(args.path)
        fps = video.get(cv2.CAP_PROP_FPS)

        (frame_width, frame_height) = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        out_sequence = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'XVID'), fps,
                                       (frame_width, frame_height))

        print('Processing video: {}'.format(args.path))
        progress_bar = tqdm.tqdm(total=n_frames)

        possession_computation_scheduler = sched.scheduler(time.time, time.sleep)
        possession_computation_scheduler.enter(1, 1, self.compute_possession, (possession_computation_scheduler,))
        scheduler_thread = Thread(target=lambda: possession_computation_scheduler.run())
        scheduler_thread.start()

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            img_tensor = augmentations.numpy2tensor(frame)

            with torch.no_grad():
                img_tensor = img_tensor.unsqueeze(dim=0).to(self.__args.device)
                detections = self.__model(img_tensor)[0]

            frame = self.analyse_frame(frame, detections, self.__args.team1_color, self.__args.team2_color)
            out_sequence.write(frame)
            progress_bar.update(1)

        progress_bar.close()
        video.release()
        out_sequence.release()
        self.close_calculator = True
        scheduler_thread.join()


if __name__ == '__main__':
    print('Running FootballAnalysis algorithm on input video')

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to video', type=str, required=True)
    parser.add_argument('--model', help='Model name', type=str, default='fb1')
    parser.add_argument('--weights', help='Path to model weights', type=str, required=True)
    parser.add_argument('--ball_threshold', help='Ball confidence detection threshold', type=float, default=0.01)
    parser.add_argument('--player_threshold', help='Player confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--out_video', help='Path to video with detection results', type=str, required=True,
                        default=None)
    parser.add_argument('--device', help='Device (CPU or CUDA)', type=str, default='cuda:0')
    parser.add_argument('--team1_color', help='Kit color for the first team', type=str, default='white')
    parser.add_argument('--team2_color', help='Kit color for the second team', type=str, default='dark_blue')
    args = parser.parse_args()

    print('Video path: {}'.format(args.path))
    print('Model: {}'.format(args.model))
    print('Model weights path: {}'.format(args.weights))
    print('Ball confidence detection threshold [0..1]: {}'.format(args.ball_threshold))
    print('Player confidence detection threshold [0..1]: {}'.format(args.player_threshold))
    print('Output video path: {}'.format(args.out_video))
    print('Device: {}'.format(args.device))
    print('Team1 color: {}'.format(args.team1_color))
    print('Team2 color: {}'.format(args.team2_color))

    print('')

    assert os.path.exists(args.weights), 'Cannot find FootAndBall model weights: {}'.format(args.weights)
    assert os.path.exists(args.path), 'Cannot open video: {}'.format(args.path)

    model = footandball.model_factory(args.model, 'detect', ball_threshold=args.ball_threshold,
                                      player_threshold=args.player_threshold)

    analyzer = GameAnalyzer(model, args)
    analyzer.run()
