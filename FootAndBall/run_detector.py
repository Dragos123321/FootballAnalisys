# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Run FootAndBall detector on ISSIA-CNR Soccer videos
#

import argparse
import os
import sched
import time
from threading import Thread
from typing import List

import cv2
import numpy as np
import torch
import tqdm
from norfair import mean_euclidean
from norfair.camera_motion import MotionEstimator
from norfair.tracker import Detection, Tracker
from norfair.tracker import TrackedObject

import data.augmentation as augmentations
import network.footandball as footandball
from data.augmentation import PLAYER_LABEL, BALL_LABEL

team_blue_secs = 0
team_white_secs = 0
total_time_secs = 0
team_blue_poss = ""
team_white_poss = ""
players_colors = {}
current_possession_player = None
current_possession_team = None
close_calculator = False
last_player_touching = []


def calc_possession(scheduler):
    global team_blue_secs, team_white_secs, total_time_secs, team_blue_poss, team_white_poss, current_possession_team, close_calculator
    if close_calculator:
        return
    scheduler.enter(1, 1, calc_possession, (scheduler,))
    if current_possession_team is None:
        if total_time_secs > 0:
            team_blue_poss = "{}%".format(round(100 * (team_blue_secs / total_time_secs)))
            team_white_poss = "{}%".format(round(100 * (team_white_secs / total_time_secs)))
    elif current_possession_team == 1:
        total_time_secs += 1
        team_blue_secs += 1
        team_blue_poss = "{}%".format(round(100 * (team_blue_secs / total_time_secs)))
        team_white_poss = "{}%".format(round(100 * (team_white_secs / total_time_secs)))
    else:
        total_time_secs += 1
        team_white_secs += 1
        team_blue_poss = "{}%".format(round(100 * (team_blue_secs / total_time_secs)))
        team_white_poss = "{}%".format(round(100 * (team_white_secs / total_time_secs)))


def create_mask(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    mask = np.ones(frame.shape[:2], dtype=frame.dtype)

    mask[69:200, 160:510] = 0

    return mask


def update_motion_estimator(
        motion_estimator: MotionEstimator,
        detections: List[Detection],
        frame: np.ndarray
):
    mask = create_mask(frame, detections)
    coord_transformations = motion_estimator.update(frame, mask)
    return coord_transformations


def get_player_team(image):
    frame = image.copy()
    frame = cv2.resize(frame, (250, 400))

    height, width, _ = frame.shape

    y_start = int(height * 0.275)
    y_end = int(height * 0.65)
    x_start = int(width * 0.225)
    x_end = int(width * 0.775)

    img = frame[y_start:y_end, x_start:x_end]

    # cv2.imshow("img", img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !

    # WHITE Range
    lower_white = np.array([0, 0, 210], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)

    # GREEN Range
    lower_green = np.array([33, 50, 70], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)

    # BLUE Range
    lower_blue_strong = np.array([110, 50, 60], dtype=np.uint8)
    upper_blue_strong = np.array([130, 255, 255], dtype=np.uint8)

    # BLUE Range
    lower_blue_light = np.array([100, 100, 100], dtype=np.uint8)
    upper_blue_light = np.array([110, 255, 255], dtype=np.uint8)

    # RED Range
    lower_red_low = np.array([0, 70, 70], dtype=np.uint8)
    upper_red_low = np.array([8, 255, 255], dtype=np.uint8)

    # PINK Range
    lower_red_high = np.array([175, 70, 70], dtype=np.uint8)
    upper_red_high = np.array([180, 255, 255], dtype=np.uint8)

    # BLACK Range
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 45], dtype=np.uint8)

    # ORANGE Range
    lower_orange = np.array([13, 120, 120], dtype=np.uint8)
    upper_orange = np.array([20, 255, 255], dtype=np.uint8)

    # Yellow Range
    lower_yellow = np.array([25, 200, 150], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # PURPLE Range
    lower_purple = np.array([130, 80, 80], dtype=np.uint8)
    upper_purple = np.array([160, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue_strong = cv2.inRange(hsv, lower_blue_strong, upper_blue_strong)
    mask_blue_light = cv2.inRange(hsv, lower_blue_light, upper_blue_light)
    mask_blue = cv2.bitwise_or(mask_blue_light, mask_blue_strong)
    mask_red_low = cv2.inRange(hsv, lower_red_low, upper_red_low)
    mask_red_high = cv2.inRange(hsv, lower_red_high, upper_red_high)
    mask_red = cv2.bitwise_or(mask_red_low, mask_red_high)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    nzCount = [0, 0, 0, 0, 0, 0, 0]

    res_green = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_green))

    # Bitwise-AND mask and original image
    res_white = cv2.bitwise_and(res_green, res_green, mask=mask_white)
    res_white = cv2.cvtColor(res_white, cv2.COLOR_HSV2BGR)
    res_white = cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY)
    nzCount[0] = cv2.countNonZero(res_white)

    res_blue = cv2.bitwise_and(res_green, res_green, mask=mask_blue)
    res_blue = cv2.cvtColor(res_blue, cv2.COLOR_HSV2BGR)
    res_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    nzCount[1] = cv2.countNonZero(res_blue)

    res_red = cv2.bitwise_and(res_green, res_green, mask=mask_red)
    res_red = cv2.cvtColor(res_red, cv2.COLOR_HSV2BGR)
    res_red = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
    nzCount[2] = cv2.countNonZero(res_red)

    res_black = cv2.bitwise_and(res_green, res_green, mask=mask_black)
    res_black = cv2.cvtColor(res_black, cv2.COLOR_HSV2BGR)
    res_black = cv2.cvtColor(res_black, cv2.COLOR_BGR2GRAY)
    nzCount[3] = cv2.countNonZero(res_black)

    res_orange = cv2.bitwise_and(res_green, res_green, mask=mask_orange)
    res_orange = cv2.cvtColor(res_orange, cv2.COLOR_HSV2BGR)
    res_orange = cv2.cvtColor(res_orange, cv2.COLOR_BGR2GRAY)
    nzCount[4] = cv2.countNonZero(res_orange)

    res_yellow = cv2.bitwise_and(res_green, res_green, mask=mask_yellow)
    res_yellow = cv2.cvtColor(res_yellow, cv2.COLOR_HSV2BGR)
    res_yellow = cv2.cvtColor(res_yellow, cv2.COLOR_BGR2GRAY)
    nzCount[5] = cv2.countNonZero(res_yellow)

    res_purple = cv2.bitwise_and(res_green, res_green, mask=mask_purple)
    res_purple = cv2.cvtColor(res_purple, cv2.COLOR_HSV2BGR)
    res_purple = cv2.cvtColor(res_purple, cv2.COLOR_BGR2GRAY)
    nzCount[6] = cv2.countNonZero(res_purple)

    max_zeroes = nzCount[0]
    max_index = 0

    # cv2.waitKey(250)

    for i in range(len(nzCount)):
        if nzCount[i] > max_zeroes and i in [0, 1]:
            max_zeroes = nzCount[i]
            max_index = i

    if max_zeroes > 1500:
        return max_index

    return None


player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)


def TrackedObjects_to_Detections(
        tracked_objects: List[TrackedObject],
) -> List[Detection]:
    live_objects = [
        entity for entity in tracked_objects if entity.live_points.any()
    ]

    detections = []

    for tracked_object in live_objects:
        detection = tracked_object.last_detection
        detection.data["id"] = int(tracked_object.id)
        detections.append(detection)

    return detections


def get_most_dominant_color(colors):
    return max(colors, key=colors.count)


def check_intersection(ball_x, ball_y, player_x1, player_x2, player_y1, player_y2):
    if player_x1 - 30 <= ball_x <= player_x2 + 30 and player_y1 <= ball_y <= player_y2:
        return True

    return False


def check_player_changed(touchings_array, id):
    for i in range(1, len(touchings_array)):
        if touchings_array[i] != touchings_array[i - 1]:
            return False

    if touchings_array[0] == id:
        return False

    return True


def draw_bboxes(image, detections):
    global current_possession_player, players_colors, team_white_poss, team_blue_poss, current_possession_team, last_player_touching
    font = cv2.FONT_HERSHEY_SIMPLEX
    boxes = []
    ball = []
    box_detections = []
    colors = [(255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0), (0, 165, 255), (0, 128, 128), (128, 0, 128)]
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            img = image[int(y1):int(y2), int(x1):int(x2)]

            if img.size <= 0:
                continue
            color_index = get_player_team(img)

            if color_index is None:
                continue

            box = np.array(
                [
                    [int(x1), int(y1)],
                    [int(x2), int(y2)],
                ]
            )

            data = {label: "player", "color_index": color_index, "score": score}

            box_detections.append(Detection(points=box, data=data))

            boxes.append([x1, x2, y1, y2, color_index, score])

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            radius = 25
            ball = [x, y, color, radius, score]

    motion_estimator = MotionEstimator()

    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=box_detections,
        frame=image
    )

    player_track_objects = player_tracker.update(detections=box_detections, coord_transformations=coord_transformations)

    player_detections = TrackedObjects_to_Detections(player_track_objects)

    for detection in player_detections:
        if detection.data["id"] in player_detections and len(players_colors[detection.data["id"]]) > 5:
            players_colors[detection.data["id"]].pop(0)

        if detection.data["id"] not in players_colors:
            players_colors[detection.data["id"]] = []

        players_colors[detection.data["id"]].append(detection.data["color_index"])

    dominant_colors = {}

    for key in players_colors:
        dominant_colors[key] = get_most_dominant_color(players_colors[key])

    for detection in player_detections:
        x1 = detection.points[0][0]
        x2 = detection.points[1][0]
        y1 = detection.points[0][1]
        y2 = detection.points[1][1]
        id = detection.data["id"]
        color_index = dominant_colors[id]
        score = detection.data["score"]

        if len(ball) > 0:
            has_ball = check_intersection(ball[0], ball[1], x1, x2, y1, y2)
            if has_ball and id != current_possession_player:
                if len(last_player_touching) < 7:
                    last_player_touching.append(id)
                    current_possession_player = id
                    current_possession_team = 1 if color_index == 1 else 2
                else:
                    last_player_touching.pop()
                    last_player_touching.append(id)
                    if check_player_changed(last_player_touching, id):
                        current_possession_player = id
                        current_possession_team = 1 if color_index == 1 else 2

        cv2.rectangle(image, (x1, y1), (x2, y2), colors[color_index], 2)
        cv2.putText(image, team_white_poss, (x1, max(0, y1 - 70)), font, 1, (255, 255, 255), 2)
        cv2.putText(image, team_blue_poss, (int(x1), max(0, int(y1) - 110)), font, 1, (255, 0, 0), 2)
        cv2.putText(image, "YES" if id == current_possession_player else "NO", (int(x1), max(0, int(y1) - 30)), font, 1,
                    colors[color_index], 2)

    # for box in boxes:
    #     cv2.rectangle(image, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), colors[box[4]], 2)
    # cv2.putText(image, '{:0.2f}'.format(box[5]), (int(box[0]), max(0, int(box[2]) - 30)), font, 1, colors[box[4]], 2)

    if len(ball) > 0:
        cv2.circle(image, (int(ball[0]), int(ball[1])), ball[3], ball[2], 2)
        cv2.putText(image, '{:0.2f}'.format(ball[4]),
                    (max(0, int(ball[0] - ball[3])), max(0, (ball[1] - ball[3] - 10))), font, 1,
                    ball[2], 2)

    return image


def run_detector(model, args):
    global close_calculator
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    _, file_name = os.path.split(args.path)

    if args.device == 'cpu':
        print('Loading CPU weights...')
        state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
    else:
        print('Loading GPU weights...')
        state_dict = torch.load(args.weights)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    sequence = cv2.VideoCapture(args.path)
    fps = sequence.get(cv2.CAP_PROP_FPS)
    (frame_width, frame_height) = (int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sequence = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'XVID'), fps,
                                   (frame_width, frame_height))

    print('Processing video: {}'.format(args.path))
    pbar = tqdm.tqdm(total=n_frames)

    my_scheduler = sched.scheduler(time.time, time.sleep)
    my_scheduler.enter(1, 1, calc_possession, (my_scheduler,))
    sched_thread = Thread(target=lambda: my_scheduler.run())
    sched_thread.start()

    while sequence.isOpened():
        ret, frame = sequence.read()
        if not ret:
            # End of video
            break

        # Convert color space from BGR to RGB, convert to tensor and normalize
        img_tensor = augmentations.numpy2tensor(frame)

        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)
            detections = model(img_tensor)[0]

        frame = draw_bboxes(frame, detections)
        out_sequence.write(frame)
        pbar.update(1)

    pbar.close()
    sequence.release()
    out_sequence.release()
    close_calculator = True
    sched_thread.join()


if __name__ == '__main__':
    print('Run FootAndBall detector on input video')

    # Train the DeepBall ball detector model
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to video', type=str, required=True)
    parser.add_argument('--model', help='model name', type=str, default='fb1')
    parser.add_argument('--weights', help='path to model weights', type=str, required=True)
    parser.add_argument('--ball_threshold', help='ball confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--player_threshold', help='player confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--out_video', help='path to video with detection results', type=str, required=True,
                        default=None)
    parser.add_argument('--device', help='device (CPU or CUDA)', type=str, default='cuda:0')
    args = parser.parse_args()

    print('Video path: {}'.format(args.path))
    print('Model: {}'.format(args.model))
    print('Model weights path: {}'.format(args.weights))
    print('Ball confidence detection threshold [0..1]: {}'.format(args.ball_threshold))
    print('Player confidence detection threshold [0..1]: {}'.format(args.player_threshold))
    print('Output video path: {}'.format(args.out_video))
    print('Device: {}'.format(args.device))

    print('')

    assert os.path.exists(args.weights), 'Cannot find FootAndBall model weights: {}'.format(args.weights)
    assert os.path.exists(args.path), 'Cannot open video: {}'.format(args.path)

    model = footandball.model_factory(args.model, 'detect', ball_threshold=args.ball_threshold,
                                      player_threshold=args.player_threshold)

    run_detector(model, args)
