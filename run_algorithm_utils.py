import os
import sched
import time
import cv2
import numpy as np
import torch
import tqdm

from threading import Thread
from typing import List

from norfair import mean_euclidean
from norfair.camera_motion import MotionEstimator
from norfair.tracker import Detection, Tracker
from norfair.tracker import TrackedObject

import FootAndBall.data.augmentation as augmentations
import FootAndBall.network.footandball as footandball
from FootAndBall.data.augmentation import PLAYER_LABEL, BALL_LABEL


def get_nr_of_zeros(mask, image):
    res = cv2.bitwise_and(image, image, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return cv2.countNonZero(res)


def get_most_dominant_color(colors):
    return max(colors, key=colors.count)


def tracked_objects_to_detections(tracked_objects: List[TrackedObject]) -> List[Detection]:
    live_objects = [entity for entity in tracked_objects if entity.live_points.any()]

    detections = []

    for tracked_object in live_objects:
        detection = tracked_object.last_detection
        detection.data["id"] = int(tracked_object.id)
        detections.append(detection)

    return detections


def create_mask(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    mask = np.ones(frame.shape[:2], dtype=frame.dtype)
    mask[69:200, 160:510] = 0

    return mask


def update_motion_estimator(motion_estimator: MotionEstimator,
                            detections: List[Detection],
                            frame: np.ndarray):
    mask = create_mask(frame, detections)
    coord_transformations = motion_estimator.update(frame, mask)

    return coord_transformations


def check_intersection(ball_x, ball_y, player_x1, player_x2, player_y1, player_y2):
    if player_x1 - 30 <= ball_x <= player_x2 + 30 and player_y1 <= ball_y <= player_y2:
        return True

    return False


def check_last_touchings_equal(touchings_array):
    for i in range(1, len(touchings_array)):
        if touchings_array[i] != touchings_array[i - 1]:
            return False

    return True
