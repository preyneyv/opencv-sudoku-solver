import math
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


overlay = np.zeros((1080, 1920, 1), np.uint8)
last = None


def handle_results(result, image, ts):
    global last
    if not len(result.hand_landmarks):
        return

    data = image.numpy_view()
    h, w, c = data.shape

    hand = result.hand_landmarks[0]
    world = result.hand_world_landmarks[0]
    for mark in hand:
        cv2.circle(data, (int(mark.x * w), int(mark.y * h)), 10, (255, 0, 0, 127), -1)

    thumb = hand[4]
    index = hand[8]

    w_thumb = world[4]
    w_index = world[8]

    dist = math.sqrt((w_thumb.x - w_index.x) ** 2 + (w_thumb.y - w_index.y) ** 2)

    if dist < (0.01 if last is None else 0.02):
        mid = (thumb.x + index.x) / 2, (thumb.y + index.y) / 2
        if last is None:
            last = mid
        cv2.line(
            overlay,
            (int(last[0] * w), int(last[1] * h)),
            (int(mid[0] * w), int(mid[1] * h)),
            (255,),
            10,
        )
        last = mid
    else:
        last = None


base_options = mp_python.BaseOptions(
    model_asset_path="hand_assets/hand_landmarker.task"
)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    result_callback=handle_results,
)
detector = mp_vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)
    image = mp.Image(
        mp.ImageFormat.SRGB,
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    detector.detect_async(image, int(time.time() * 1000))

    mix = cv2.bitwise_or(img, cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR))
    cv2.imshow("", mix)
    cv2.waitKey(1)
