#!/usr/bin/env python3
"""2026 电子设计竞赛 H 题（车载平衡滚球视觉系统）· 香橙派实时检测主程序.

YOLO26 INT8 high-FPS camera inference for Orange Pi 5 Pro / RK3588.

The RKNN model must expose:
  output[0]: decoded xywh boxes [1, 4, 5376]
  output[1]: pre-Sigmoid class logits [1, 1, 5376]
"""
import os
import json
import socket
import struct
import time
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Full, Queue

import cv2
import numpy as np
from rknnlite.api import RKNNLite

try:
    import serial
except ImportError:
    serial = None


# ==================== Configuration ====================
MODEL_FILE = os.environ.get(
    "RKNN_MODEL", "ball_yolo26_int8_512_raw.rknn"
)
CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
IMG_SIZE = 512
# A low candidate threshold keeps an established fast-moving track alive.
# New/reacquired tracks still require TRACK_CONFIRM_THRESH.
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.12"))
NMS_THRESH = float(os.environ.get("NMS_THRESH", "0.45"))
TRACK_CONFIRM_THRESH = float(
    os.environ.get("TRACK_CONFIRM_THRESH", "0.25")
)
TRACK_GATE_PIXELS = float(os.environ.get("TRACK_GATE_PIXELS", "160"))
TRACK_GATE_RADIUS = float(os.environ.get("TRACK_GATE_RADIUS", "8"))
TRACK_HOLD_FRAMES = int(os.environ.get("TRACK_HOLD_FRAMES", "2"))
MIN_AXIS_PIXELS = 100.0
MAX_CANDIDATES = 300
NPU_WORKERS = 3

CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.environ.get("CAMERA_FPS", "90"))

MJPEG_HOST = "0.0.0.0"
MJPEG_PORT = int(os.environ.get("MJPEG_PORT", "8080"))
MJPEG_MAX_FPS = int(os.environ.get("MJPEG_FPS", "30"))
MJPEG_QUALITY = int(os.environ.get("MJPEG_QUALITY", "75"))
MJPEG_WIDTH = int(os.environ.get("MJPEG_WIDTH", "640"))
STREAM_ROI_ENABLED = os.environ.get("STREAM_ROI_ENABLED", "1") == "1"
STREAM_ANNOTATED = os.environ.get("STREAM_ANNOTATED", "1") == "1"
STREAM_PERSPECTIVE = os.environ.get("STREAM_PERSPECTIVE", "1") == "1"
STREAM_ROI_X1 = float(os.environ.get("STREAM_ROI_X1", "0.10"))
STREAM_ROI_Y1 = float(os.environ.get("STREAM_ROI_Y1", "0.25"))
STREAM_ROI_X2 = float(os.environ.get("STREAM_ROI_X2", "0.95"))
STREAM_ROI_Y2 = float(os.environ.get("STREAM_ROI_Y2", "0.44"))
STREAM_WARP_WIDTH = int(os.environ.get("STREAM_WARP_WIDTH", "640"))
STREAM_WARP_HEIGHT = int(os.environ.get("STREAM_WARP_HEIGHT", "150"))
STREAM_WARP_TOP_PX = float(os.environ.get("STREAM_WARP_TOP_PX", "80"))
STREAM_WARP_BOTTOM_PX = float(
    os.environ.get("STREAM_WARP_BOTTOM_PX", "35")
)
STREAM_WARP_END_MARGIN_PX = float(
    os.environ.get("STREAM_WARP_END_MARGIN_PX", "8")
)
DISPLAY_MODE = os.environ.get("LOCAL_DISPLAY", "auto").lower()
LOCAL_DISPLAY = DISPLAY_MODE in {"1", "true", "yes"} or (
    DISPLAY_MODE == "auto"
    and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
)
WINDOW_NAME = "BallVision - Test View 1"
AXIS_LENGTH_CM = float(os.environ.get("AXIS_LENGTH_CM", "25.5"))
VELOCITY_UPDATE_INTERVAL_S = float(
    os.environ.get("VELOCITY_UPDATE_INTERVAL_S", "0.020")
)
VELOCITY_WINDOW_S = float(os.environ.get("VELOCITY_WINDOW_S", "0.100"))
VELOCITY_LOST_TIMEOUT_S = float(
    os.environ.get("VELOCITY_LOST_TIMEOUT_S", "0.150")
)
CALIBRATION_FILE = os.path.expanduser(
    os.environ.get(
        "CALIBRATION_FILE", "~/ball_axis_calibration.json"
    )
)

UART_PORT = os.environ.get("UART_PORT", "/dev/ttyS0")
UART_BAUD = 115200
UART_ENABLED = os.environ.get("UART_ENABLED", "1") == "1"
UART_VALUE_SCALE = 100.0


# Keep only a short inference backlog. When overloaded, discard the oldest
# unprocessed frame and control from the newest measured ball position.
frame_queue = Queue(maxsize=max(NPU_WORKERS, 1))
inference_result_queue = Queue()
dropped_sequence_queue = Queue()
# Wireless preview may replace an old frame; it is not the detection path.
encode_queue = Queue(maxsize=1)
stop_event = threading.Event()
result_lock = threading.Lock()
jpeg_condition = threading.Condition()

completion_times = deque()
capture_times = deque()
display_times = deque()
latest_jpeg = None
latest_jpeg_seq = -1


def request_stop(reason):
    if not stop_event.is_set():
        print(f"[Main] Stopping: {reason}")
        stop_event.set()
        with jpeg_condition:
            jpeg_condition.notify_all()


def put_latest(queue, item):
    """Replace an old wireless-preview frame; never use for detection."""
    try:
        queue.put_nowait(item)
        return
    except Full:
        pass
    try:
        queue.get_nowait()
    except Empty:
        pass
    try:
        queue.put_nowait(item)
    except Full:
        pass


def put_detection_frame(item):
    """Enqueue the newest detection frame without blocking camera capture."""
    try:
        frame_queue.put_nowait(item)
        return True
    except Full:
        pass
    try:
        dropped = frame_queue.get_nowait()
        dropped_sequence_queue.put_nowait(dropped[0])
    except Empty:
        pass
    try:
        frame_queue.put_nowait(item)
        return True
    except Full:
        dropped_sequence_queue.put_nowait(item[0])
        return False


def crop_stream_roi(frame):
    """Return the raw groove ROI used only by the wireless preview."""
    if not STREAM_ROI_ENABLED:
        return frame
    height, width = frame.shape[:2]
    x1 = int(round(np.clip(STREAM_ROI_X1, 0.0, 1.0) * width))
    y1 = int(round(np.clip(STREAM_ROI_Y1, 0.0, 1.0) * height))
    x2 = int(round(np.clip(STREAM_ROI_X2, 0.0, 1.0) * width))
    y2 = int(round(np.clip(STREAM_ROI_Y2, 0.0, 1.0) * height))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return frame
    return frame[y1:y2, x1:x2]


def letterbox(image):
    height, width = image.shape[:2]
    scale = min(IMG_SIZE / height, IMG_SIZE / width)
    new_width = round(width * scale)
    new_height = round(height * scale)
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    x_offset = (IMG_SIZE - new_width) // 2
    y_offset = (IMG_SIZE - new_height) // 2
    canvas[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width,
    ] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, 0), scale, x_offset, y_offset


def stable_sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def normalize_outputs(outputs):
    if len(outputs) != 2:
        raise RuntimeError(f"expected 2 RKNN outputs, got {len(outputs)}")

    boxes = np.asarray(outputs[0], dtype=np.float32)
    logits = np.asarray(outputs[1], dtype=np.float32)
    if boxes.shape == (1, 4, 5376):
        boxes = boxes[0].T
    elif boxes.shape == (1, 5376, 4):
        boxes = boxes[0]
    else:
        raise RuntimeError(f"unexpected boxes output: {boxes.shape}")

    if logits.shape == (1, 1, 5376):
        logits = logits.reshape(-1)
    elif logits.shape == (1, 5376, 1):
        logits = logits.reshape(-1)
    else:
        raise RuntimeError(f"unexpected logits output: {logits.shape}")
    return boxes, logits


def nms_xyxy(boxes, scores):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[index], x1[rest])
        yy1 = np.maximum(y1[index], y1[rest])
        xx2 = np.minimum(x2[index], x2[rest])
        yy2 = np.minimum(y2[index], y2[rest])
        intersection = (
            np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        )
        union = areas[index] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-7)
        order = rest[iou <= NMS_THRESH]
    return np.asarray(keep, dtype=np.int64)


def postprocess(outputs, scale, x_offset, y_offset, width, height):
    boxes_xywh, logits = normalize_outputs(outputs)
    scores = stable_sigmoid(logits)
    max_conf = float(scores.max(initial=0.0))
    selected = np.flatnonzero(scores >= CONF_THRESH)
    if selected.size == 0:
        return [], max_conf

    if selected.size > MAX_CANDIDATES:
        local_scores = scores[selected]
        top = np.argpartition(
            local_scores, -MAX_CANDIDATES
        )[-MAX_CANDIDATES:]
        selected = selected[top]

    boxes_xywh = boxes_xywh[selected]
    selected_scores = scores[selected]
    cx, cy, bw, bh = boxes_xywh.T
    boxes = np.column_stack(
        (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    )
    keep = nms_xyxy(boxes, selected_scores)
    boxes = boxes[keep]
    selected_scores = selected_scores[keep]

    boxes[:, [0, 2]] = np.clip(
        (boxes[:, [0, 2]] - x_offset) / scale, 0, width - 1
    )
    boxes[:, [1, 3]] = np.clip(
        (boxes[:, [1, 3]] - y_offset) / scale, 0, height - 1
    )

    balls = []
    for box, confidence in zip(boxes, selected_scores):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        if x2 <= x1 or y2 <= y1:
            continue
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = max(x2 - x1, y2 - y1) // 2
        balls.append(
            (
                x1,
                y1,
                x2,
                y2,
                center_x,
                center_y,
                radius,
                float(confidence),
            )
        )
    return balls, max_conf


class BallStabilizer:
    """Loose single-ball association with speed-adaptive alpha-beta filtering."""

    def __init__(self):
        self.center = None
        self.size = None
        self.radius = None
        self.confidence = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.missed = 0

    def reset(self):
        self.center = None
        self.size = None
        self.radius = None
        self.confidence = None
        self.velocity[:] = 0
        self.missed = 0

    def _emit(self, image_width, image_height):
        cx, cy = self.center
        box_width, box_height = self.size
        x1 = int(round(np.clip(cx - box_width / 2, 0, image_width - 1)))
        y1 = int(round(np.clip(cy - box_height / 2, 0, image_height - 1)))
        x2 = int(round(np.clip(cx + box_width / 2, 0, image_width - 1)))
        y2 = int(round(np.clip(cy + box_height / 2, 0, image_height - 1)))
        return [
            (
                x1,
                y1,
                x2,
                y2,
                int(round(np.clip(cx, 0, image_width - 1))),
                int(round(np.clip(cy, 0, image_height - 1))),
                max(1, int(round(self.radius))),
                float(self.confidence),
            )
        ]

    def _miss(self, image_width, image_height):
        if self.center is None:
            return []
        self.missed += 1
        if self.missed > TRACK_HOLD_FRAMES:
            self.reset()
            return []
        # Predict through only a very short miss; decay velocity/confidence.
        self.center = self.center + self.velocity
        self.velocity *= 0.80
        self.confidence *= 0.88
        return self._emit(image_width, image_height)

    @staticmethod
    def _observation(ball):
        x1, y1, x2, y2, cx, cy, radius, confidence = ball
        return {
            "ball": ball,
            "center": np.array([cx, cy], dtype=np.float32),
            "size": np.array(
                [max(1, x2 - x1), max(1, y2 - y1)], dtype=np.float32
            ),
            "radius": float(radius),
            "confidence": float(confidence),
        }

    def update(self, candidates, image_width, image_height):
        observations = [self._observation(ball) for ball in candidates]
        if not observations:
            return self._miss(image_width, image_height)

        if self.center is None:
            chosen = max(observations, key=lambda item: item["confidence"])
            if chosen["confidence"] < TRACK_CONFIRM_THRESH:
                return []
            self.center = chosen["center"]
            self.size = chosen["size"]
            self.radius = chosen["radius"]
            self.confidence = chosen["confidence"]
            self.velocity[:] = 0
            self.missed = 0
            return self._emit(image_width, image_height)

        predicted_center = self.center + self.velocity
        gate = max(TRACK_GATE_PIXELS, TRACK_GATE_RADIUS * self.radius)
        associated = []
        for observation in observations:
            distance = float(
                np.linalg.norm(observation["center"] - predicted_center)
            )
            if distance <= gate:
                # Confidence dominates; distance only gently breaks ties.
                score = observation["confidence"] - 0.20 * distance / gate
                associated.append((score, observation))

        if associated:
            chosen = max(associated, key=lambda item: item[0])[1]
        else:
            chosen = max(observations, key=lambda item: item["confidence"])
            if chosen["confidence"] < TRACK_CONFIRM_THRESH:
                return self._miss(image_width, image_height)
            # A strong detection outside the loose gate immediately reacquires.
            self.center = chosen["center"]
            self.size = chosen["size"]
            self.radius = chosen["radius"]
            self.confidence = chosen["confidence"]
            self.velocity[:] = 0
            self.missed = 0
            return self._emit(image_width, image_height)

        movement = float(np.linalg.norm(chosen["center"] - self.center))
        speed_in_radii = movement / max(self.radius, 6.0)
        # Slow ball: alpha near 0.30 for strong smoothing.
        # Fast ball: alpha approaches 0.90 to avoid visible trailing.
        alpha = float(np.clip(0.30 + 0.18 * speed_in_radii, 0.30, 0.90))
        beta = float(np.clip(0.06 + 0.05 * speed_in_radii, 0.06, 0.28))
        residual = chosen["center"] - predicted_center
        self.center = predicted_center + alpha * residual
        self.velocity = 0.82 * self.velocity + beta * residual
        size_alpha = float(np.clip(alpha * 0.75, 0.28, 0.72))
        self.size = (
            (1.0 - size_alpha) * self.size + size_alpha * chosen["size"]
        )
        self.radius = (
            (1.0 - size_alpha) * self.radius
            + size_alpha * chosen["radius"]
        )
        self.confidence = (
            0.55 * self.confidence + 0.45 * chosen["confidence"]
        )
        self.missed = 0
        return self._emit(image_width, image_height)


class AxisCalibration:
    """Three-point 1D calibration: center, left stop, right stop."""

    def __init__(self):
        self.length_cm = AXIS_LENGTH_CM
        self.normalized_points = []
        self.selecting = False
        self.last_image_size = (CAMERA_WIDTH, CAMERA_HEIGHT)
        self.load()

    def load(self):
        if not os.path.isfile(CALIBRATION_FILE):
            return
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            points = data.get("normalized_points", [])
            if len(points) == 2:
                # Backward compatibility with the former [left, right] file.
                left = np.asarray(points[0], dtype=np.float32)
                right = np.asarray(points[1], dtype=np.float32)
                center = (left + right) * 0.5
                points = [center.tolist(), left.tolist(), right.tolist()]
                print(
                    "[Calibration] Legacy two-point file loaded; "
                    "press C to replace it with a three-point calibration."
                )
            elif len(points) != 3:
                raise ValueError("expected center, left and right points")
            axis_pixels = np.linalg.norm(
                np.array(points[2], dtype=np.float32)
                * np.array([CAMERA_WIDTH, CAMERA_HEIGHT], dtype=np.float32)
                - np.array(points[1], dtype=np.float32)
                * np.array([CAMERA_WIDTH, CAMERA_HEIGHT], dtype=np.float32)
            )
            if axis_pixels < MIN_AXIS_PIXELS:
                raise ValueError(
                    f"calibration endpoints are only {axis_pixels:.1f}px apart"
                )
            self.length_cm = float(
                data.get("length_cm", AXIS_LENGTH_CM)
            )
            self.normalized_points = [
                (float(point[0]), float(point[1])) for point in points
            ]
            if self._geometry(CAMERA_WIDTH, CAMERA_HEIGHT) is None:
                raise ValueError("center is not between the two endpoints")
            print(
                f"[Calibration] Loaded: {CALIBRATION_FILE}, "
                f"length={self.length_cm:.1f} cm"
            )
        except Exception as error:
            print(f"[Calibration] Ignored invalid file: {error}")
            self.normalized_points = []

    def save(self):
        data = {
            "format_version": 2,
            "point_order": ["center", "left", "right"],
            "length_cm": self.length_cm,
            "normalized_points": self.normalized_points,
        }
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        print(f"[Calibration] Saved: {CALIBRATION_FILE}")

    def begin(self):
        # A new calibration always uses the current configured physical length.
        self.length_cm = AXIS_LENGTH_CM
        self.normalized_points = []
        self.selecting = True
        print(
            "[Calibration] Step 1/3: click the physical CENTER "
            "of the tube (0.00 cm)."
        )

    def mouse_callback(self, event, x, y, flags, param):
        del flags, param
        if event != cv2.EVENT_LBUTTONUP or not self.selecting:
            return
        image_width, image_height = self.last_image_size
        # WINDOW_AUTOSIZE keeps callback coordinates equal to image pixels.
        image_x = np.clip(x, 0, image_width - 1)
        image_y = np.clip(y, 0, image_height - 1)
        new_point = (
            float(image_x / image_width),
            float(image_y / image_height),
        )
        click = np.array([image_x, image_y], dtype=np.float32)
        step = len(self.normalized_points)
        if step == 1:
            center = np.array(
                [
                    self.normalized_points[0][0] * image_width,
                    self.normalized_points[0][1] * image_height,
                ],
                dtype=np.float32,
            )
            distance = float(np.linalg.norm(click - center))
            if distance < 30.0:
                print(
                    f"[Calibration] Left point rejected: only "
                    f"{distance:.1f}px from center."
                )
                return
        elif step == 2:
            candidate_points = self.normalized_points + [new_point]
            old_points = self.normalized_points
            self.normalized_points = candidate_points
            geometry = self._geometry(image_width, image_height)
            self.normalized_points = old_points
            if geometry is None:
                print(
                    "[Calibration] Right point rejected: center must lie "
                    "between the left and right endpoints, and the full "
                    f"axis must be at least {MIN_AXIS_PIXELS:.0f}px."
                )
                return

        self.normalized_points.append(new_point)
        if len(self.normalized_points) == 1:
            print(
                "[Calibration] Center 0.00 cm accepted. Step 2/3: "
                "click the LEFT inner stop at "
                f"{-self.length_cm / 2.0:+.2f} cm."
            )
        elif len(self.normalized_points) == 2:
            print(
                f"[Calibration] Left {-self.length_cm / 2.0:+.2f} cm "
                "accepted. Step 3/3: click the RIGHT inner stop at "
                f"{self.length_cm / 2.0:+.2f} cm."
            )
        else:
            self.selecting = False
            self.save()
            print(
                "[Calibration] Complete: CENTER=0.00 cm with independent "
                "left/right scales."
            )

    def pixel_points(self, image_width, image_height):
        if len(self.normalized_points) != 3:
            return None
        return [
            np.array(
                [point[0] * image_width, point[1] * image_height],
                dtype=np.float32,
            )
            for point in self.normalized_points
        ]

    def _geometry(self, image_width, image_height):
        points = self.pixel_points(image_width, image_height)
        if points is None:
            return None
        center, left, right = points
        axis = right - left
        denominator = float(np.dot(axis, axis))
        if denominator < MIN_AXIS_PIXELS * MIN_AXIS_PIXELS:
            return None
        center_fraction = float(
            np.dot(center - left, axis) / denominator
        )
        if not 0.05 < center_fraction < 0.95:
            return None
        center_projection = left + center_fraction * axis
        return left, right, axis, denominator, center_fraction, center_projection

    def warp_stream_band(self, frame):
        """Rectify a narrow band around the calibrated tube axis."""
        image_height, image_width = frame.shape[:2]
        geometry = self._geometry(image_width, image_height)
        if geometry is None:
            return None
        left, right, axis, denominator, _center_fraction, _ = geometry
        axis_length = float(np.sqrt(denominator))
        if axis_length < MIN_AXIS_PIXELS:
            return None
        direction = axis / axis_length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        if normal[1] < 0:
            normal = -normal
        left = left - direction * STREAM_WARP_END_MARGIN_PX
        right = right + direction * STREAM_WARP_END_MARGIN_PX
        source_quad = np.array(
            [
                left - normal * STREAM_WARP_TOP_PX,
                right - normal * STREAM_WARP_TOP_PX,
                right + normal * STREAM_WARP_BOTTOM_PX,
                left + normal * STREAM_WARP_BOTTOM_PX,
            ],
            dtype=np.float32,
        )
        destination_quad = np.array(
            [
                [0, 0],
                [STREAM_WARP_WIDTH - 1, 0],
                [STREAM_WARP_WIDTH - 1, STREAM_WARP_HEIGHT - 1],
                [0, STREAM_WARP_HEIGHT - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(
            source_quad, destination_quad
        )
        return cv2.warpPerspective(
            frame,
            transform,
            (STREAM_WARP_WIDTH, STREAM_WARP_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def position_cm(self, center, image_width, image_height):
        geometry = self._geometry(image_width, image_height)
        if geometry is None or center is None:
            return None
        left, _right, axis, denominator, center_fraction, _ = geometry
        center_array = np.asarray(center, dtype=np.float32)
        fraction = float(
            np.dot(center_array - left, axis) / denominator
        )
        half_length = self.length_cm / 2.0
        if fraction <= center_fraction:
            centered_cm = (
                half_length
                * (fraction - center_fraction)
                / center_fraction
            )
        else:
            centered_cm = (
                half_length
                * (fraction - center_fraction)
                / (1.0 - center_fraction)
            )
        return centered_cm, left + fraction * axis

    def draw(self, frame, ball_center):
        image_height, image_width = frame.shape[:2]
        self.last_image_size = (image_width, image_height)
        points = self.pixel_points(image_width, image_height)
        geometry = self._geometry(image_width, image_height)
        if geometry is not None:
            left, right, axis, _denominator, center_fraction, center_projection = (
                geometry
            )
            left_i = tuple(np.rint(left).astype(int))
            right_i = tuple(np.rint(right).astype(int))
            cv2.line(frame, left_i, right_i, (255, 0, 255), 2)
            half_length = self.length_cm / 2.0
            tick_values = [
                -half_length,
                -10.0,
                -5.0,
                0.0,
                5.0,
                10.0,
                half_length,
            ]
            for centimeters in tick_values:
                if centimeters <= 0.0:
                    fraction = center_fraction * (
                        1.0 + centimeters / half_length
                    )
                else:
                    fraction = center_fraction + (
                        centimeters / half_length
                    ) * (1.0 - center_fraction)
                tick = left + fraction * axis
                tick_i = tuple(np.rint(tick).astype(int))
                cv2.circle(frame, tick_i, 4, (255, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"{centimeters:+g}" if centimeters != 0.0 else "0",
                    (tick_i[0] + 4, max(12, tick_i[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 0, 255),
                    1,
                )
            center_i = tuple(np.rint(center_projection).astype(int))
            cv2.circle(frame, center_i, 7, (0, 165, 255), 2)

        position = self.position_cm(
            ball_center, image_width, image_height
        )
        if position is not None:
            centimeters, projection = position
            projection_i = tuple(np.rint(projection).astype(int))
            cv2.circle(frame, projection_i, 5, (255, 255, 0), -1)
            cv2.putText(
                frame,
                f"POS {centimeters:+.2f}cm  (CENTER = 0)",
                (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 0),
                2,
            )

        if self.selecting:
            step = len(self.normalized_points)
            targets = [
                "CENTER 0.00 cm",
                f"LEFT {-self.length_cm / 2.0:+.2f} cm INNER stop",
                f"RIGHT {self.length_cm / 2.0:+.2f} cm INNER stop",
            ]
            target = targets[min(step, 2)]
            cv2.putText(
                frame,
                f"CALIBRATION {step + 1}/3: click {target}",
                (10, image_height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 255),
                2,
            )
        elif points is None:
            cv2.putText(
                frame,
                f"Press C to calibrate the {self.length_cm:g} cm tube",
                (10, image_height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 255),
                2,
            )
        return position


class BallVelocityEstimator:
    """Windowed position slope using camera capture timestamps."""

    def __init__(
        self,
        update_interval_s=VELOCITY_UPDATE_INTERVAL_S,
        window_s=VELOCITY_WINDOW_S,
        lost_timeout_s=VELOCITY_LOST_TIMEOUT_S,
    ):
        self.update_interval_s = max(float(update_interval_s), 0.001)
        # Samples arrive at the camera/detection rate, so a 100 ms fit window
        # remains valid even when velocity/UART output is limited to 10 Hz.
        self.window_s = max(float(window_s), self.update_interval_s)
        self.lost_timeout_s = max(float(lost_timeout_s), self.window_s)
        self.samples = deque()
        self.last_output_time = None
        self.last_valid_sample_time = None
        self.velocity_cm_s = None
        self.output_updated = False

    def reset(self):
        self.samples.clear()
        self.last_output_time = None
        self.last_valid_sample_time = None
        self.velocity_cm_s = None
        self.output_updated = False

    def update(self, capture_time, position_cm):
        self.output_updated = False
        capture_time = float(capture_time)
        if position_cm is None:
            if (
                self.last_valid_sample_time is not None
                and capture_time - self.last_valid_sample_time
                > self.lost_timeout_s
            ):
                self.reset()
            return self.velocity_cm_s

        if self.samples and capture_time <= self.samples[-1][0]:
            return self.velocity_cm_s

        self.samples.append((capture_time, float(position_cm)))
        self.last_valid_sample_time = capture_time
        window_start = capture_time - self.window_s
        while len(self.samples) > 2 and self.samples[1][0] < window_start:
            self.samples.popleft()

        if (
            self.last_output_time is not None
            and capture_time - self.last_output_time
            < self.update_interval_s
        ):
            return self.velocity_cm_s
        if len(self.samples) < 3:
            return self.velocity_cm_s

        times = np.asarray(
            [sample[0] for sample in self.samples], dtype=np.float64
        )
        positions = np.asarray(
            [sample[1] for sample in self.samples], dtype=np.float64
        )
        times -= times.mean()
        denominator = float(np.dot(times, times))
        if denominator < 1e-9:
            return self.velocity_cm_s

        centered_positions = positions - positions.mean()
        self.velocity_cm_s = float(
            np.dot(times, centered_positions) / denominator
        )
        self.last_output_time = capture_time
        self.output_updated = True
        return self.velocity_cm_s


def open_camera():
    pipeline = (
        f"v4l2src device=/dev/video{CAMERA_ID} io-mode=2 ! "
        f"image/jpeg,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},"
        f"framerate={CAMERA_FPS}/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    backend = "GStreamer MJPEG"
    if capture.isOpened():
        return capture, backend

    capture.release()
    capture = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    backend = "V4L2 MJPEG"
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture, backend


def capture_worker():
    capture, backend = open_camera()
    if not capture.isOpened():
        request_stop(f"cannot open /dev/video{CAMERA_ID}")
        return
    print(
        f"[Camera] {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{capture.get(cv2.CAP_PROP_FPS):.1f} FPS, {backend}"
    )

    sequence = 0
    while not stop_event.is_set():
        ok, frame = capture.read()
        if not ok:
            continue
        capture_time = time.perf_counter()
        sequence += 1
        # In raw-stream mode this branch is independent from inference.
        # Annotated-stream mode publishes the completed frame in main().
        if not STREAM_ANNOTATED:
            put_latest(encode_queue, (sequence, frame, False))
        network_input, scale, x_offset, y_offset = letterbox(frame)

        with result_lock:
            capture_times.append(capture_time)
            while (
                capture_times
                and capture_times[0] < capture_time - 1.0
            ):
                capture_times.popleft()
            camera_fps = float(len(capture_times))

        put_detection_frame(
            (
                sequence,
                capture_time,
                frame,
                network_input,
                scale,
                x_offset,
                y_offset,
                camera_fps,
            )
        )
    capture.release()


def inference_worker(worker_id, core_mask):
    print(f"[NPU-{worker_id}] Loading: {MODEL_FILE}")
    rknn = RKNNLite()
    try:
        if rknn.load_rknn(MODEL_FILE) != 0:
            request_stop(f"NPU-{worker_id} load failed")
            return
        if rknn.init_runtime(core_mask=core_mask) != 0:
            request_stop(f"NPU-{worker_id} runtime init failed")
            return
        print(f"[NPU-{worker_id}] Ready on core {worker_id}")

        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.2)
            except Empty:
                continue
            (
                sequence,
                capture_time,
                frame,
                network_input,
                scale,
                x_offset,
                y_offset,
                camera_fps,
            ) = item
            outputs = rknn.inference(
                inputs=[network_input], data_format=["nhwc"]
            )
            if not outputs:
                request_stop(f"NPU-{worker_id} inference failed")
                return
            try:
                balls, max_conf = postprocess(
                    outputs,
                    scale,
                    x_offset,
                    y_offset,
                    frame.shape[1],
                    frame.shape[0],
                )
            except Exception as error:
                print(f"[NPU-{worker_id}] Postprocess error: {error}")
                request_stop("invalid RKNN output")
                return

            now = time.perf_counter()
            with result_lock:
                completion_times.append(now)
                while (
                    completion_times
                    and completion_times[0] < now - 1.0
                ):
                    completion_times.popleft()
                npu_fps = float(len(completion_times))
            inference_result_queue.put(
                {
                    "seq": sequence,
                    "frame": frame,
                    "balls": balls,
                    "npu_fps": npu_fps,
                    "camera_fps": camera_fps,
                    "capture_time": capture_time,
                    "max_conf": max_conf,
                }
            )
    finally:
        rknn.release()


def jpeg_encoder_worker():
    global latest_jpeg, latest_jpeg_seq
    minimum_interval = 1.0 / max(MJPEG_MAX_FPS, 1)
    last_encode = 0.0
    while not stop_event.is_set():
        try:
            sequence, frame, prepared = encode_queue.get(timeout=0.2)
        except Empty:
            continue
        delay = minimum_interval - (time.perf_counter() - last_encode)
        if delay > 0:
            stop_event.wait(delay)
        if stop_event.is_set():
            break
        # The encoder may have waited for its 30 FPS slot. Discard any older
        # preview item accumulated during that wait and encode the newest one.
        while True:
            try:
                sequence, frame, prepared = encode_queue.get_nowait()
            except Empty:
                break
        encode_frame = frame if prepared else crop_stream_roi(frame)
        frame_height, frame_width = encode_frame.shape[:2]
        if 0 < MJPEG_WIDTH and MJPEG_WIDTH != frame_width:
            stream_height = max(
                1, int(round(frame_height * MJPEG_WIDTH / frame_width))
            )
            interpolation = (
                cv2.INTER_AREA
                if MJPEG_WIDTH < frame_width
                else cv2.INTER_LINEAR
            )
            encode_frame = cv2.resize(
                encode_frame,
                (MJPEG_WIDTH, stream_height),
                interpolation=interpolation,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            encode_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY],
        )
        if not ok:
            continue
        with jpeg_condition:
            latest_jpeg = encoded.tobytes()
            latest_jpeg_seq = sequence
            jpeg_condition.notify_all()
        last_encode = time.perf_counter()


WEB_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BallVision</title>
<style>
:root{color-scheme:dark;font-family:Arial,"Microsoft YaHei",sans-serif}
*{box-sizing:border-box}
body{margin:0;background:#16191d;color:#f4f6f8}
header{height:58px;padding:0 20px;display:flex;align-items:center;
justify-content:space-between;border-bottom:1px solid #30353c;background:#20242a}
h1{margin:0;font-size:20px}.status{color:#aeb7c2;font-size:14px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;
background:#32cd74;margin-right:7px}
main{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);
gap:16px;padding:16px;max-width:1400px;margin:auto}
.panel{background:#20242a;border:1px solid #30353c;border-radius:10px;
overflow:hidden}
.panel-title{padding:12px 14px;border-bottom:1px solid #30353c;
font-size:15px;color:#cbd2da}
.live-wrap{aspect-ratio:4/3;background:#000;position:relative}
/* Keep the MJPEG image visible so Chromium does not throttle its decoder.
   The tiny canvas is only the display-side MediaRecorder source. */
#source{display:block;width:100%;height:100%;object-fit:contain}
#live{position:absolute;left:0;top:0;width:1px;height:1px;opacity:0;
pointer-events:none}
.toolbar{display:flex;gap:10px;padding:13px;align-items:center;flex-wrap:wrap}
button{height:42px;border:0;border-radius:6px;padding:0 18px;
font-size:15px;color:white;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
#start{background:#2ebd6b}#stop{background:#d84a4a}
#requirement{height:42px;width:92px;border:1px solid #46505b;border-radius:6px;
background:#16191d;color:#fff;padding:0 10px;font-size:15px}
.field-label{color:#cbd2da;font-size:14px}
#timer{margin-left:auto;align-self:center;font-variant-numeric:tabular-nums;
color:#cbd2da}
.playback{aspect-ratio:4/3;background:#090a0c}
#player{display:block;width:100%;height:100%;object-fit:contain}
#recordings{max-height:300px;overflow:auto}
.empty{padding:22px;color:#8f99a5;text-align:center}
.recording{padding:11px 13px;border-top:1px solid #30353c;cursor:pointer;
display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center}
.recording:hover,.recording.active{background:#2a3037}
.recording strong{display:block;font-size:14px;margin-bottom:4px}
.recording span{font-size:12px;color:#98a3af}
.download{height:32px;background:#386b9d;padding:0 12px;font-size:13px}
@media(max-width:850px){main{grid-template-columns:1fr}.playback{aspect-ratio:16/9}}
</style>
</head>
<body>
<header>
  <h1>BallVision</h1>
  <div class="status"><span class="dot"></span><span id="state">实时画面</span></div>
</header>
<main>
  <section class="panel">
    <div class="panel-title">钢球与水管检测画面</div>
    <div class="live-wrap" id="liveWrap">
      <img id="source" src="/stream.mjpg" alt="">
      <canvas id="live" width="640" height="480"></canvas>
    </div>
    <div class="toolbar">
      <label class="field-label" for="requirement">要求编号</label>
      <input id="requirement" type="number" min="1" step="1" value="1">
      <button id="start">开始录像</button>
      <button id="stop" disabled>停止并保存</button>
      <span id="timer">00:00.0</span>
    </div>
  </section>
  <section class="panel">
    <div class="panel-title">保存与回放（显示端）</div>
    <div class="playback" id="playbackWrap"><video id="player" controls playsinline></video></div>
    <div id="recordings"><div class="empty">尚未保存录像</div></div>
  </section>
</main>
<script>
const source=document.getElementById("source");
const canvas=document.getElementById("live");
const ctx=canvas.getContext("2d",{alpha:false});
const liveWrap=document.getElementById("liveWrap");
const playbackWrap=document.getElementById("playbackWrap");
const requirementInput=document.getElementById("requirement");
const startBtn=document.getElementById("start");
const stopBtn=document.getElementById("stop");
const timer=document.getElementById("timer");
const stateText=document.getElementById("state");
const list=document.getElementById("recordings");
const player=document.getElementById("player");
let recorder=null,chunks=[],startedAt=0,timerId=null,db=null,currentUrl=null;
let activeRequirement=1;

function drawLive(){
  if(source.naturalWidth>0){
    if(canvas.width!==source.naturalWidth||canvas.height!==source.naturalHeight){
      canvas.width=source.naturalWidth;canvas.height=source.naturalHeight;
      const ratio=source.naturalWidth+"/"+source.naturalHeight;
      liveWrap.style.aspectRatio=ratio;
      playbackWrap.style.aspectRatio=ratio;
    }
    ctx.drawImage(source,0,0,canvas.width,canvas.height);
  }
  requestAnimationFrame(drawLive);
}
drawLive();
source.onerror=()=>stateText.textContent="图传连接中断";
source.onload=()=>{if(!recorder||recorder.state==="inactive")stateText.textContent="实时画面"};

let streamReconnectTimer=null;
function reconnectStream(){
  clearTimeout(streamReconnectTimer);
  source.src="/stream.mjpg?t="+Date.now();
}
source.onerror=()=>{
  stateText.textContent="图传连接中断，正在重连";
  clearTimeout(streamReconnectTimer);
  streamReconnectTimer=setTimeout(reconnectStream,800);
};
source.onload=()=>{
  clearTimeout(streamReconnectTimer);
  if(!recorder||recorder.state==="inactive")stateText.textContent="实时画面";
};
window.addEventListener("online",reconnectStream);
document.addEventListener("visibilitychange",()=>{
  if(!document.hidden)reconnectStream();
});

function formatDuration(ms){
  const seconds=Math.max(0,ms)/1000;
  return String(Math.floor(seconds/60)).padStart(2,"0")+":"+
    (seconds%60).toFixed(1).padStart(4,"0");
}
function formatSize(bytes){
  return bytes<1048576?(bytes/1024).toFixed(0)+" KB":
    (bytes/1048576).toFixed(1)+" MB";
}
function preferredMime(){
  const types=["video/mp4;codecs=avc1.42E01E","video/webm;codecs=vp9",
    "video/webm;codecs=vp8","video/webm"];
  return types.find(type=>MediaRecorder.isTypeSupported(type))||"";
}
function pad2(value){return String(value).padStart(2,"0")}
function buildFilename(requirement,created,mime){
  const date=new Date(created);
  const stamp=date.getFullYear()+pad2(date.getMonth()+1)+pad2(date.getDate())+"_"+
    pad2(date.getHours())+pad2(date.getMinutes())+pad2(date.getSeconds());
  const extension=mime.includes("mp4")?"mp4":"webm";
  return "要求"+requirement+"_"+stamp+"."+extension;
}
function downloadRecording(item){
  const url=URL.createObjectURL(item.blob);
  const link=document.createElement("a");
  link.href=url;link.download=item.filename||buildFilename(
    item.requirement||1,item.created,item.mime||item.blob.type);
  document.body.appendChild(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

const openRequest=indexedDB.open("BallVisionDisplay",1);
openRequest.onupgradeneeded=()=>{
  openRequest.result.createObjectStore("recordings",{keyPath:"id",autoIncrement:true});
};
openRequest.onsuccess=()=>{db=openRequest.result;refreshRecordings()};
openRequest.onerror=()=>stateText.textContent="显示端存储不可用";

function saveRecording(blob,duration,requirement){
  return new Promise((resolve,reject)=>{
    if(!db){reject(new Error("database not ready"));return}
    const now=Date.now();
    const filename=buildFilename(requirement,now,blob.type);
    const transaction=db.transaction("recordings","readwrite");
    transaction.objectStore("recordings").add({
      created:now,requirement:requirement,filename:filename,
      duration:duration,mime:blob.type,blob:blob
    });
    transaction.oncomplete=()=>resolve({
      created:now,requirement:requirement,filename:filename,
      duration:duration,mime:blob.type,blob:blob
    });
    transaction.onerror=()=>reject(transaction.error);
  });
}
function playRecording(item,element){
  document.querySelectorAll(".recording").forEach(node=>node.classList.remove("active"));
  if(element)element.classList.add("active");
  if(currentUrl)URL.revokeObjectURL(currentUrl);
  currentUrl=URL.createObjectURL(item.blob);
  player.src=currentUrl;
  player.play().catch(()=>{});
}
function refreshRecordings(autoPlayNewest=false){
  if(!db)return;
  const request=db.transaction("recordings","readonly")
    .objectStore("recordings").getAll();
  request.onsuccess=()=>{
    const items=request.result.sort((a,b)=>
      (Number(a.requirement||999)-Number(b.requirement||999))||
      (b.created-a.created));
    list.replaceChildren();
    if(!items.length){
      const empty=document.createElement("div");
      empty.className="empty";empty.textContent="尚未保存录像";list.appendChild(empty);
      return;
    }
    items.forEach((item,index)=>{
      const row=document.createElement("div");row.className="recording";
      const info=document.createElement("div");
      const title=document.createElement("strong");
      title.textContent=item.filename||buildFilename(
        item.requirement||1,item.created,item.mime||item.blob.type);
      const detail=document.createElement("span");
      detail.textContent=new Date(item.created).toLocaleString()+" · "+
        formatDuration(item.duration)+" · "+formatSize(item.blob.size);
      const download=document.createElement("button");
      download.className="download";download.textContent="下载";
      download.onclick=event=>{event.stopPropagation();downloadRecording(item)};
      info.append(title,detail);row.append(info,download);
      row.onclick=()=>playRecording(item,row);
      list.appendChild(row);
      if(autoPlayNewest&&index===0)playRecording(item,row);
    });
  };
}

startBtn.onclick=()=>{
  if(!db){stateText.textContent="显示端存储正在初始化";return}
  activeRequirement=Math.max(1,Math.floor(Number(requirementInput.value)||1));
  requirementInput.value=String(activeRequirement);
  const stream=canvas.captureStream(30);
  const mime=preferredMime();
  chunks=[];
  recorder=new MediaRecorder(stream,mime?{
    mimeType:mime,videoBitsPerSecond:4000000
  }:undefined);
  recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data)};
  recorder.onstop=async()=>{
    const duration=performance.now()-startedAt;
    const blob=new Blob(chunks,{type:recorder.mimeType});
    try{
      const item=await saveRecording(blob,duration,activeRequirement);
      downloadRecording(item);
      refreshRecordings(true);
      stateText.textContent="已保存 "+item.filename;
      requirementInput.value=String(activeRequirement+1);
    }catch(error){
      console.error(error);stateText.textContent="录像保存失败";
    }
  };
  recorder.start(1000);
  startedAt=performance.now();
  timerId=setInterval(()=>timer.textContent=formatDuration(
    performance.now()-startedAt),100);
  startBtn.disabled=true;stopBtn.disabled=false;stateText.textContent="录像中";
};
stopBtn.onclick=()=>{
  if(recorder&&recorder.state!=="inactive")recorder.stop();
  clearInterval(timerId);startBtn.disabled=false;stopBtn.disabled=true;
};
</script>
</body>
</html>""".encode("utf-8")


class MjpegHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5.0)
        self.connection.setsockopt(
            socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
        )

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(WEB_PAGE)))
            self.end_headers()
            self.wfile.write(WEB_PAGE)
            return
        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache")
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.end_headers()
        seen_sequence = -1
        while not stop_event.is_set():
            with jpeg_condition:
                jpeg_condition.wait_for(
                    lambda: (
                        latest_jpeg_seq != seen_sequence
                        or stop_event.is_set()
                    ),
                    timeout=1.0,
                )
                data = latest_jpeg
                seen_sequence = latest_jpeg_seq
            if data is None:
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(data)}\r\n\r\n".encode()
                )
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (
                BrokenPipeError,
                ConnectionResetError,
                TimeoutError,
                OSError,
            ):
                break


def web_server_worker():
    server = ThreadingHTTPServer((MJPEG_HOST, MJPEG_PORT), MjpegHandler)
    server.daemon_threads = True
    server.timeout = 0.2
    print(
        f"[MJPEG] http://{MJPEG_HOST}:{MJPEG_PORT}/  "
        f"{MJPEG_WIDTH}px Q{MJPEG_QUALITY} {MJPEG_MAX_FPS}FPS"
    )
    if STREAM_ROI_ENABLED:
        print(
            "[MJPEG] ROI normalized="
            f"({STREAM_ROI_X1:.3f},{STREAM_ROI_Y1:.3f})-"
            f"({STREAM_ROI_X2:.3f},{STREAM_ROI_Y2:.3f})"
        )
    print(
        "[MJPEG] source="
        + ("annotated detection frame" if STREAM_ANNOTATED else "raw frame")
    )
    if STREAM_ANNOTATED and STREAM_PERSPECTIVE:
        print(
            f"[MJPEG] axis warp={STREAM_WARP_WIDTH}x{STREAM_WARP_HEIGHT} "
            f"top={STREAM_WARP_TOP_PX:g}px "
            f"bottom={STREAM_WARP_BOTTOM_PX:g}px"
        )
    while not stop_event.is_set():
        server.handle_request()
    server.server_close()


def open_uart():
    if not UART_ENABLED or serial is None:
        print("[UART] Off")
        return None
    try:
        connection = serial.Serial(
            UART_PORT,
            UART_BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0,
            write_timeout=0.05,
        )
        print(f"[UART] {UART_PORT} @ {UART_BAUD}, 8N1")
        print(
            "[UART] BB 06 VV_H VV_L PP_H PP_L A5 5A, "
            "int16 BE x0.01; PP=position relative to center"
        )
        return connection
    except Exception as error:
        print(f"[UART] Off: {error}")
        return None


def uart_int16(value):
    scaled = int(round(float(value) * UART_VALUE_SCALE))
    return max(-32768, min(32767, scaled))


def build_uart_packet(velocity_cm_s, position_cm):
    return struct.pack(
        ">BBhhBB",
        0xBB,
        0x06,
        uart_int16(velocity_cm_s),
        uart_int16(position_cm),
        0xA5,
        0x5A,
    )


def main():
    cv2.setNumThreads(2)
    uart = open_uart()
    uart_tx_count = 0
    uart_last_log_time = 0.0
    calibration = AxisCalibration()
    velocity_estimator = BallVelocityEstimator()
    threads = [
        threading.Thread(target=capture_worker, name="capture"),
        threading.Thread(target=jpeg_encoder_worker, name="jpeg"),
        threading.Thread(target=web_server_worker, name="web", daemon=True),
    ]
    core_masks = [
        RKNNLite.NPU_CORE_0,
        RKNNLite.NPU_CORE_1,
        RKNNLite.NPU_CORE_2,
    ]
    threads.extend(
        threading.Thread(
            target=inference_worker,
            args=(index, core_masks[index]),
            name=f"npu-{index}",
        )
        for index in range(NPU_WORKERS)
    )
    for thread in threads:
        thread.start()

    if LOCAL_DISPLAY:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(
            WINDOW_NAME, calibration.mouse_callback
        )
        print(f"[Display] Opened: {WINDOW_NAME}")
        print(
            "[Display] Press C, then click CENTER 0.00 cm, LEFT "
            f"{-calibration.length_cm / 2.0:+.2f} cm inner stop, "
            f"and RIGHT {calibration.length_cm / 2.0:+.2f} cm inner stop."
        )
    else:
        print("[Display] Headless mode")
    print("[Main] INT8 only, 3 NPU workers. Ctrl+C to stop.")
    print(
        f"[Velocity] update={VELOCITY_UPDATE_INTERVAL_S * 1000:.0f}ms "
        f"window={VELOCITY_WINDOW_S * 1000:.0f}ms, cm/s"
    )

    stabilizer = BallStabilizer()
    pending_results = {}
    skipped_sequences = set()
    dropped_frame_count = 0
    next_sequence = 1
    processed_frames = 0
    try:
        while not stop_event.is_set():
            try:
                result = inference_result_queue.get(timeout=0.02)
                pending_results[result["seq"]] = result
                # Gather results that completed at nearly the same time.
                while True:
                    result = inference_result_queue.get_nowait()
                    pending_results[result["seq"]] = result
            except Empty:
                pass

            while True:
                try:
                    skipped_sequences.add(
                        dropped_sequence_queue.get_nowait()
                    )
                    dropped_frame_count += 1
                except Empty:
                    break
            while next_sequence in skipped_sequences:
                skipped_sequences.discard(next_sequence)
                next_sequence += 1

            processed_this_cycle = False
            while next_sequence in pending_results:
                processed_this_cycle = True
                result = pending_results.pop(next_sequence)
                processed_frames += 1
                frame = result["frame"].copy()
                balls = stabilizer.update(
                    result["balls"], frame.shape[1], frame.shape[0]
                )
                now = time.perf_counter()
                display_times.append(now)
                while display_times and display_times[0] < now - 1.0:
                    display_times.popleft()
                latency_ms = (
                    now - result["capture_time"]
                ) * 1000.0

                best = None
                for ball in balls:
                    x1, y1, x2, y2, cx, cy, radius, confidence = ball
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"{confidence:.2f}",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )
                    if best is None or confidence > best[3]:
                        best = (cx, cy, radius, confidence)

                ball_center = None if best is None else best[:2]
                axis_position = calibration.draw(frame, ball_center)
                position_cm = (
                    None if axis_position is None else axis_position[0]
                )
                velocity_cm_s = velocity_estimator.update(
                    result["capture_time"], position_cm
                )
                velocity_text = (
                    "VEL -- cm/s"
                    if velocity_cm_s is None
                    else f"VEL {velocity_cm_s:+.2f} cm/s"
                )
                cv2.putText(
                    frame,
                    velocity_text,
                    (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 0),
                    2,
                )

                if (
                    uart is not None
                    and velocity_estimator.output_updated
                    and position_cm is not None
                    and velocity_cm_s is not None
                ):
                    packet = build_uart_packet(
                        velocity_cm_s, position_cm
                    )
                    try:
                        uart.write(packet)
                        uart_tx_count += 1
                        if now - uart_last_log_time >= 1.0:
                            print(
                                f"[UART TX #{uart_tx_count}] "
                                f"{packet.hex(' ').upper()}  "
                                f"VEL={velocity_cm_s:+.2f}cm/s "
                                f"POS={position_cm:+.2f}cm"
                            )
                            uart_last_log_time = now
                    except (
                        serial.SerialException,
                        serial.SerialTimeoutException,
                    ) as error:
                        print(f"[UART] Write failed: {error}")
                        uart.close()
                        uart = None

                cv2.putText(
                    frame,
                    f"E2E {len(display_times):.0f}  "
                    f"LAT {latency_ms:.1f}ms",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"NPU {result['npu_fps']:.0f}  "
                    f"CAM {result['camera_fps']:.0f}  "
                    f"BALL {len(balls)}  MAX {result['max_conf']:.2f}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"SEQ {next_sequence}  SKIP {dropped_frame_count}  "
                    f"DQ {frame_queue.qsize()}  "
                    f"REORDER {len(pending_results)}",
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    1,
                )

                if STREAM_ANNOTATED:
                    stream_frame = frame
                    prepared = False
                    if STREAM_PERSPECTIVE:
                        warped = calibration.warp_stream_band(frame)
                        if warped is not None:
                            stream_frame = warped
                            prepared = True
                    put_latest(
                        encode_queue,
                        (next_sequence, stream_frame, prepared),
                    )

                if LOCAL_DISPLAY:
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        request_stop("ESC pressed")
                        break
                    if key in (ord("c"), ord("C")):
                        calibration.begin()
                        velocity_estimator.reset()
                next_sequence += 1
                while next_sequence in skipped_sequences:
                    skipped_sequences.discard(next_sequence)
                    next_sequence += 1

            if LOCAL_DISPLAY and not processed_this_cycle:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    request_stop("ESC pressed")
                if key in (ord("c"), ord("C")):
                    calibration.begin()
                    velocity_estimator.reset()
    except KeyboardInterrupt:
        request_stop("Ctrl+C")
    finally:
        request_stop("shutdown")
        for thread in threads:
            if not thread.daemon:
                thread.join(timeout=2.0)
        if uart is not None:
            uart.close()
        if LOCAL_DISPLAY:
            cv2.destroyAllWindows()
        print("[Main] Exit")


if __name__ == "__main__":
    main()
