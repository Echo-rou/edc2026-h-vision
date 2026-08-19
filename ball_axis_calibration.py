#!/usr/bin/env python3
"""Standalone lossless frame capture and 25.5 cm tube-axis calibration.

This tool does not load RKNN. Stop the detector before running it so the
camera is owned by only one process.

Controls:
  C           freeze current frame and start a new calibration
  left click  left -12.75 cm inner stop, then right +12.75 cm inner stop
  right click optional ball center for a manual position check
  S           save lossless raw/annotated PNG files
  L/SPACE     return to live view
  ESC         exit
"""
import json
import os

import cv2
import numpy as np


CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 90
AXIS_LENGTH_CM = 25.5
MIN_AXIS_PIXELS = 100.0
WINDOW_NAME = "BallVision Axis Calibration"

HOME = os.path.expanduser("~")
CALIBRATION_FILE = os.path.join(HOME, "ball_axis_calibration.json")
RAW_IMAGE_FILE = os.path.join(HOME, "ball_calibration_raw.png")
ANNOTATED_IMAGE_FILE = os.path.join(
    HOME, "ball_calibration_annotated.png"
)


class CalibrationTool:
    def __init__(self):
        self.frame = None
        self.frozen_frame = None
        self.collecting = False
        self.normalized_points = []
        self.test_ball = None
        # Calibration sessions start clean so stale P0/P1 markers are never
        # mistaken for newly selected points. Set LOAD_EXISTING_CALIBRATION=1
        # only when an existing calibration must be inspected.
        if os.environ.get("LOAD_EXISTING_CALIBRATION") == "1":
            self.load_existing()
        else:
            print("[Calibration] Clean session; existing points not loaded.")

    def load_existing(self):
        if not os.path.isfile(CALIBRATION_FILE):
            return
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            points = data.get("normalized_points", [])
            if len(points) == 2:
                self.normalized_points = [
                    (float(point[0]), float(point[1]))
                    for point in points
                ]
                print(f"[Calibration] Loaded: {CALIBRATION_FILE}")
        except Exception as error:
            print(f"[Calibration] Existing file ignored: {error}")

    def current_source(self):
        if self.frozen_frame is not None:
            return self.frozen_frame
        return self.frame

    def map_click_to_normalized(self, x, y):
        source = self.current_source()
        if source is None:
            return None
        image_height, image_width = source.shape[:2]
        # The calibration window is WINDOW_AUTOSIZE at the native camera
        # resolution, so callback coordinates are already image pixels.
        image_x = np.clip(x, 0, image_width - 1)
        image_y = np.clip(y, 0, image_height - 1)
        print(
            f"[Click] image=({int(image_x)}, {int(image_y)}) "
            f"size={image_width}x{image_height}"
        )
        return float(image_x / image_width), float(image_y / image_height)

    def mouse_callback(self, event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONUP and self.collecting:
            point = self.map_click_to_normalized(x, y)
            if point is None:
                return
            source = self.current_source()
            height, width = source.shape[:2]
            if self.normalized_points:
                first = np.array(
                    [
                        self.normalized_points[0][0] * width,
                        self.normalized_points[0][1] * height,
                    ],
                    dtype=np.float32,
                )
                second = np.array(
                    [point[0] * width, point[1] * height],
                    dtype=np.float32,
                )
                distance = float(np.linalg.norm(second - first))
                if distance < MIN_AXIS_PIXELS:
                    print(
                        f"[Calibration] P1 rejected: only "
                        f"{distance:.1f}px from P0; click the opposite stop."
                    )
                    return
            self.normalized_points.append(point)
            if len(self.normalized_points) == 1:
                print(
                    "[Calibration] Left -12.75 cm accepted. Click the "
                    "right +12.75 cm white-stop INNER face on the tube centerline."
                )
            elif len(self.normalized_points) == 2:
                self.collecting = False
                self.test_ball = None
                self.save_all()
                print(
                    "[Calibration] Complete. Optional: right-click the "
                    "ball center to verify its position."
                )
        elif (
            event == cv2.EVENT_RBUTTONUP
            and len(self.normalized_points) == 2
        ):
            point = self.map_click_to_normalized(x, y)
            if point is None:
                return
            self.test_ball = point
            self.save_all()

    def start_calibration(self):
        if self.frame is None:
            return
        self.frozen_frame = self.frame.copy()
        self.normalized_points = []
        self.test_ball = None
        self.collecting = True
        print(
            "[Calibration] Frozen. Click the LEFT -12.75 cm white-stop "
            "INNER face on the tube centerline."
        )

    def pixel_points(self, frame):
        if len(self.normalized_points) != 2:
            return None
        height, width = frame.shape[:2]
        return [
            np.array(
                [point[0] * width, point[1] * height],
                dtype=np.float32,
            )
            for point in self.normalized_points
        ]

    def test_position(self, frame):
        points = self.pixel_points(frame)
        if points is None or self.test_ball is None:
            return None
        height, width = frame.shape[:2]
        ball = np.array(
            [self.test_ball[0] * width, self.test_ball[1] * height],
            dtype=np.float32,
        )
        start, end = points
        axis = end - start
        denominator = float(np.dot(axis, axis))
        if denominator < 1.0:
            return None
        fraction = float(np.dot(ball - start, axis) / denominator)
        projection = start + fraction * axis
        centered_cm = (fraction - 0.5) * AXIS_LENGTH_CM
        return centered_cm, ball, projection

    def annotate(self, frame):
        output = frame.copy()
        height, width = output.shape[:2]
        for index, normalized in enumerate(self.normalized_points):
            point = (
                int(round(normalized[0] * width)),
                int(round(normalized[1] * height)),
            )
            cv2.circle(output, point, 7, (0, 165, 255), 2)
            cv2.putText(
                output,
                "-12.75" if index == 0 else "+12.75",
                (point[0] + 8, point[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )
        points = self.pixel_points(output)
        if points is not None:
            start, end = points
            cv2.line(
                output,
                tuple(np.rint(start).astype(int)),
                tuple(np.rint(end).astype(int)),
                (255, 0, 255),
                2,
            )
            half_length = AXIS_LENGTH_CM / 2.0
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
                fraction = centimeters / AXIS_LENGTH_CM + 0.5
                tick = start + fraction * (end - start)
                tick_i = tuple(np.rint(tick).astype(int))
                cv2.circle(output, tick_i, 4, (255, 0, 255), -1)
                cv2.putText(
                    output,
                    f"{centimeters:+g}" if centimeters != 0.0 else "0",
                    (tick_i[0] + 4, max(14, tick_i[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 0, 255),
                    1,
                )

        position = self.test_position(output)
        if position is not None:
            centimeters, ball, projection = position
            ball_i = tuple(np.rint(ball).astype(int))
            projection_i = tuple(np.rint(projection).astype(int))
            cv2.circle(output, ball_i, 6, (0, 0, 255), -1)
            cv2.circle(output, projection_i, 5, (255, 255, 0), -1)
            cv2.line(output, ball_i, projection_i, (255, 255, 0), 1)
            cv2.putText(
                output,
                f"POS {centimeters:+.2f} cm  (CENTER = 0)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

        if self.collecting:
            target = "0 cm" if not self.normalized_points else "25.5 cm"
            message = f"Click {target} INNER stop at tube centerline"
        elif points is None:
            message = "Press C to freeze and calibrate"
        else:
            message = "Calibrated | Right-click ball | S save | L live"
        cv2.putText(
            output,
            message,
            (10, output.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        return output

    def save_all(self):
        source = self.current_source()
        if source is None:
            return
        if len(self.normalized_points) == 2:
            points = self.pixel_points(source)
            axis_pixels = float(np.linalg.norm(points[1] - points[0]))
            if axis_pixels < MIN_AXIS_PIXELS:
                raise RuntimeError(
                    f"invalid calibration axis: {axis_pixels:.1f}px"
                )
            data = {
                "length_cm": AXIS_LENGTH_CM,
                "normalized_points": self.normalized_points,
            }
            with open(CALIBRATION_FILE, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
            print(f"[Saved] calibration: {CALIBRATION_FILE}")
        annotated = self.annotate(source)
        png_options = [int(cv2.IMWRITE_PNG_COMPRESSION), 2]
        if not cv2.imwrite(RAW_IMAGE_FILE, source, png_options):
            raise RuntimeError(f"cannot save {RAW_IMAGE_FILE}")
        if not cv2.imwrite(
            ANNOTATED_IMAGE_FILE, annotated, png_options
        ):
            raise RuntimeError(f"cannot save {ANNOTATED_IMAGE_FILE}")
        print(f"[Saved] raw PNG:       {RAW_IMAGE_FILE}")
        print(f"[Saved] annotated PNG: {ANNOTATED_IMAGE_FILE}")


def open_camera():
    pipeline = (
        f"v4l2src device=/dev/video{CAMERA_ID} io-mode=2 ! "
        f"image/jpeg,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},"
        f"framerate={CAMERA_FPS}/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if capture.isOpened():
        return capture, "GStreamer MJPEG"
    capture.release()
    capture = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture, "V4L2 MJPEG"


def main():
    capture, backend = open_camera()
    if not capture.isOpened():
        raise RuntimeError(f"cannot open /dev/video{CAMERA_ID}")
    print(
        f"[Camera] {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{capture.get(cv2.CAP_PROP_FPS):.1f} FPS, {backend}"
    )
    print(__doc__)

    tool = CalibrationTool()
    # Native-size window is required for exact mouse-to-pixel calibration.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, tool.mouse_callback)
    try:
        while True:
            if tool.frozen_frame is None:
                ok, frame = capture.read()
                if not ok:
                    continue
                tool.frame = frame
            source = tool.current_source()
            if source is None:
                continue
            cv2.imshow(WINDOW_NAME, tool.annotate(source))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("c"), ord("C")):
                tool.start_calibration()
            elif key in (ord("s"), ord("S")):
                tool.save_all()
            elif key in (ord("l"), ord("L"), 32):
                tool.frozen_frame = None
                tool.collecting = False
                print("[Calibration] Live view")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
