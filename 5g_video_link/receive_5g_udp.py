#!/usr/bin/env python3
"""Windows receiver: low-latency display and required off-board recording."""

import argparse
import os
import socket
import struct
import time
from datetime import datetime

import cv2
import numpy as np

PACKET_MAGIC = b"BV5G"
PACKET_HEADER = struct.Struct("!4sIQHH")
ASSEMBLY_TIMEOUT = 0.15
DISCOVERY_MAGIC = b"BV5G_HELLO"


def parse_args():
    parser = argparse.ArgumentParser(description="Ball Vision 5 GHz UDP receiver")
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--sender", default="192.168.12.1")
    parser.add_argument("--discovery-port", type=int, default=5601)
    parser.add_argument("--output", default="received_recordings_5g")
    parser.add_argument("--record-fps", type=float, default=30.0)
    return parser.parse_args()


def cleanup(assemblies, now):
    expired = [
        frame_id for frame_id, assembly in assemblies.items()
        if now - assembly["created"] > ASSEMBLY_TIMEOUT
    ]
    for frame_id in expired:
        del assemblies[frame_id]


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.1)
    discovery_endpoint = (args.sender, args.discovery_port)
    last_discovery = 0.0

    assemblies = {}
    last_frame_id = -1
    writer = None
    record_path = None
    record_start = None
    written_frames = 0
    display_times = []
    minimum_clock_delta = None

    print(f"[Receiver] Listening on UDP {args.port}")
    print("[Receiver] Press ESC to stop and finish the off-board recording")
    try:
        while True:
            now = time.monotonic()
            if now - last_discovery >= 1.0:
                sock.sendto(DISCOVERY_MAGIC, discovery_endpoint)
                last_discovery = now
            try:
                packet, _ = sock.recvfrom(2048)
            except socket.timeout:
                cleanup(assemblies, time.monotonic())
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            if len(packet) <= PACKET_HEADER.size:
                continue
            magic, frame_id, timestamp_ms, packet_index, packet_count = PACKET_HEADER.unpack(
                packet[:PACKET_HEADER.size]
            )
            if magic != PACKET_MAGIC or packet_count == 0 or packet_index >= packet_count:
                continue
            if frame_id <= last_frame_id:
                continue

            now = time.monotonic()
            assembly = assemblies.setdefault(
                frame_id,
                {"created": now, "timestamp_ms": timestamp_ms, "count": packet_count, "parts": {}},
            )
            if assembly["count"] != packet_count:
                del assemblies[frame_id]
                continue
            assembly["parts"][packet_index] = packet[PACKET_HEADER.size:]
            cleanup(assemblies, now)
            if len(assembly["parts"]) != packet_count:
                continue

            encoded = b"".join(assembly["parts"][index] for index in range(packet_count))
            del assemblies[frame_id]
            last_frame_id = frame_id
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            clock_delta = int(time.time() * 1000) - timestamp_ms
            if minimum_clock_delta is None or clock_delta < minimum_clock_delta:
                minimum_clock_delta = clock_delta
            queue_delay_ms = max(0, clock_delta - minimum_clock_delta)
            display_times.append(now)
            while display_times and display_times[0] < now - 1.0:
                display_times.pop(0)
            cv2.putText(
                frame, f"RX {len(display_times)} FPS  Queue {queue_delay_ms} ms", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
            )

            if writer is None:
                height, width = frame.shape[:2]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                record_path = os.path.abspath(
                    os.path.join(args.output, f"ball_test_5g_{timestamp}.mp4")
                )
                writer = cv2.VideoWriter(
                    record_path, cv2.VideoWriter_fourcc(*"mp4v"), args.record_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create recording: {record_path}")
                record_start = now
                print(f"[Receiver] Recording: {record_path}")

            target_frames = int((now - record_start) * args.record_fps) + 1
            while written_frames < target_frames:
                writer.write(frame)
                written_frames += 1
            cv2.imshow("Ball Vision 5G - Receiving and Recording", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        sock.close()
        if writer is not None:
            writer.release()
            print(f"[Receiver] Saved: {record_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
