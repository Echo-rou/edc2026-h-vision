#!/usr/bin/env python3
""" 钢球检测 - 多线程 + NPU 加速 + 显示器实时画面 """
import cv2, numpy as np, serial, time, threading
from collections import deque
from queue import Queue, Empty
from rknnlite.api import RKNNLite

# ==================== 配置 ====================
MODEL_FILE  = "ball_best_int8_split.rknn"
CAMERA_ID   = 0
IMG_SIZE    = 640
CONF_THRESH = 0.5
NMS_THRESH  = 0.45
UART_PORT   = "/dev/ttyS1"
UART_BAUD   = 115200
NPU_WORKERS = 3

# ==================== 全局状态 ====================
frame_queue = Queue(maxsize=NPU_WORKERS)
result_lock  = threading.Lock()
latest_result = {
    "seq": -1,
    "frame": None,
    "balls": [],
    "fps": 0.0,
    "max_conf": 0.0,
    "capture_fps": 0.0,
    "capture_time": 0.0,
}
completion_times = deque()
capture_times = deque()
display_times = deque()
running = True

# ==================== YOLO 工具函数 ====================
def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size/h, size/w)
    nh, nw = int(h*scale), int(w*scale)
    canvas = np.zeros((size,size,3), dtype=np.uint8)
    resized = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh))
    canvas[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = resized
    return canvas, scale, (size-nw)//2, (size-nh)//2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def postprocess(output, scale, xo, yo, iw, ih):
    data = output[0].T
    mask = data[:,4] > CONF_THRESH
    f = data[mask]
    if len(f) == 0:
        return [], []
    xc, yc, bw, bh = f[:,0], f[:,1], f[:,2], f[:,3]
    x1 = np.clip(((xc-bw/2)-xo)/scale, 0, iw)
    y1 = np.clip(((yc-bh/2)-yo)/scale, 0, ih)
    x2 = np.clip(((xc+bw/2)-xo)/scale, 0, iw)
    y2 = np.clip(((yc+bh/2)-yo)/scale, 0, ih)
    boxes = np.stack([x1,y1,x2,y2], axis=1)
    return f[:,4].tolist(), boxes.tolist()

# ==================== 线程 1: 摄像头采集 ====================
def capture_thread(camera_id):
    global running
    pipeline = (
        f"v4l2src device=/dev/video{camera_id} io-mode=2 ! "
        "image/jpeg,width=640,height=480,framerate=90/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    backend = "GStreamer"
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        backend = "V4L2"
    if not cap.isOpened():
        print(f"[Thread-1] Cannot open /dev/video{camera_id}")
        running = False
        return
    if backend == "V4L2":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 90)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fmt = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f"[Thread-1] Camera OK: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{cap.get(cv2.CAP_PROP_FPS):.1f} FPS, {fmt}, {backend}"
    )

    seq = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        now = time.perf_counter()
        with result_lock:
            capture_times.append(now)
            while capture_times and capture_times[0] < now - 1.0:
                capture_times.popleft()
            latest_result["capture_fps"] = float(len(capture_times))
        seq += 1
        # 队列满时丢弃最旧帧，优先保证低延迟。
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put((seq, now, frame))
    cap.release()

# ==================== NPU 推理线程 ====================
def inference_thread(worker_id, core_mask):
    global running, latest_result
    print(f"[NPU-{worker_id}] Loading RKNN: {MODEL_FILE}")
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_FILE) != 0:
        print(f"[NPU-{worker_id}] Load failed")
        running = False
        return
    if rknn.init_runtime(core_mask=core_mask) != 0:
        print(f"[NPU-{worker_id}] Runtime init failed")
        running = False
        rknn.release()
        return
    print(f"[NPU-{worker_id}] Ready")

    while running:
        try:
            seq, capture_time, frame = frame_queue.get(timeout=0.5)
        except Empty:
            continue

        oh, ow = frame.shape[:2]
        inp, s, xo, yo = letterbox(frame, IMG_SIZE)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0)
        outputs = rknn.inference(inputs=[inp], data_format=["nhwc"])
        if not outputs:
            print(f"[NPU-{worker_id}] Inference failed")
            running = False
            break
        if len(outputs) == 2:
            out = np.concatenate((outputs[0], outputs[1]), axis=1)
        else:
            out = outputs[0]
        max_conf = float(np.max(out[0, 4]))
        confs, boxes_raw = postprocess(out, s, xo, yo, ow, oh)
        balls = []

        if boxes_raw:
            nms_boxes = [
                [x1, y1, x2 - x1, y2 - y1]
                for x1, y1, x2, y2 in boxes_raw
            ]
            idx = cv2.dnn.NMSBoxes(
                nms_boxes, confs, CONF_THRESH, NMS_THRESH, top_k=100
            )
            for i in idx.flatten():
                x1,y1,x2,y2 = map(int, boxes_raw[i])
                cx,cy = (x1+x2)//2, (y1+y2)//2
                r = max((x2-x1)//2, (y2-y1)//2)
                balls.append((x1,y1,x2,y2,cx,cy,r,confs[i]))

        with result_lock:
            now = time.perf_counter()
            completion_times.append(now)
            while completion_times and completion_times[0] < now - 1.0:
                completion_times.popleft()
            fps = float(len(completion_times))
            if seq > latest_result["seq"]:
                capture_fps = latest_result["capture_fps"]
                latest_result = {
                    "seq": seq,
                    "frame": frame,
                    "balls": balls,
                    "fps": fps,
                    "max_conf": max_conf,
                    "capture_fps": capture_fps,
                    "capture_time": capture_time,
                }

    rknn.release()

# ==================== 主线程: 显示 + 串口 ====================
def main():
    global running

    cv2.setNumThreads(2)

    # 尝试开串口
    uart = None
    try:
        uart = serial.Serial(UART_PORT, UART_BAUD, timeout=0.1)
        print(f"[Main] UART: {UART_PORT} @ {UART_BAUD}")
    except:
        print("[Main] UART off")

    # 每个 RK3588 NPU 核使用独立 Runtime，并行处理不同帧。
    t_cap = threading.Thread(target=capture_thread, args=(CAMERA_ID,), daemon=True)
    t_cap.start()
    core_masks = [
        RKNNLite.NPU_CORE_0,
        RKNNLite.NPU_CORE_1,
        RKNNLite.NPU_CORE_2,
    ]
    workers = [
        threading.Thread(
            target=inference_thread,
            args=(worker_id, core_masks[worker_id]),
            daemon=True,
        )
        for worker_id in range(NPU_WORKERS)
    ]
    for worker in workers:
        worker.start()
    print(f"[Main] Capture + {NPU_WORKERS} NPU workers. Press ESC to stop")

    # 创建窗口
    cv2.namedWindow("Ball Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Detection", 960, 540)

    displayed_seq = -1
    while running:
        with result_lock:
            seq = latest_result["seq"]
            frame = latest_result["frame"]
            balls = latest_result["balls"]
            fps   = latest_result["fps"]
            max_conf = latest_result["max_conf"]
            capture_fps = latest_result["capture_fps"]
            capture_time = latest_result["capture_time"]

        if frame is None or seq == displayed_seq:
            if cv2.waitKey(1) & 0xFF == 27:
                running = False
            time.sleep(0.001)
            continue
        displayed_seq = seq
        frame = frame.copy()
        display_time = time.perf_counter()
        display_times.append(display_time)
        while display_times and display_times[0] < display_time - 1.0:
            display_times.popleft()
        e2e_fps = float(len(display_times))
        latency_ms = (display_time - capture_time) * 1000.0

        # 画检测框
        best = None
        for (x1,y1,x2,y2,cx,cy,r,conf) in balls:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)
            cv2.putText(frame, f"{conf:.2f}", (x1,y1-6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            if best is None or conf > best[3]:
                best = (cx,cy,r,conf)

        # 串口发送
        if uart:
            if best:
                uart.write(f"B{best[0]},{best[1]},{best[2]}\n".encode())
            else:
                uart.write(b"BNO\n")

        # 显示 FPS 和检出数
        cv2.putText(
            frame,
            f"E2E: {e2e_fps:.0f} FPS  Lat: {latency_ms:.1f} ms",
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(
            frame,
            f"NPU: {fps:.0f}  CAM: {capture_fps:.0f}  "
            f"Balls: {len(balls)}  Max: {max_conf:.2f}",
            (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2
        )

        cv2.imshow("Ball Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            running = False
            break

    # 清理
    cv2.destroyAllWindows()
    if uart: uart.close()
    print("[Main] Exit")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        running = False
        print("\n[Main] Interrupted")
