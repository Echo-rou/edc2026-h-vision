"""
货物自动分拣系统 —— 视觉识别模块

功能：
1. 颜色识别（Green/Orange/Blue/Pink）
2. 形状分类（Cube/Cuboid/Sphere）
3. 透视变换：像素坐标 → 世界坐标（mm）
4. 区域划分：A区 / B区 / 货架
5. 结构化数据输出，供 IK 和任务调度模块调用

标定流程：
  运行后按 'c' 进入标定模式，鼠标依次点击场地的 4 个角点
  （顺序：左上 → 右上 → 右下 → 左下），按 's' 保存，按 'r' 重来。

透视变换说明：
  场地是白色底板 + 黑色边界线。4 个角点的世界坐标定义在 FIELD_CORNERS_WORLD 中。
"""

import cv2
import numpy as np
import json
import os
import struct
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# 脚本所在目录（固定基准，不依赖当前工作目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 配置 — 按实际硬件修改
# ============================================================

CAMERA_INDEX = 1                # 摄像头索引
MIN_AREA = 1000                 # 最小轮廓面积（过滤噪点）

# 相机内参文件路径（棋盘格标定后生成）
CAMERA_MATRIX_PATH = os.path.join(SCRIPT_DIR, "camera_matrix.npy")
DIST_COEFFS_PATH = os.path.join(SCRIPT_DIR, "dist_coeffs.npy")

# 透视变换矩阵保存路径
HOMOGRAPHY_PATH = os.path.join(SCRIPT_DIR, "homography.npy")

# 场地 4 个角点的世界坐标（mm），原点 = 机械臂基座中心
# 顺序：左上 → 右上 → 右下 → 左下
# 以下为示例值，需按实际场地测量后修改
FIELD_CORNERS_WORLD = np.array([
    [-200.9, 199.8],   # 左上 (A区远角)
    [ 201.3, 202.0],   # 右上 (B区远角)
    [ 201.3,  51.0],   # 右下 (B区近角)
    [-200.9,  51.0],   # 左下 (A区近角)
], dtype=np.float32)

# 区域边界（世界坐标，mm），用于区分 A区 / B区
# A区在左侧(X<0)，B区在右侧(X>0)
REGION_BOUNDARY_X = 0

# --- ROI 区域（像素坐标，限制识别范围，避免误识别货架/杂物）---
# 格式: (x, y, w, h)，设为 None 则禁用 ROI 限制
# 摄像头固定后，运行程序按 'i' 鼠标框选设置
ROI_ZONE_A = (192, 242, 121, 97)
ROI_ZONE_B = (317, 244, 124, 96)

# 颜色定义（HSV 范围）
COLOR_RANGES = {
    "Green":  ((35, 60, 50),  (85, 255, 255)),
    "Orange": ((5, 100, 80),  (25, 255, 255)),
    "Blue":   ((90, 70, 50),  (130, 255, 255)),
    "Pink":   ((145, 45, 70), (179, 255, 255)),
}

# 货物实际尺寸（mm），用于区分 Cube / Cuboid / Sphere
# 正方体 50×50   长方体 70×40   球体 Ø50
CUBE_EDGE_MM = 50
CUBOID_LONG_MM = 70
CUBOID_SHORT_MM = 40
SPHERE_DIAMETER_MM = 50

# --- 串口通信（香橙派→STM32）---
SERIAL_PORT = "COM3"             # 香橙派上改成 "/dev/ttyS1"
SERIAL_BAUD = 115200
SERIAL_ENABLED = False           # PC 调试时 False，上香橙派后改 True

# ============================================================
# 相机标定 & 透视变换
# ============================================================

def load_camera_params():
    """加载相机内参和畸变系数，如果文件不存在则返回 None"""
    if os.path.exists(CAMERA_MATRIX_PATH) and os.path.exists(DIST_COEFFS_PATH):
        K = np.load(CAMERA_MATRIX_PATH)
        D = np.load(DIST_COEFFS_PATH)
        return K, D
    return None, None


def load_homography():
    """加载透视变换矩阵"""
    if os.path.exists(HOMOGRAPHY_PATH):
        return np.load(HOMOGRAPHY_PATH)
    return None


def pixel_to_world(pixel_pts, H):
    """
    像素坐标 → 世界坐标
    pixel_pts: (N, 2) 或 (2,) numpy 数组
    H: 3×3 单应矩阵
    返回: (N, 2) 世界坐标（mm）
    """
    pts = np.atleast_2d(pixel_pts).astype(np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, H)
    return world.reshape(-1, 2)


def calibrate_perspective_interactive():
    """
    交互式标定：让用户点击场地 4 个角点，计算透视变换矩阵
    返回 3×3 的 H 矩阵，或 None（取消）
    """
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("无法打开摄像头")
        return None

    corners = []
    frame_copy = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal corners, frame_copy
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            frame_copy = frame.copy()
            for i, pt in enumerate(corners):
                cv2.circle(frame_copy, pt, 6, (0, 0, 255), -1)
                cv2.putText(frame_copy, str(i + 1), (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(frame_copy, [pts], True, (0, 255, 0), 2)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    print("\n===== 透视变换标定 =====")
    print("请按顺序点击场地的 4 个角点：")
    print("  1. 左上角 → 2. 右上角 → 3. 右下角 → 4. 左下角")
    print("按 's' 保存, 'r' 重来, 'q' 退出\n")

    while True:
        ok, frame = camera.read()
        if not ok:
            continue

        if frame_copy is not None:
            cv2.imshow("Calibration", frame_copy)
        else:
            cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            camera.release()
            cv2.destroyWindow("Calibration")
            return None
        elif key == ord('r'):
            corners.clear()
            frame_copy = None
            print("已重置，请重新点击 4 个角点")
        elif key == ord('s') and len(corners) == 4:
            break

    camera.release()
    cv2.destroyWindow("Calibration")

    src_pts = np.array(corners, dtype=np.float32)
    dst_pts = FIELD_CORNERS_WORLD.astype(np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    np.save(HOMOGRAPHY_PATH, H)
    print(f"透视变换矩阵已保存到 {HOMOGRAPHY_PATH}")
    print(f"源点（像素）: {src_pts.tolist()}")
    print(f"目标点（mm）: {dst_pts.tolist()}")
    return H


# ============================================================
# 区域划分
# ============================================================

def classify_region(world_xy):
    """
    根据世界坐标判断货物属于哪个区域
    返回 "A" 或 "B"
    """
    x, y = world_xy
    if x < REGION_BOUNDARY_X:
        return "A"
    else:
        return "B"


# ============================================================
# ROI 蒙版
# ============================================================

def build_roi_mask(frame_shape):
    """
    根据 ROI_ZONE_A / ROI_ZONE_B 生成二值蒙版
    蒙版内（白色）允许识别，蒙版外（黑色）屏蔽
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if ROI_ZONE_A is not None:
        x, y, w, h = ROI_ZONE_A
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    if ROI_ZONE_B is not None:
        x, y, w, h = ROI_ZONE_B
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    if ROI_ZONE_A is None and ROI_ZONE_B is None:
        mask[...] = 255  # 未设置 ROI 时全图允许

    return mask


def draw_roi_zones(frame):
    """在画面上绘制 ROI 区域框"""
    if ROI_ZONE_A is not None:
        x, y, w, h = ROI_ZONE_A
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Zone A", (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if ROI_ZONE_B is not None:
        x, y, w, h = ROI_ZONE_B
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Zone B", (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def setup_roi_interactive():
    """
    交互式设置 ROI 区域：鼠标框选 A区 和 B区
    操作：按住左键拖拽框选，松开完成
         先框 A区，再框 B区，按 's' 保存，'r' 重来
    """
    global ROI_ZONE_A, ROI_ZONE_B

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("无法打开摄像头")
        return

    zones = []          # [(x, y, w, h), ...]
    drawing = False
    start_pt = (0, 0)
    current_rect = None
    frame_copy = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_pt, current_rect, frame_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_rect = (start_pt[0], start_pt[1],
                            x - start_pt[0], y - start_pt[1])
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if current_rect is not None and len(zones) < 2:
                zones.append(current_rect)
                current_rect = None
                start_pt = (0, 0)

    cv2.namedWindow("ROI Setup")
    cv2.setMouseCallback("ROI Setup", mouse_callback)

    print("\n===== ROI 区域设置 =====")
    print("鼠标拖拽框选：先框 A区，再框 B区")
    print("按 's' 保存, 'r' 重来, 'q' 退出\n")

    while True:
        ok, frame = camera.read()
        if not ok:
            continue

        display = frame.copy()

        # 画已完成的区域
        colors = [(0, 255, 0), (255, 255, 0)]
        labels = ["Zone A", "Zone B"]
        for i, (x, y, w, h) in enumerate(zones):
            cv2.rectangle(display, (x, y), (x + w, y + h), colors[i], 2)
            cv2.putText(display, labels[i], (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        # 画正在拖拽的矩形
        if drawing and current_rect is not None:
            x, y, w, h = current_rect
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 提示文字
        hint = f"Zones: {len(zones)}/2"
        if len(zones) == 0:
            hint += " | Drag to select Zone A"
        elif len(zones) == 1:
            hint += " | Drag to select Zone B"
        else:
            hint += " | Press S to save"

        cv2.putText(display, hint, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(display, hint, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("ROI Setup", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            camera.release()
            cv2.destroyWindow("ROI Setup")
            return
        elif key == ord('r'):
            zones.clear()
            current_rect = None
            print("已重置")
        elif key == ord('s') and len(zones) == 2:
            break

    camera.release()
    cv2.destroyWindow("ROI Setup")

    ROI_ZONE_A = zones[0]
    ROI_ZONE_B = zones[1]
    print(f"Zone A (pixels): {ROI_ZONE_A}")
    print(f"Zone B (pixels): {ROI_ZONE_B}")
    print("请手动将以上数值写入代码顶部的 ROI_ZONE_A / ROI_ZONE_B，下次运行自动生效。")


# ============================================================
# 串口通信（坐标帧 → STM32）
# ============================================================

_serial_dev = None

def serial_init():
    """打开串口，失败返回 None"""
    global _serial_dev
    if not HAS_SERIAL or not SERIAL_ENABLED:
        return None
    try:
        _serial_dev = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2.0)
        print(f"[SERIAL] 已连接 {SERIAL_PORT} @ {SERIAL_BAUD}")
        return _serial_dev
    except Exception as e:
        print(f"[SERIAL] 连接失败: {e}")
        _serial_dev = None
        return None

def serial_send_coord(x_mm, y_mm, rot_deg):
    """
    发送二进制坐标帧: 0xA5 0x5A X Y Rot 0x00 0xBB (10字节)
    x_mm, y_mm: 世界坐标 (mm, int)
    rot_deg:    旋转角 (°)
    返回: STM32 应答字符串, 失败返回 None
    """
    if _serial_dev is None or not _serial_dev.is_open:
        return None
    rot_01deg = int(rot_deg * 10)
    frame = struct.pack('<BBhhHBB',
        0xA5, 0x5A, x_mm, y_mm, rot_01deg,
        0x00, 0xBB)
    _serial_dev.write(frame)
    _serial_dev.flush()
    try:
        resp = _serial_dev.readline().decode(errors="ignore").strip()
        return resp
    except Exception:
        return None

def serial_close():
    global _serial_dev
    if _serial_dev and _serial_dev.is_open:
        _serial_dev.close()
        _serial_dev = None


# ============================================================
# 物体识别（保留原有逻辑，增加坐标输出）
# ============================================================

def recognize(frame, homography=None):
    """
    识别帧中的物体。
    参数：
        frame: BGR 图像
        homography: 透视变换矩阵（可选，传 None 则不计算世界坐标）
    返回：
        list[dict]: 每个物体的属性
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    results = []

    # 生成 ROI 蒙版（限制识别范围）
    roi_mask = build_roi_mask(frame.shape)

    for color, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # 应用 ROI 限制：只保留 A/B 区域内的识别结果
        mask = cv2.bitwise_and(mask, roi_mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # 形状分类
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            rect = cv2.minAreaRect(contour)
            (cx_px, cy_px), (w_px, h_px), angle = rect
            if min(w_px, h_px) == 0:
                continue

            ratio = max(w_px, h_px) / min(w_px, h_px)

            if circularity > 0.80:
                shape = "Sphere"
            elif ratio > 1.25:
                shape = "Cuboid"
            else:
                shape = "Cube"

            # 像素坐标
            x, y, w, h = cv2.boundingRect(contour)
            center_px = np.array([cx_px, cy_px], dtype=np.float32)

            # 物体主轴角度（长边方向，用于夹爪旋转对齐）
            # minAreaRect 的 angle 是 w_px 边的偏角。取更长边作为主轴。
            if w_px >= h_px:
                principal_angle = angle          # w_px 是长边
            else:
                principal_angle = angle + 90     # h_px 是长边

            # 世界坐标 + 世界空间旋转角（如果有透视变换矩阵）
            world_xy = None
            world_rotation = None
            region = "unknown"
            if homography is not None:
                # 中心 → 世界坐标
                world_xy = pixel_to_world(center_px, homography).flatten()

                # 主轴方向 → 世界空间角度
                rad = np.radians(principal_angle)
                half_len = max(w_px, h_px) / 2.0
                tip_px = center_px + np.array([np.cos(rad) * half_len,
                                               np.sin(rad) * half_len])
                world_tip = pixel_to_world(tip_px, homography).flatten()
                world_dx = world_tip[0] - world_xy[0]
                world_dy = world_tip[1] - world_xy[1]
                world_rotation = float(np.degrees(np.arctan2(world_dy, world_dx)))
                # 归一化到 [0, 180)
                if world_rotation < 0:
                    world_rotation += 180.0
                world_rotation = round(world_rotation, 1)

                region = classify_region(world_xy)

            results.append({
                "color": color,
                "shape": shape,
                "px": (int(cx_px), int(cy_px)),
                "bbox": (x, y, w, h),
                "contour": contour,
                "area": area,
                "circularity": round(circularity, 3),
                "ratio": round(ratio, 3),
                "world_xy": world_xy.tolist() if world_xy is not None else None,
                "world_rotation": world_rotation,
                "region": region,
            })

    return results


# ============================================================
# 可视化
# ============================================================

def draw_results(frame, results, homography=None):
    """在图像上绘制识别结果"""
    for obj in results:
        color_name = obj["color"]
        shape = obj["shape"]
        label = f"{color_name} {shape}"

        contour = obj["contour"]
        x, y, w, h = obj["bbox"]

        cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 主标签：颜色 + 形状
        cv2.putText(frame, label, (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # 世界坐标 + 旋转角 + 区域（如果有）
        if obj["world_xy"] is not None:
            wx, wy = obj["world_xy"]
            rot = obj.get("world_rotation")
            if rot is not None:
                coord_str = f"({wx:.0f},{wy:.0f}) {rot:.0f}deg [{obj['region']}]"
            else:
                coord_str = f"({wx:.0f},{wy:.0f}) [{obj['region']}]"
            cv2.putText(frame, coord_str, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

            # 画旋转方向箭头
            if rot is not None and obj["shape"] != "Sphere":
                px_cx, px_cy = obj["px"]
                rad = np.radians(rot)
                arrow_len = min(w, h) // 2 + 10
                tip_x = int(px_cx + np.cos(rad) * arrow_len)
                tip_y = int(px_cy + np.sin(rad) * arrow_len)
                cv2.arrowedLine(frame, (px_cx, px_cy), (tip_x, tip_y),
                                (0, 165, 255), 2, tipLength=0.3)

    # 绘制 ROI 区域框
    draw_roi_zones(frame)

    # 状态栏
    status = f"Objects: {len(results)}"
    if homography is not None:
        status += " | World coords: ON"
    else:
        status += " | No calibration"
    status += " | C=calibrate  Q=quit"

    cv2.putText(frame, status, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(frame, status, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


# ============================================================
# 主循环
# ============================================================

def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        print(f"无法打开摄像头 (index {CAMERA_INDEX})")
        return

    # 加载标定
    H = load_homography()
    if H is not None:
        print(f"已加载透视变换矩阵: {HOMOGRAPHY_PATH}")
    else:
        print("未检测到透视变换矩阵，按 'c' 进行标定")

    K, D = load_camera_params()
    if K is not None:
        print(f"已加载相机内参: {CAMERA_MATRIX_PATH}")

    # 初始化串口
    ser = serial_init()

    print("\n操作说明:")
    print("  c  = 标定透视变换（点击 4 个场地角点）")
    print("  i  = 设置 ROI 区域（拖拽框选 A区/B区）")
    print("  s  = 发送第一个目标的坐标帧到 STM32")
    print("  r  = 清空标定数据")
    print("  q  = 退出\n")

    last_send = ""
    _window_ok = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue

            if K is not None and D is not None:
                frame = cv2.undistort(frame, K, D)

            results = recognize(frame, homography=H)
            display = draw_results(frame, results, homography=H)

            # 状态栏补上串口信息
            ser_status = "ON" if (_serial_dev and _serial_dev.is_open) else "OFF"
            status_line = f"Serial: {ser_status} | S=send"
            if last_send:
                status_line += f" | Last: {last_send}"
            cv2.putText(display, status_line, (15, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(display, status_line, (15, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Object Recognition", display)

            # 点叉号关闭: 窗口必须先活过, 之后消失才算关闭
            if cv2.getWindowProperty("Object Recognition", cv2.WND_PROP_VISIBLE) >= 1:
                _window_ok = True
            elif _window_ok:
                break

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                camera.release()
                H = calibrate_perspective_interactive()
                camera = cv2.VideoCapture(CAMERA_INDEX)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if not camera.isOpened():
                    print("重新打开摄像头失败")
                    break
            elif key == ord('r'):
                H = None
                if os.path.exists(HOMOGRAPHY_PATH):
                    os.remove(HOMOGRAPHY_PATH)
                print("已清空标定数据")
            elif key == ord('i'):
                camera.release()
                setup_roi_interactive()
                camera = cv2.VideoCapture(CAMERA_INDEX)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if not camera.isOpened():
                    print("重新打开摄像头失败")
                    break
            elif key == ord('s'):
                # 发送第一个有效目标的坐标
                valid = [o for o in results if o["world_xy"] is not None]
                if valid:
                    obj = valid[0]
                    x = int(round(obj["world_xy"][0]))
                    y = int(round(obj["world_xy"][1]))
                    rot = obj.get("world_rotation", 0) or 0
                    resp = serial_send_coord(x, y, rot)
                    if resp:
                        last_send = f"X={x} Y={y} R={int(rot)} -> {resp}"
                        print(f"[SEND] {last_send}")
                    else:
                        last_send = "FAIL (串口未连接)"
                        print(f"[SEND] {last_send}")
                else:
                    last_send = "FAIL (无有效目标)"
                    print(f"[SEND] {last_send}")

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        serial_close()


if __name__ == "__main__":
    main()
