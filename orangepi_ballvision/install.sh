#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="/opt/ballvision"
SERVICE_FILE="/etc/systemd/system/ballvision.service"
RUN_USER="${SUDO_USER:-$(id -un)}"
SSID="${BALLVISION_SSID:-BallVision5G}"
PASSWORD="${BALLVISION_PASSWORD:-12345678}"
WIFI_DEVICE="${BALLVISION_WIFI_DEVICE:-}"

if [ "$(id -u)" -ne 0 ]; then
    echo "请运行：sudo bash install.sh"
    exit 1
fi
if [ "$RUN_USER" = "root" ]; then
    echo "请从普通用户会话运行 sudo bash install.sh，不能直接使用 root 会话安装。"
    exit 1
fi
if [ ${#PASSWORD} -lt 8 ]; then
    echo "热点密码至少需要 8 个字符。"
    exit 1
fi

echo "[1/6] 检查系统依赖"
export DEBIAN_FRONTEND=noninteractive
if ! command -v nmcli >/dev/null 2>&1 ||
   ! command -v v4l2-ctl >/dev/null 2>&1 ||
   ! python3 -c "import cv2, numpy" >/dev/null 2>&1; then
    apt-get update
    apt-get install -y \
        network-manager iw v4l-utils \
        python3-opencv python3-numpy \
        gstreamer1.0-tools gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good
fi

if ! python3 -c "from rknnlite.api import RKNNLite" >/dev/null 2>&1; then
    echo
    echo "未检测到 RKNNLite Python 运行库。"
    echo "请先按 Rockchip/Orange Pi 镜像对应版本安装 rknn_toolkit_lite2，随后重新运行本脚本。"
    exit 1
fi

echo "[2/6] 安装程序和模型"
install -d -m 0755 "$INSTALL_DIR"
install -m 0755 "$SOURCE_DIR/ballvision_web.py" "$INSTALL_DIR/ballvision_web.py"
install -m 0755 "$SOURCE_DIR/hotspot.sh" "$INSTALL_DIR/hotspot.sh"
install -m 0644 "$SOURCE_DIR/ball_best_int8_split.rknn" "$INSTALL_DIR/ball_best_int8_split.rknn"

echo "[3/6] 配置用户权限"
usermod -aG video,dialout "$RUN_USER"

echo "[4/6] 写入热点配置"
if [ -n "$WIFI_DEVICE" ]; then
    WIFI_ENV="Environment=BALLVISION_WIFI_DEVICE=$WIFI_DEVICE"
else
    WIFI_ENV=""
fi

echo "[5/6] 创建开机自启服务"
cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=BallVision RKNN camera and Wi-Fi hotspot
After=NetworkManager.service dev-video0.device
Wants=NetworkManager.service

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_USER
SupplementaryGroups=video dialout
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONUNBUFFERED=1
Environment=BALLVISION_SSID=$SSID
Environment=BALLVISION_PASSWORD=$PASSWORD
$WIFI_ENV
ExecStartPre=+$INSTALL_DIR/hotspot.sh
ExecStart=/usr/bin/python3 $INSTALL_DIR/ballvision_web.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo "[6/6] 启用服务"
systemctl daemon-reload
systemctl enable ballvision.service

echo
echo "安装完成。为避免当前 SSH/Wi-Fi 连接立即中断，本脚本没有马上切换热点。"
echo "请执行 sudo reboot；重启后连接热点："
echo "  名称：$SSID"
echo "  密码：$PASSWORD"
echo "  网页：http://192.168.5.1:8080/"
echo
echo "不重启而立即测试：sudo systemctl start ballvision"
