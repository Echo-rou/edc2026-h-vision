#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

if [ "$(id -u)" -eq 0 ]; then
    echo "检测程序不应以 root 身份运行。请直接执行：bash start_now.sh"
    exit 1
fi

sudo env \
    BALLVISION_SSID="${BALLVISION_SSID:-BallVision5G}" \
    BALLVISION_PASSWORD="${BALLVISION_PASSWORD:-12345678}" \
    BALLVISION_WIFI_DEVICE="${BALLVISION_WIFI_DEVICE:-}" \
    bash "$APP_DIR/hotspot.sh"

echo
echo "手机或电脑连接 BallVision5G 后，打开 http://192.168.5.1:8080/"
echo "按 Ctrl+C 退出。"
exec python3 "$APP_DIR/ballvision_web.py" "$@"
