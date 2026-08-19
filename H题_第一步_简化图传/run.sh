#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

cleanup() {
    sudo nmcli connection down BallVision-Hotspot >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

sudo bash "$APP_DIR/start_hotspot.sh"

echo
echo "保持这个终端窗口开启。"
echo "电脑连接热点 BallVision，密码 12345678。"
echo "然后用浏览器打开：http://192.168.5.1:8080/"
echo "按 Ctrl+C 结束程序并关闭热点。"
echo

python3 "$APP_DIR/ballvision_web.py" "$@"
