#!/usr/bin/env bash
set -euo pipefail

CONNECTION_NAME="${BALLVISION_CONNECTION:-BallVision-Hotspot}"
SSID="${BALLVISION_SSID:-BallVision5G}"
PASSWORD="${BALLVISION_PASSWORD:-12345678}"
AP_ADDRESS="${BALLVISION_AP_CIDR:-192.168.5.1/24}"
WIFI_DEVICE="${BALLVISION_WIFI_DEVICE:-}"

if [ "$(id -u)" -ne 0 ]; then
    echo "请使用 sudo 运行热点脚本。"
    exit 1
fi

if ! command -v nmcli >/dev/null 2>&1; then
    echo "缺少 nmcli，请安装 NetworkManager。"
    exit 1
fi

if [ -z "$WIFI_DEVICE" ]; then
    WIFI_DEVICE="$(
        nmcli -t -f DEVICE,TYPE device status |
        awk -F: '$2 == "wifi" {print $1; exit}'
    )"
fi
if [ -z "$WIFI_DEVICE" ]; then
    echo "未发现 Wi-Fi 网卡。请检查无线模块和驱动。"
    exit 1
fi

if ! nmcli -t -f NAME connection show | grep -Fxq "$CONNECTION_NAME"; then
    nmcli connection add \
        type wifi ifname "$WIFI_DEVICE" con-name "$CONNECTION_NAME" \
        autoconnect yes ssid "$SSID"
    nmcli connection modify "$CONNECTION_NAME" \
        802-11-wireless.mode ap \
        802-11-wireless.band a \
        802-11-wireless.channel 36 \
        wifi-sec.key-mgmt wpa-psk \
        wifi-sec.psk "$PASSWORD" \
        ipv4.method shared \
        ipv4.addresses "$AP_ADDRESS" \
        ipv6.method disabled
fi

nmcli radio wifi on
if ! nmcli connection up "$CONNECTION_NAME" ifname "$WIFI_DEVICE"; then
    echo "5 GHz 热点启动失败，自动回退到 2.4 GHz 信道 6。"
    nmcli connection modify "$CONNECTION_NAME" \
        802-11-wireless.band bg \
        802-11-wireless.channel 6
    nmcli connection up "$CONNECTION_NAME" ifname "$WIFI_DEVICE"
fi

iw dev "$WIFI_DEVICE" set power_save off >/dev/null 2>&1 || true
echo "热点已启动：$SSID"
echo "浏览器地址：http://${AP_ADDRESS%/*}:8080/"
