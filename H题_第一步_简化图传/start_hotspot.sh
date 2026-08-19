#!/usr/bin/env bash
set -euo pipefail

CONNECTION="BallVision-Hotspot"
SSID="${BALLVISION_SSID:-BallVision}"
PASSWORD="${BALLVISION_PASSWORD:-12345678}"
ADDRESS="192.168.5.1/24"

if [ "$(id -u)" -ne 0 ]; then
    echo "请使用：sudo bash start_hotspot.sh"
    exit 1
fi

WIFI_DEVICE="$(
    nmcli -t -f DEVICE,TYPE device status |
    awk -F: '$2 == "wifi" {print $1; exit}'
)"
if [ -z "$WIFI_DEVICE" ]; then
    echo "没有找到 Wi-Fi 网卡。"
    exit 1
fi

nmcli connection delete "$CONNECTION" >/dev/null 2>&1 || true
nmcli connection add \
    type wifi ifname "$WIFI_DEVICE" con-name "$CONNECTION" \
    autoconnect no ssid "$SSID"
nmcli connection modify "$CONNECTION" \
    802-11-wireless.mode ap \
    802-11-wireless.band a \
    802-11-wireless.channel 36 \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$PASSWORD" \
    ipv4.method shared \
    ipv4.addresses "$ADDRESS" \
    ipv6.method disabled

nmcli radio wifi on
if ! nmcli connection up "$CONNECTION"; then
    echo "5 GHz 启动失败，改用 2.4 GHz。"
    nmcli connection modify "$CONNECTION" \
        802-11-wireless.band bg \
        802-11-wireless.channel 6
    nmcli connection up "$CONNECTION"
fi

iw dev "$WIFI_DEVICE" set power_save off >/dev/null 2>&1 || true
echo "热点名称：$SSID"
echo "热点密码：$PASSWORD"
echo "网页地址：http://192.168.5.1:8080/"
