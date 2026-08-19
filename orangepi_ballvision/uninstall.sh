#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "请运行：sudo bash uninstall.sh"
    exit 1
fi

systemctl disable --now ballvision.service 2>/dev/null || true
rm -f /etc/systemd/system/ballvision.service
systemctl daemon-reload
nmcli connection delete BallVision-Hotspot 2>/dev/null || true
echo "BallVision 服务和专用热点配置已移除。/opt/ballvision 保留，可手动删除。"
