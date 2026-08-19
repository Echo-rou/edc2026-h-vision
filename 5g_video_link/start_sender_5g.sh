#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DISPLAY_OPTION=""
if [ -n "${DISPLAY:-}" ]; then
    DISPLAY_OPTION="--display"
fi

python3 vision_5g_udp.py $DISPLAY_OPTION "$@"
