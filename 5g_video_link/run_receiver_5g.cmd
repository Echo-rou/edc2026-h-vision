@echo off
cd /d "%~dp0"
python receive_5g_udp.py --port 5600 --record-fps 30 --sender 192.168.12.1
pause
