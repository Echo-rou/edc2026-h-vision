# 香橙派 MJPEG 无线图传使用说明

## 方案

香橙派直接作为图传服务器：

```text
USB 摄像头 -> Orange Pi 5 Pro -> RKNN 钢球检测 -> MJPEG 网页图传
```

手机、平板或电脑连接到香橙派所在局域网后，用浏览器打开：

```text
http://香橙派IP:8080/
```

如果香橙派自己开热点并使用 `192.168.2.1` 网段，则访问：

```text
http://192.168.2.1:8080/
```

## 开热点

先确认无线网卡名称：

```bash
ip -br link
```

如果无线网卡是 `wlan0`，可以用手册里的 `create_ap`：

```bash
sudo create_ap -m nat wlan0 enP4p65s0 BallVision 12345678 -g 192.168.2.1
```

如果不需要通过网口共享互联网，只需要局域网图传，也可以按板端实际 `create_ap -h` 支持的参数创建普通热点。

## 运行图传检测程序

在香橙派上进入程序目录：

```bash
cd /home/orangepi
python3 camera_detect.py
```

当前程序已内置 MJPEG 服务：

```python
MJPEG_ENABLED = True
MJPEG_HOST = "0.0.0.0"
MJPEG_PORT = 8080
MJPEG_QUALITY = 75
MJPEG_MAX_FPS = 30
```

## 摄像头检查

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

如果摄像头不是 `/dev/video0`，修改 `camera_detect.py`：

```python
CAMERA_ID = 1
```

## 常见排查

查看香橙派 IP：

```bash
ip -br addr
```

确认 8080 端口监听：

```bash
ss -lntp | grep 8080
```

关闭 Wi-Fi 省电：

```bash
sudo iw dev wlan0 set power_save off
```

如果浏览器打不开，先在香橙派本机测试：

```bash
curl http://127.0.0.1:8080/
```

