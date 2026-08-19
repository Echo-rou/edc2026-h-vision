# Orange Pi 5 Pro 一键钢球视觉网页

这个目录可整体复制到 Orange Pi 5 Pro。程序会读取 USB UVC 摄像头，使用
RK3588S NPU 和随包提供的 INT8 RKNN 模型检测钢球，并把画面发布到香橙派自己
创建的 Wi-Fi 热点中。手机、平板或电脑不需要安装接收程序，只用浏览器即可。

## 最快使用

将整个 `orangepi_ballvision` 目录复制到香橙派，例如：

```bash
scp -r orangepi_ballvision orangepi@香橙派当前IP:/home/orangepi/
```

登录香橙派后安装：

```bash
cd /home/orangepi/orangepi_ballvision
sudo bash install.sh
sudo reboot
```

重启后：

1. 手机或电脑连接热点 `BallVision5G`
2. 默认密码 `12345678`
3. 浏览器打开 `http://192.168.5.1:8080/`

网页可实时观看检测画面、保存截图、在接收设备上开始/停止录像以及全屏显示。
录像由浏览器保存到手机或电脑的下载目录，不占用香橙派存储。

## 改热点名称或密码

安装前执行：

```bash
sudo env BALLVISION_SSID=我的热点 BALLVISION_PASSWORD=至少8位密码 bash install.sh
```

默认优先创建 5 GHz 信道 36；如果无线网卡或驱动不支持，启动脚本会自动回退到
2.4 GHz 信道 6。

## 不安装服务，直接运行

```bash
cd /home/orangepi/orangepi_ballvision
bash start_now.sh
```

如果只想测试摄像头和网页、不启用 NPU：

```bash
bash start_now.sh --no-detect
```

## 检查和排错

```bash
sudo systemctl status ballvision --no-pager
sudo journalctl -u ballvision -n 100 --no-pager
v4l2-ctl --list-devices
curl http://127.0.0.1:8080/health
```

摄像头不是 `/dev/video0` 时，在服务文件的 `ExecStart` 末尾增加
`--camera 1`，然后运行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart ballvision
```

手动停止或启动：

```bash
sudo systemctl stop ballvision
sudo systemctl start ballvision
```

卸载服务与热点配置：

```bash
sudo bash uninstall.sh
```

卸载脚本会保留 `/opt/ballvision`，避免误删模型和程序。

## 与用户手册的对应

- 手册第 3.6.4 节说明了 Wi-Fi 热点、`-g` 固定网关以及 5 GHz 频段。
- 手册第 3.12.4 节说明 USB UVC 摄像头及 `/dev/videoN` 设备节点。
- 本包使用 NetworkManager 的共享/AP 模式实现同等热点功能，固定网关为
  `192.168.5.1`，便于每次用同一个浏览器地址访问。
