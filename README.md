# IBVS_PAD_TOUCH

宇树 H2 空调面板视觉点按。网页和两台 RealSense 跑在 Orin 上，手臂指令经 `eth2` 的组播 DDS 发到 H2。

- 胸口 Intel RealSense D435（`347622074029`）做 Eye-to-Hand 粗定位，分割和姿态在 GPU 上以 FP16 运行。
- 右腕 Intel RealSense D435I（`233722073797`）只做姿态，在 Orin 的 CPU 上计算，用于 Eye-in-Hand 进近。
- 右手 BrainCo Revo2 用食指指尖点按，经桥接 `192.168.1.3:8092`。左手 Revo2 仍走 DDS。
- 当前运行代码在 `ibvs_pad_touch/web/`、`perception/`、`servo/`。同名的 `ibvs_pad_touch/ibvs_pad_touch/` 是旧副本，网页不加载它。

运行前确认 H2 已上电并进入运控，人员在手臂工作空间之外。控制链路说明见 `控制链路.md`。

躯干坐标：`+X` 向前、`+Y` 向左、`+Z` 向上。指尖的 X 不得超过目标按钮的 X，否则会戳进面板。到位门槛是 2 mm，指尖匹配门槛是 20 mm，这两处不要放宽。

## 网络

| 位置 | 地址 | 用途 |
| --- | --- | --- |
| Orin `wlan0` | `192.168.1.72/24` | 浏览器和 SSH |
| Orin `eth2` | `192.168.123.166/24` | 到 H2 的 DDS。NetworkManager 连接名 `Wired connection 3`，静态地址，`ipv4.never-default yes` |
| H2 | `192.168.123.162` | 运控链路对端 |

`eth2` 必须带着 `192.168.123.166/24` 处于 UP。地址被 DHCP 清掉时，CycloneDDS 会报 `does not match an available interface`。正常启动不要设置 `IBVS_DDS_ADDRESS` 和 `IBVS_DDS_PEERS`，`rt/lowstate` 走组播。

## 启动

在 Orin 上：

```bash
unset IBVS_DDS_ADDRESS IBVS_DDS_PEERS
cd ~/IBVS_PAD_TOUCH
./scripts/start_web_remote.sh 8080 \
  --iface eth2 \
  --right-ee revo2 \
  --revo2-bridge 192.168.1.3:8092 \
  --direct-pbvs-poke \
  --direct-pbvs-max-m 1.20 \
  --direct-pbvs-max-joint-deg 120 \
  --eye-to-hand-x-bias-m 0.04 \
  --eye-to-hand-y-bias-m -0.065 \
  --eye-to-hand-z-bias-m 0.035
```

浏览器打开 `http://192.168.1.72:8080/`。客户记录页在 `8082`。

启动后应同时看到：

```text
[ibvs-cam] chest live frames=... depth_miss=0
[ibvs-cam] d435i live frames=... depth_miss=0
[ibvs-yolo] chest loaded ... fp16
[ibvs-yolo] wrist loaded cpu fp32
```

日志里不要出现 `[dds] unicast`。`arm_sdk` 拉起后应出现 `rt/arm_sdk iface=eth2` 和 `arm_sdk 已接`。

前台 `Ctrl+C` 会先放下右臂，大约 8 秒，然后再停网页、关节和相机进程。不要在右臂下降时重复按中断。

从这台开发机同步到 Orin 用 WSL：

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
export IBVS_ROBOT_HOST=user@192.168.1.72
export IBVS_ROBOT_DIR=IBVS_PAD_TOUCH
./scripts/sync_to_robot.sh
```

同步不会重启已经打开的网页。

## 网页操作

1. 确认两路画面的 `live frames` 在增加。
2. 点 `arm_sdk`。这一步只拉起本页的 `arm_sdk`，不启动运控服务。
3. 选 Button 1 / 2 / 3。Direct PBVS 用胸口坐标一次到位；Hybrid VS 先胸口粗定位，再右腕进近。
4. 连续点按用组合启动。失败会在页面上标明 `【重试】` 和第几次。
5. `释放双臂` 只把当前 `arm_sdk` 权重打到 0 并关闭。它不拉起运控服务，不把进行中的回退再发一遍，也不结束相机进程。
6. `Reset` 和 `下垂` 会改变手臂姿态。

`arm_sdk` 和 `h2_arm_service_node` 不能同时发布 `rt/arm_sdk`。

## 视觉权重

```text
outputs/runs/yolo12_seg_20261014_1045/weights/best.pt
outputs/runs/yolo26s_pose_20261103_1130/weights/best.pt
```

姿态点顺序是 `gripper_tip`、`button_1`、`button_2`、`button_3`、`FingerTip`。Revo2 点按用 `FingerTip`。右腕画面不画分割。

## 右手 Revo2

右手串口是 `/dev/serial/by-id/usb-Unitree_Unitree_CDC_COM_26674C9F1974-if00`，从站 127，波特率 460800。官方 `brainco_hand_service` 不要占用这只右手。左手仍订阅：

```text
rt/brainco/left/cmd
rt/brainco/left/state
```

## 本地开发

WSL：

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh
./scripts/run_ibvs.sh --source synthetic --target button_1
```

不要在 WSL 里启动实机网页，RealSense 和 `eth2` 都在 Orin 上。测试单独运行对应文件，不要一次跑完整套 pytest。

## 目录

```text
ibvs_pad_touch/perception/   面板、按钮和 YOLO
ibvs_pad_touch/servo/        图像进近
ibvs_pad_touch/web/          网页、PBVS 和 Revo2
scripts/                     启动、相机 sidecar、同步
configs/                     相机、面板和 IBVS 配置
outputs/                     权重、记录和轨迹
tests/                       单元测试
控制链路.md                   从网页按钮到关节指令的路径
```
