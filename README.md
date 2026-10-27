# IBVS_PAD_TOUCH

宇树 H2 空调面板视觉点按系统。当前主流程使用：

- 胸口 Intel RealSense D435（序列号 `347622074029`）做 Eye-to-Hand 粗定位；
- 右腕 Intel RealSense D435I（序列号 `233722073797`）做 Eye-in-Hand 精对准、进近和接触确认；
- 右臂与 Dex1 完成实际点按；
- 左手 BrainCo Revo2 提供独立的手指姿态控制，目前不参与面板点按。

系统已接入 H2 实机运动，不再是只显示检测结果的监视页面。运行前必须确认 H2 处于稳定站立状态，人员位于手臂工作空间之外。

## 工作流程

1. 胸口相机通过 YOLO 分割、姿态关键点、深度和 Eye-to-Hand 标定获取按钮的躯干坐标。
2. 胸口 PBVS 将右腕移动到目标按钮附近，并保留进近净空。
3. 控制切换到右腕相机；Eye-in-Hand 视觉把按钮与工具尖端对齐。
4. 右腕相机监测最后进近和接触，避免盲目向墙面推进。
5. 组合模式按顺序点按多个按钮；可选读取屏幕图像判断按键是否生效并重试。

H2 躯干坐标约定为 `+X` 向前、`+Y` 向左、`+Z` 向上。RealSense 光学坐标为 `X` 向右、`Y` 向下、`Z` 向前，代码通过标定矩阵完成转换。

## 视觉模型与标定

YOLO 权重默认位于：

```text
outputs/runs/yolo12_seg_20261014_1045/weights/best.pt
outputs/runs/yolo26s_pose_20261014_1045/weights/best.pt
```

胸口和右腕各自加载一套 YOLO-seg 与 YOLO-pose 实例，避免两个相机线程并发使用同一个 Ultralytics 模型。启动日志应分别出现：

```text
[ibvs-yolo] chest loaded ...
[ibvs-yolo] wrist loaded ...
```

胸口相机使用 Eye-to-Hand 标定；右腕相机使用 Eye-in-Hand 标定。不要用现场偏置替代错误的标定文件，也不要放宽最终工具尖端与目标按钮的安全门槛。

## 部署到 H2

推荐从 WSL 同步：

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
export IBVS_ROBOT_HOST=unitree@192.168.1.3
export IBVS_ROBOT_DIR=IBVS_PAD_TOUCH
./scripts/sync_to_robot.sh
```

`IBVS_ROBOT_DIR` 是相对 H2 用户家目录的路径。不要把本地展开后的 `~/IBVS_PAD_TOUCH` 当成远端路径。

同步不会自动重启正在运行的网页。行为或页面代码更新后，需要在 H2 上结束旧进程并重新启动。

## 启动实机网页

在 H2 本体上：

```bash
cd ~/IBVS_PAD_TOUCH
chmod +x scripts/*.sh
./scripts/start_web_remote.sh 8080 \
  --direct-pbvs-poke \
  --direct-pbvs-max-m 1.20 \
  --direct-pbvs-max-joint-deg 120 \
  --eye-to-hand-x-bias-m 0.05 \
  --eye-to-hand-y-bias-m -0.01 \
  --eye-to-hand-z-bias-m 0.025
```

浏览器打开：

```text
http://192.168.1.3:8080/
```

启动脚本会：

- 在同一个相机 sidecar 进程中打开胸口和右腕 RealSense，绕开同一 USB Hub 下两个独立 librealsense 进程互相抢占的问题；
- 启动关节状态 sidecar；
- 启动 Flask 网页和视觉推理；
- 保留 `arm_sdk` 与 `h2_arm_service_node` 两种运动链路，但不会同时占用 `rt/arm_sdk`。

前台运行时按 `Ctrl+C` 会先放下右臂再退出，通常需要约 8 秒。不要在右臂下降过程中重复发送中断。

## 网页操作

推荐顺序：

1. 检查两路画面、深度和 YOLO 状态。
2. 点击 `arm_sdk`，等待页面显示控制链路已拉起。
3. 选择 Button 1、Button 2 或 Button 3。
4. 选择 `Hybrid VS`，然后点击 `Start`；连续点按使用“组合启动”。
5. 需要人工细调时先点 `Pause / 细调`。已停止的任务不会自行恢复，需要重新启动。
6. `释放双臂` 释放控制；`Reset`、`下垂` 和退出程序会改变手臂姿态，操作前清空工作区。

“读取屏幕判断进度”默认关闭。打开后，组合和单次运行会在点按前后抓取屏幕；若屏幕没有变化，该按钮最多重试三次。

页面还提供截图、数据集保存、会话记录、运动轨迹记录/回放和 Eye-to-Hand 现场偏置显示。轨迹回放前必须确认记录来源、环境和起始姿态一致。

## 左手 BrainCo Revo2

网页提供三个只作用于左手的按钮：

- `左手食指`：食指伸直，其余手指收紧，命令为 `(1, 1, 0, 1, 1, 1)`；
- `左手张开`：六通道均为 `0`；
- `左手停止`：停止持续发送左手命令。

话题为：

```text
rt/brainco/left/cmd
rt/brainco/left/state
```

需要先运行 Unitree `brainco_hand_service`。左手控制不会发布右手 BrainCo 话题；H2 右手仍是 Dex1。独立保持食指姿态也可运行：

```bash
python3 scripts/revo2_left_index.py
```

如果没有收到 `rt/brainco/left/state`，程序不会发送动作，并提示检查手部服务、`eth0` 和 RS485 连接。

## 相机与深度状态

正常日志每约 5 秒出现：

```text
[ibvs-cam] chest live frames=... depth_miss=0
[ibvs-cam] d435i live frames=... depth_miss=0
```

仅出现首帧并不代表流仍在更新，应确认 `frames` 持续增加。深度文件超过 2 秒未更新时，网页会显示 depth unavailable。

## 常见问题

### 页面一直显示 loading

确认浏览器持续请求 `/state`，并检查浏览器控制台是否有 JavaScript 语法错误。MJPEG 请求是长连接，服务器通常要等客户端断开后才打印该请求日志。

### arm_sdk 一直等待

确认 H2 正在发布 `rt/lowstate`，且没有遗留的 `h2_arm_service_node` 占用运动话题。网页的 `arm_sdk` 按钮会在后台建立连接，HTTP 请求本身不应长时间阻塞。

### 右腕没有检测结果

先确认右腕 `live frames` 持续增加，再检查日志是否有独立的 `wrist loaded`。画面正常但显示 `SEG unavailable` 或 `POSE unavailable` 时，查看同名 `[ibvs-yolo] wrist ...` 异常。

### CPU 占用看起来超过 100%

Linux `top` 按单核计数，`660%` 表示约使用 6.6 个 CPU 核。双路 YOLO-seg 与 YOLO-pose 默认在 CPU 上推理时占用较高；应同时观察整机 idle、负载和推理延迟。

## 本地开发

Windows：

```powershell
cd E:\MscapeTech\IBVS_PAD_TOUCH
pip install -r requirements.txt
python scripts\run_ibvs.py --smoke
python scripts\annotate_photos.py
```

WSL：

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh
./scripts/run_ibvs.sh --source synthetic --target button_1
```

WSL 的 RealSense USB 转发可能不稳定，本地开发优先使用合成或静态图。实拍标注输出位于 `outputs/annotate/`，会话、轨迹和数据集分别写入 `outputs/recordings/`、`outputs/trajectories/` 和 `outputs/dataset/`。

## 目录

```text
ibvs_pad_touch/camera/       图像、深度与 RealSense 工具
ibvs_pad_touch/perception/   面板、按钮、工具尖端和 YOLO 推理
ibvs_pad_touch/web/          网页、视觉融合、运动控制和 Revo2 控制
scripts/                     启动、同步、sidecar、标注与诊断脚本
configs/                     相机、面板、IBVS 和机器人配置
outputs/                     权重、标定、记录、轨迹与数据集
tests/                       单元和接口测试
```
