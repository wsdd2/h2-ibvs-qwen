# G1 网页手眼标定采集流程

本文记录 `capture_handeye.py` 的网页调试流、实时 FK、`rt/arm_sdk` 右臂 waypoint 运动，以及保存图片和机器人位姿的操作。

## 1. 同步代码到机器人

在本地 WSL/同步终端执行：

```bash
rsync -av /mnt/e/MscapeTech/Hand_Eye_Calib/ \
  unitree@192.168.1.86:~/MscapeTech/Hand_Eye_Calib/
```

同步后在机器人上确认新版参数存在：

```bash
cd ~/MscapeTech/Hand_Eye_Calib
python capture_handeye.py -h | grep arm
```

应能看到 `--enable-arm-waypoints`、`--arm-network-interface` 等参数。

## 2. 启动网页采集

在机器人主机上运行：

```bash
cd ~/MscapeTech/Hand_Eye_Calib

python capture_handeye.py \
  --mode eye-in-hand \
  --stream-debug \
  --stream-fk \
  --headless \
  --enable-arm-waypoints \
  --arm-network-interface eth0 \
  --fk-network-interface eth0 \
  --hand-frame right_wrist_yaw_link \
  --camera-mount wrist \
  --color-only
```

如果默认 DDS 网卡已验证可用，也可以去掉 `--arm-network-interface eth0` 和 `--fk-network-interface eth0`。

默认网页地址：

```text
http://192.168.1.86:8080/
```

这是机器人主机直接监听 `0.0.0.0:8080`，本地浏览器直接访问机器人 IP。

## 3. 网页按钮说明

右侧 `Arm Waypoints` 区域用于 `rt/arm_sdk` 关节 waypoint 控制：

- `Save Current`：把当前右臂 7 个关节角保存为一个 waypoint。
- `Prev` / `Next`：切换当前选中的 waypoint。
- `Move`：平滑运动到当前选中的 waypoint，并保持。
- `Random Right Arm`：只随机扰动右臂 7 个关节；命令帧会同时给腰部和左臂发送当前值作为 0 delta 保持，避免上半身目标不完整。
- `Hold Current`：保持当前姿态。
- `Arm Default`：释放 `arm_sdk`，并调用 high-level `release_arm(99)`，只请求手臂回默认/释放状态。
- `Release Arm SDK`：释放 `arm_sdk` 权重，把手臂控制权交还内部控制。

标定采集按钮：

- `Save / SPACE`：棋盘格检测成功时，保存当前图像和 FK 位姿。
- `Solve / S`：用已保存样本求解手眼标定。
- `Quit / Q`：退出采集程序。

## 4. 推荐采集顺序

1. 确认机器人站稳、急停可用、手臂周围空间安全。
2. 打开网页，确认画面和 `fk` 状态正常刷新。
3. 在当前安全姿态点 `Save Current`，保存 home waypoint。
4. 用高层预设动作、手动示教或其它安全方式让手臂到另一个姿态，再点 `Save Current`。
5. 重复保存多个不同姿态。
6. 用 `Prev` / `Next` 选择 waypoint，点 `Move` 到目标姿态。
7. 或者点 `Random Right Arm`，自动生成一个小幅随机右臂姿态并保持。
8. 等画面稳定且棋盘格检测成功后，点 `Save / SPACE` 保存图片和当前 FK 位姿。
9. 采集完成后点 `Release Arm SDK`。
10. 样本足够后点 `Solve / S` 求解。

注意：如果只保存了一个当前姿态，点 `Move` 基本不会动，因为目标就是当前位置。至少需要保存两个不同姿态，或者保存一批提前构造好的 waypoint。

随机右臂运动默认肩/肘关节最大扰动为 `0.08 rad`，手腕关节最大扰动为 `0.18 rad`。可以通过启动参数调整：

```bash
--arm-random-shoulder-elbow-max-delta-rad 0.04 \
--arm-random-wrist-max-delta-rad 0.18 \
--arm-limit-margin-rad 0.03
```

随机目标和 waypoint 目标都会按 `--fk-urdf` 中的关节上下限 clamp，并预留 `--arm-limit-margin-rad` 安全边界。网页状态里的 `last_random_right_arm_clamped` 会显示被截断的关节。

## 5. Waypoint 文件

默认 waypoint 文件保存在当前 session 目录下：

```text
<data_root>/<session_name>/<camera_name>/arm_waypoints.json
```

也可以用固定路径：

```bash
--arm-waypoints-json ~/MscapeTech/Hand_Eye_Calib/data/calib_arm_waypoints.json
```

文件格式示例：

```json
[
  {
    "name": "home",
    "joints": {
      "waist_yaw": 0.0,
      "waist_roll": 0.0,
      "waist_pitch": -0.03,
      "right_shoulder_pitch": 0.29,
      "right_shoulder_roll": -0.22,
      "right_shoulder_yaw": 0.04,
      "right_elbow": 0.98,
      "right_wrist_roll": -0.10,
      "right_wrist_pitch": 0.01,
      "right_wrist_yaw": 0.02
    }
  }
]
```

当前网页 waypoint 只会改变右臂 7 个关节：

```text
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

发送 `rt/arm_sdk` 时会同时带上腰部和左臂当前关节角作为 0 delta 保持值。也就是说，目标变化只发生在右臂；腰部和左臂用于补全上半身命令，不主动产生位移。不会向下肢写入目标关节命令。

## 6. 安全注意

- `--enable-arm-waypoints` 会允许网页按钮向 `rt/arm_sdk` 发布手臂控制命令。
- 首次测试建议使用小幅姿态变化，并保持急停可用。
- `Release Arm SDK` 后，内部控制可能把手臂拉回默认姿态，这是正常现象。
- 如果需要稳定停在某个标定姿态，保持在 waypoint 后再点 `Save / SPACE`，不要先 release。
- 默认启动时会先释放一次 `arm_sdk`，避免上一次程序残留的保持命令导致一进入网页就持续发力。
- 默认 `Move` / `Random Right Arm` 到点并短暂稳定后会自动 `Release Arm SDK`，避免机器人一直保持/抖动。
- 如果需要到点后持续保持姿态用于拍照，启动时加 `--no-arm-release-after-move`，然后拍完再手动点 `Release Arm SDK`。
- 网页端不提供下肢/全身控制。需要站立、阻尼或恢复姿态时，优先使用手柄/App 或单独的高层 Unitree 工具，并确认周围空间安全。
