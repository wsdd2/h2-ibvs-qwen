# Unitree H2 Hybrid Visual Servoing

宇树 H2 右臂、闭合且未上电的 Dex1 夹爪，戳 STULZ 空调面板上的三颗黄键。胸口 D435 做眼在手外，左手侧 D435I 做眼在手上核对。运动只走 DDS `rt/arm_sdk`，或车上已有的 `h2_arm_service_node`。这个进程里没有 ROS / MoveIt 规划器。

子目录里的 `IBVS_PAD_TOUCH/README.md` 和 `Hand_Eye_Calib/README.md` 是早期说明，和当前实机行为不一致。改代码以本文件和源码常量为准。

## 现在不要做的事

胸口相机支架已经撞断，相机不在 `eye_to_hand_20260924_101819.json` 标定的位置上。在相机装回原位之前：

- 不要点网页 **Start**、**组合启动**、**Right level**。
- 不要把 `FOREARM_LEVEL_DEG` 改回肩 `-90°`、肘 `85°`。那组关节把前臂折回胸口，支架就是这样断的。
- 像素检测可以是对的，旧标定算出的 torso XYZ 不是黄键的真实位置。不要靠加大步长或放宽到位门槛去“够到画面上的点”。

AI 会话里不要自己 SSH 开车、不要 `rsync`、不要重启 `h2_arm_service_node`、不要调用 `/h2_arm/singlearmcartesian`。人自己同步、自己启动。

## 仓库里有什么

```text
IBVS_PAD_TOUCH/          实机网页、检测、伺服、手臂动作
Hand_Eye_Calib/          手眼标定和运行时 IK（h2_ee_ik.py、h2_arm_sdk.py）
```

运行时真正要读的文件：

| 路径 | 作用 |
|---|---|
| `IBVS_PAD_TOUCH/scripts/start_web_remote.sh` | 车上启动入口。拉起网页、关节 sidecar、两路相机 sidecar、`h2_arm_service_node` |
| `IBVS_PAD_TOUCH/scripts/run_web.py` | Flask 参数 |
| `IBVS_PAD_TOUCH/ibvs_pad_touch/web/dashboard.py` | 网页按钮和 HTTP |
| `IBVS_PAD_TOUCH/ibvs_pad_touch/web/arm_motion.py` | Direct PBVS、组合、平举、到位门槛 |
| `IBVS_PAD_TOUCH/ibvs_pad_touch/web/live_ibvs.py` | 检测、标定加载、胸口尖端 |
| `IBVS_PAD_TOUCH/ibvs_pad_touch/web/arm_service.py` | service 后端写 yaml |
| `IBVS_PAD_TOUCH/ibvs_pad_touch/perception/yolo_touch.py` | YOLO pose / seg |
| `IBVS_PAD_TOUCH/ibvs_pad_touch/camera/devices.py` | 相机序列号占用，禁止互抢 |
| `Hand_Eye_Calib/handeye_calib/h2_ee_ik.py` | 本地 IK。末端连杆是 `right_wrist_yaw_link` |

不在本仓库、在机器人上：

- 标定 JSON：`~/MscapeTech/Hand_Eye_Calib/artifacts/eye_to_hand/outputs/` 和同级 `eye_in_hand/outputs/`。当前胸口标定文件名是 `eye_to_hand_20260924_101819.json`。
- 手臂服务源码和已安装二进制：`~/h2arm_control_ws`。本仓库没有它。不要擅自 `colcon build`。
- URDF：`/home/unitree/MscapeTech/urdf/H2.urdf`。
- 训练日志、`last.pt`、`epoch*.pt`。权重要用 `IBVS_PAD_TOUCH/weights/yolo26m_seg.pt` 和 `yolo26s_pose.pt`。

## 快速部署

开发机是 Windows + WSL。机器人是 `unitree@192.168.1.3`，Ubuntu 22.04，网页端口 `8080`。ROS 域从车上 `~/.bashrc` 读，不要写死成 0、42 或 53。`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`，`ROS_LOCALHOST_ONLY=0`。DDS 网卡 `eth0`。

本机只跑测试，不打开 RealSense：

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
python -m pytest tests/test_arm_motion.py tests/test_web_dashboard.py tests/test_camera_devices.py tests/test_live_ibvs.py tests/test_yolo_touch.py -q
```

同步到车上（人自己执行）：

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
export IBVS_ROBOT_HOST=unitree@192.168.1.3
./scripts/sync_to_robot.sh
```

`sync_to_robot.sh` 会把 `IBVS_PAD_TOUCH` 同步到 `~/IBVS_PAD_TOUCH`，并把 `Hand_Eye_Calib/handeye_calib/h2_ee_ik.py`、`h2_arm_sdk.py` 单独同步到 `~/MscapeTech/Hand_Eye_Calib/handeye_calib/`。车上的标定 JSON 不要被这次同步覆盖。

车上启动。下面这组偏置是当前现场值，不要改回代码里的默认 `0.03 / 0.0 / 0.025`：

```bash
cd ~/IBVS_PAD_TOUCH
./scripts/start_web_remote.sh 8080 \
  --direct-pbvs-poke \
  --direct-pbvs-max-m 1.20 \
  --direct-pbvs-max-joint-deg 120 \
  --eye-to-hand-x-bias-m 0.012 \
  --eye-to-hand-y-bias-m 0.02 \
  --eye-to-hand-z-bias-m 0.03 \
  --eye-to-hand-json /home/unitree/MscapeTech/Hand_Eye_Calib/artifacts/eye_to_hand/outputs/eye_to_hand_20260924_101819.json
```

脚本自己加上 `--confirm-arm-motion`。浏览器打开 `http://192.168.1.3:8080/`。Ctrl+C 会先把右臂放下，再停服务。日志在启动它的那个 SSH 窗口里。

SSH 密码不要写进仓库。车上脚本需要时用环境变量 `IBVS_SSH_PASSWORD`。

## 运行时谁在说话

`start_web_remote.sh` 同时起四样东西。网页进程自己不开 RealSense。

| 进程 | 作用 |
|---|---|
| `camera_sidecar.py --name chest --serial 347622074029` | 胸口 D435。先启动，等 2 秒 |
| `camera_sidecar.py --name d435i --serial 233722073797` | 左手侧 D435I。序列号曾经在右腕，现在固定在左手 |
| `joint_sidecar.py eth0` | 读 `rt/lowstate` |
| `run_web.py` | 读 sidecar 文件，发运动 |

指定了序列号却没找到设备时，sidecar 必须打印 `waiting for <serial>` 并空等。不许改去占用另一台相机。胸口相机掉线时，`/depth/chest.png` 是 503，这是对的。

Python 优先 `/home/unitree/anaconda3/envs/unitree/bin/python`。`rs-enumerate-devices -s` 能看到两台、Python 只能看到一台时，先查这个环境里的 `pyrealsense2`，不要改序列号。

运动后端默认 `arm_sdk`（`H2ArmSdkController`，重力补偿，kp 85 / kd 2.2）。`service` 后端走 `/h2_arm/singlearmcartesian`。两个后端不能同时往 `rt/arm_sdk` 上发。服务节点活着时，这个网页在 service 模式下不调用 `enable_arm_sdk`。忙的时候拒绝换后端。

## 坐标和末端

Torso：X 前，Y 左，Z 上。胸口相机光学：X 右，Y 下，Z 前。相机 Z 大致对着 torso X。

`right_ee_link` 是 `right_wrist_yaw_link` 上的固定关节，平移 `0.205 0 0`。

- 服务的 `target_pose` XYZ 已经是指尖。写 yaml 之前不要再减 0.205 m。
- 本地 `H2CameraIK` 的目标连杆是 `right_wrist_yaw_link`，要减 `R * (0.205, 0, 0)`。
- 网页「末端 XYZ」是 `tip_xyz(q, RIGHT_EE_OFFSET_WRIST_M)`，和指尖是同一个点。
- 「目标−末端」是 `target_torso_m - right_ee_xyz_m`。Y 为负表示目标在手的右侧。

眼在手外：`cam2reference`，方法必须是 `eye_to_hand`，参考系 `torso_link`，序列号若存在必须是 `347622074029`。三个 bias 加在 `torso_T_camera` 的平移上。Torso Y 正方向向左。

眼在手上：读 `cam2hand.transform`。`p_hand = T_cam2hand * p_cam`，`T_torso_cam = T_torso_hand(q) * T_cam2hand`，手部坐标系默认 `right_wrist_yaw_link`。它只在平举后停留 1 秒做一次比对，不是连续眼在手上伺服。任何一颗配对按键误差 ≥ 2 cm，整组保持胸口坐标。平举过程中始终没看到面板和按键，就跳过眼在手上。

## 感知

胸口尖端只用 YOLO，不用颜色。

- Pose `weights/yolo26s_pose.pt`，关键点 `gripper_tip, button_1, button_2, button_3`。
- Seg `weights/yolo26m_seg.pt`，类 `panel / gripper / screen`。预测 conf 0.15。夹爪 mask 保留门槛 0.3。
- 夹爪是黑的。白支架、白手臂、白相机不是夹爪。
- 有 pose 尖端、这一帧没有 gripper mask 时，保留 pose 点。
- pose 点落在按键上、mask 在别处时，用朝向面板的 mask 边缘，来源 `face`。
- 不要把 `detect_dex1_tip` 接回实机胸口尖端。不要放宽 Canny。不要改柜子 YOLO 的类别表和路径。

PBVS 开始时锁定三颗键的 torso XYZ，只接受 `source=="pose"`。之后 IBVS 冻结这组坐标。画面跳变超过 40 px 丢掉。叠加层仍画实时检测。

## Direct PBVS 怎么走

一次按钮是少量笛卡尔整段，不是很多小步，也不是顶着 24 秒截止时间一直试。目标是锁定的按键 torso XYZ，没有最终 standoff。Hybrid 才保留 3 cm standoff（`PBVS_STANDOFF_M`）。

X 是朝面板的方向。Y、Z 还没对准时不能把 X 一次走完，下颌会顶在面板下沿。X 也不能冻住，否则关节顶限。

| 常量 | 值 | 含义 |
|---|---|---|
| `DIRECT_PBVS_X_CLEAR_M` | 0.08 | 起始时停在按键前 8 cm |
| `DIRECT_PBVS_X_LEAD_M` | 0.05 | Y/Z 未对准时，每枪 X 最多前进 5 cm |
| `DIRECT_PBVS_X_RELAX_M` | 0.02 | Y/Z 停滞不到 2 mm 时，间隙减 2 cm：8→6→4→2→0 |
| `DIRECT_PBVS_ARRIVE_TOL_M` | 0.005 | 实测尖端 5 mm 内才算到。求解残差不是到位 |
| `DIRECT_PBVS_MAX_SHOTS` | 3 | 全按键坐标的重试上限 |
| `DIRECT_PBVS_POSE_POS_TOL_M` | 0.045 | 长行程 IK 位置门限。38.9 mm 可以发，80 mm 必须拒绝 |
| `DIRECT_PBVS_POSE_ROT_TOL_DEG` | 25 | 长行程转角门限 |
| `DIRECT_PBVS_MIN_JOINT_DEG` | 165 | 关节跳变下限。命令行 `--direct-pbvs-max-joint-deg 120` 不能把 151° 的抬臂拦掉 |
| `SERVICE_IK_POS_ALLOW_M` | 0.040 | service 的 `allow_ik_pos_error` |

Y、Z 都进了 5 mm，下一枪才是完整按键坐标。间隙已经是 0 但 Y、Z 仍偏，每枪仍只朝按键领 5 cm。已在间隙里面则退回那个平面。不要把到位从 5 mm 放宽到 2 cm。不要恢复 3/8/12/15 cm 的 Direct PBVS 最终 standoff。

右臂平举目标是 `FOREARM_LEVEL_DEG = (-62, -48, 0, 48, 0, 0, 0)`。从低位过去时先向外翻腕，途中不得继续后仰、不得继续折肘。`level_right_waypoints` 同时用于 Right level 按钮和 PBVS 的眼在手上核对。

## 网页按钮

- **Start**：只按当前选中的那一颗。不跑动作组合。
- **组合启动**：按拖好的顺序执行，允许重复。队列空则报错并保持不动，连确认运动都不会发生。两颗之间 X 退 10 cm、Z 抬 8 cm。全部成功后只放右臂，并裁 YOLO `screen` 区域，显示在第二路相机下面，可下载。
- **Reset**：回到放下姿态，不释放 SDK。
- **释放双臂**：本页释放 SDK。
- 轨迹录制默认关。视频录制是另一个开关，也默认关。

## 改代码时保持这些不变量

- 到位看实测尖端，不看 IK 残差。5 mm 不要放宽。
- 长行程位置门限保持 45 mm。不要收紧回 25 mm，否则 38.9 mm 的抬臂又会被丢掉。80 mm 仍拒绝。
- `--direct-pbvs-max-m 1.20` 保持。
- 现场 bias 保持 `0.012 / 0.02 / 0.03`，除非人明确给出新的三个数。
- 相机序列号 `347622074029` 和 `233722073797` 不要对调，不要在首选序列号缺失时占用另一台。
- 胸口尖端保持 pose / seg。不要恢复颜色尖端，不要把浅色物体当成尖端。
- 不要在 IBVS 进程里加 ROS 规划器。
- 不要让 `arm_sdk` 和 `h2_arm_service_node` 同时发布 `rt/arm_sdk`。
- 平举不要回到肩 `-90°`、肘 `85°`。
- 测试在 `IBVS_PAD_TOUCH` 下用 `python -m pytest`。PowerShell 不接受 `&&` 和 `python -c` 里的嵌套引号。
