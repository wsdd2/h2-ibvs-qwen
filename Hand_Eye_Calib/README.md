# RealSense D435i 灵巧手手眼标定

本项目用于 UniTree H2 机器人灵巧手/夹爪与 RealSense D435i 的手眼标定，支持两种安装方式：

- `eye-in-hand` / `hand-in-eye`：D435i 安装在灵巧手、手腕或末端上，棋盘格固定在外部。适合末端相机做开关、旋钮、按钮的近距离精定位。
- `eye-to-hand` / `hand-to-eye`：D435i 固定在机器人头部、机身、三脚架或外部，棋盘格固定在灵巧手、手腕或末端上。适合头部/外部相机做电柜、柜门、面板的粗定位。

如果系统同时有末端相机和头部相机，建议分别采集和求解两套标定结果，不要把两台相机的数据混在同一个目录里。

当前 H2 头部 RealSense 到 `torso_link` 的眼在手外标定成果见
[`H2_EYE_TO_HAND_CALIBRATION_20260724.md`](H2_EYE_TO_HAND_CALIBRATION_20260724.md)。

当前 H2 右腕 RealSense 到 `right_wrist_yaw_link` 的眼在手上标定成果见
[`H2_EYE_IN_HAND_CALIBRATION_20260726.md`](H2_EYE_IN_HAND_CALIBRATION_20260726.md)。

## 坐标约定

每次采集需要输入机器人当前的 6D 位姿：

```text
x y z rx ry rz
```

默认解释为：

```text
T_base_hand
```

即点从灵巧手/末端坐标系变换到机器人基座坐标系。默认单位：

```text
平移：mm
旋转：deg
欧拉角约定：legacy（默认，R=Rx*Ry*Rz）
```

`hand` 可以是 `wrist`、`flange`、`palm` 或现场定义的灵巧手安装坐标，但一次标定中必须始终使用同一个刚体坐标。如果机器人控制器输出的是 TCP 位姿，请确认 TCP 和相机/棋盘安装刚体关系在采集过程中不变。

手工输入若来自 ROS RPY，请显式添加 `--pose-euler-convention ros-rpy`；
此时输入仍为 `roll pitch yaw`，旋转按 `Rz(yaw)*Ry(pitch)*Rx(roll)` 计算。

## 安装依赖

推荐在 Ubuntu ROS2 机器人主机上运行采集和求解，通过 Windows 11 的 Cursor Remote SSH 编辑与调试代码。RealSense、ROS2、Unitree 控制接口都留在机器人主机侧，Windows 只作为开发界面。

机器人 Ubuntu 主机：

```bash
sudo apt update
sudo apt install openssh-server python3-venv
sudo systemctl enable ssh
sudo systemctl start ssh
```

Windows 11 验证 SSH：

```powershell
ssh robot_user@192.168.1.50
```

项目依赖：

```powershell
cd E:\MscapeTech\Hand_Eye_Calib
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 采集并求解：Eye-In-Hand

适用：相机装在灵巧手上，棋盘格固定在桌面或外部支架上。

```powershell
python capture_handeye.py --mode eye-in-hand --cols 11 --rows 8 --square-mm 30 --hand-frame wrist --camera-name wrist_d435i --camera-mount wrist --cam-serial 123456789 --robot-host 192.168.1.50 --robot-user robot_user --ros-distro humble --ros-domain-id 20
```

求解结果核心文件：

```text
outputs/eye_in_hand_时间戳.json
outputs/eye_in_hand_时间戳_npy/T_cam2hand.npy
outputs/eye_in_hand_时间戳_npy/T_hand2cam.npy
```

其中 `T_cam2hand` 是相机坐标到灵巧手/末端坐标的变换。

## 采集并求解：Eye-To-Hand

适用：相机固定在外部，棋盘格固定在灵巧手上。

```powershell
python capture_handeye.py --mode eye-to-hand --arm-side right --fk-base-link torso_link --cols 11 --rows 8 --square-mm 30 --camera-name head_d435i --camera-mount head --cam-serial 987654321
```

H2 左臂/Revo2 标定时，标定板应与左腕保持刚性固定：

```powershell
python capture_handeye.py --mode eye-to-hand --arm-side left --fk-base-link torso_link --hand-frame left_wrist_yaw_link --cols 11 --rows 8 --square-mm 30 --camera-name head_d435i --camera-mount head --cam-serial 987654321
```

`--arm-side` 决定默认 FK 末端：左侧为 `left_wrist_yaw_link`，右侧为
`right_wrist_yaw_link`。即使 Revo2 手指模型尚未加入 URDF，也可用腕部刚体坐标完成标定。
眼在手外最终求得的是头部相机到 `torso_link` 的固定外参；左右臂只是两种移动标定板的采集方式，
结果原则上应一致，也可互相用于独立验证。

求解结果核心文件：

```text
outputs/eye_to_hand_时间戳.json
outputs/eye_to_hand_时间戳_npy/T_cam2base.npy
outputs/eye_to_hand_时间戳_npy/T_base2cam.npy
outputs/eye_to_hand_时间戳_npy/T_cam2reference.npy
outputs/eye_to_hand_时间戳_npy/T_reference2cam.npy
```

其中兼容字段 `T_cam2base` 与 `T_cam2reference` 内容相同；当
`--fk-base-link torso_link` 时，二者都是相机坐标到 torso 坐标的变换。

## 多相机建议

先列出当前 RealSense 设备：

```powershell
python -c "from handeye_calib.camera import RealSenseD435i; print(RealSenseD435i.list_devices())"
```

优先使用 `--cam-serial` 选择相机，而不是只使用 `--cam-index`。多相机、远程 Ubuntu、USB 重插或机器人重启后，设备 index 可能变化，serial 更稳定。

建议命名：

```text
wrist_d435i：末端/手腕相机，mode=eye-in-hand，camera-mount=wrist
head_d435i：头部/外部相机，mode=eye-to-hand，camera-mount=head
```

采集结果会按相机名称分目录保存：

```text
data/eye_in_hand_时间戳/wrist_d435i/
data/eye_to_hand_时间戳/head_d435i/
```

求解 JSON 中会保留 `camera_info`、`robot_context`、`camera_mount`、`hand_frame` 等上下文，便于后续把结果接入 ROS2 TF 树。

推荐 TF 使用方式：

```text
eye-in-hand：发布 base -> wrist，以及 wrist -> wrist_d435i_color
eye-to-hand：发布 base -> head_d435i_color 或 base -> external_d435i_color
```

电柜操作建议由头部/外部相机先做柜体和面板粗定位，再由末端相机靠近后做按钮、拨动开关、旋钮的精定位和操作后复核。

## 交互按键

```text
Space：当前帧检测到棋盘格后，输入机器人 hand 位姿并保存样本
Enter / s：样本数足够时执行求解
Esc / q：退出
```

建议采集 `12-20` 组样本，最低默认 `8` 组。样本不要只平移，要包含明显的 X/Y/Z 方向旋转变化。

## 相机内参

先用棋盘格采图工具保存相机标定图片：

```powershell
python -m handeye_calib.chessboard --task camera_calib --cols 11 --rows 8 --cam-serial 123456789
```

默认保存到：

```text
camera_calib_image/
```

默认直接读取当前 D435i 彩色流 profile 内参。也可以指定已有标定结果：

```powershell
python capture_handeye.py --mode eye-in-hand --camera-matrix-npy config\camera_matrix.npy --dist-coeffs-npy config\dist_coeffs.npy
```

`calibrate_camera.py` 会同时生成 `intrinsics_manifest.json`，记录相机序列号、
型号、流名称、分辨率、内参值和 NPY 文件 SHA256。采集时若发现 manifest
与当前相机/profile/文件内容不一致，会拒绝继续，避免误用另一台相机或分辨率的内参。

或指定包含 `color.fx/fy/ppx/ppy/coeffs` 的 JSON：

```powershell
python capture_handeye.py --mode eye-to-hand --camera-json config\d435i_color.json
```

## 离线重新求解

如果只需要采集用于手眼标定的棋盘图片，不在采图阶段输入机器人位姿：

```powershell
python -m handeye_calib.chessboard --task handeye --cols 11 --rows 8 --cam-serial 123456789
```

默认保存到：

```text
handeye_img/
```

`handeye_calib/chessboard.py` 只负责视频流中的棋盘格检测、预览标注和图片保存；相机畸变参数计算、PnP 位姿计算和手眼标定求解由其他脚本/模块实现。

如果已经采集完数据，只想换方法或复算：

```powershell
python solve_offline.py --mode eye-in-hand --data-dir data\eye_in_hand_20260615_160000\camera_0
```

可以切换 OpenCV 方法：

```powershell
python solve_offline.py --mode eye-to-hand --data-dir data\eye_to_hand_20260615_160000\camera_0 --handeye-method park
```

## 质量检查

采集时关注 `pnp_rms`，通常越小越好。求解器会校验所有样本的模式、基准/末端
坐标系、相机序列号、棋盘参数、内参和 SE(3) 矩阵，并拒绝旋转激励不足或转轴
近似共线的数据。求解 JSON 的 `quality` 还包含固定关系残差、样本两两闭环最大值、
留一法外参稳定性和明确的 `pass/warn` 状态；数值越小越稳定。

自动 FK 采集会同时检查图像与 `rt/lowstate` 的时间差、状态新鲜度和
手臂关节速度。默认阈值分别为 `0.10 s`、`0.25 s` 和 `0.08 rad/s`；
超过阈值会拒绝保存，避免把运动中的图像和旧 FK 配成一组样本：

```bash
--fk-sync-max-delta-sec 0.10 \
--fk-max-state-age-sec 0.25 \
--capture-max-joint-velocity-rad-s 0.08
```

如果机器人跨机 DDS 只转发低频状态，使用
`--fk-lowstate-topic rt/lf/lowstate`。显式启用 `--stream-fk` 后，FK Topic
初始化或首次状态读取失败会直接终止启动，不再静默退回手工位姿。

每个新采集 JSON 会记录相机帧/FK 时间、原始关节角和速度、URDF 绝对路径
及 SHA256。旧 JSON/NPY 仍可离线读取。

相机安装位必须与模式一致：`wrist/hand` 对应 `eye-in-hand`，
`head/abdomen/torso/fixed` 对应 `eye-to-hand`。fallback 到另一安装位时
程序默认拒绝采集；只有确认特殊机械安装后才能使用
`--allow-mount-mode-mismatch` 放行。

手工输入的欧拉角应明确区分 ROS RPY（固定轴，`Rz*Ry*Rx`）和历史
legacy xyz（`Rx*Ry*Rz`）。自动 FK 直接保存 4x4 矩阵，不经过欧拉角重建。

坐标基准的物理约束：

- eye-in-hand 且棋盘固定在外部时，作为基准的 `torso_link` 在整组采集期间
  必须相对棋盘保持静止。
- eye-to-hand 且相机刚性安装在 torso 上时，机器人可以整体运动，但相机与
  `torso_link` 之间不能发生相对运动。

眼在手上的固定求解结果是 `T_cam2hand`（即 `T_hand_cam`），不是固定的
`T_torso_cam`。相机随右腕运动，运行时必须用当前关节状态逐帧计算：
`T_torso_cam(q) = T_torso_wrist(q) * T_wrist_cam`。

如果误差明显偏大，优先检查：

- 棋盘内角点数量和方格尺寸是否填对。
- 输入位姿是否真的是同一个 `hand` 坐标相对 `base` 的位姿。
- 平移单位、旋转单位和欧拉角顺序是否与机器人控制器一致。
- eye-in-hand 和 eye-to-hand 模式是否选反。
- 相机、棋盘格或灵巧手安装件是否在采集过程中松动。
