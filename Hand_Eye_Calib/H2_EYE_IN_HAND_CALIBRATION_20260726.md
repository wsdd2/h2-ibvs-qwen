# H2 右腕 RealSense 眼在手上标定成果（2026-07-26）

## 1. 标定配置

- 模式：`eye-in-hand`
- 相机安装：右腕 / `wrist`
- 参考坐标系（FK base）：`torso_link`
- 手眼刚体坐标系：`right_wrist_yaw_link`
- 相机名称：`right_wrist_d435`
- RealSense 序列号：`346522074739`
- 彩色流：`1280x720 @ 30 FPS`
- 棋盘格：`8x5` 内角点，方格边长 `20 mm`
- 手眼算法：OpenCV Tsai
- 筛选方式：
  - 先按单帧 `target_reprojection_rms_px` 排除最高的 5 帧
  - 再按手眼闭环旋转误差排除 `> 1.0 deg` 的样本
- 最终样本：24

相机内参（腕部相机独立标定）：

```text
/home/unitree/MscapeTech/Hand_Eye_Calib/eye_in_hand/wrist_camera_intrinsics/
  camera_intrinsics_*_npy/camera_matrix.npy
  camera_intrinsics_*_npy/dist_coeffs.npy
```

采集数据：

```text
/home/unitree/MscapeTech/Hand_Eye_Calib/eye_in_hand/data/
  right_wrist_torso_20260726_114239/right_wrist_d435/
```

最终结果：

```text
/home/unitree/MscapeTech/Hand_Eye_Calib/eye_in_hand/outputs/
  eye_in_hand_20260726_115420.json
  eye_in_hand_20260726_115420_npy/
```

## 2. 最终外参

`cam2hand` 是 `T_right_wrist_yaw_link_camera`，将相机坐标中的点变换到
`right_wrist_yaw_link`：

```text
P_wrist = T_cam2hand * P_camera
```

```text
T_cam2hand =
[ 0.0157973687,  0.0296908824,  0.9994342873,  0.1165964762]
[ 0.0582501946,  0.9978340307, -0.0305640631, -0.0856009992]
[-0.9981770173,  0.0587000735,  0.0140336524,  0.0218166443]
[ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000]
```

对应平移：

```text
[0.1165964762, -0.0856009992, 0.0218166443] m
```

逆变换 `hand2cam` 是 `T_camera_right_wrist_yaw_link`：

```text
T_hand2cam =
[ 0.0157973687,  0.0582501946, -0.9981770173,  0.0249212303]
[ 0.0296908824,  0.9978340307,  0.0587000735,  0.0806730992]
[ 0.9994342873, -0.0305640631,  0.0140336524, -0.1194529977]
[ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000]
```

## 3. 到 torso 的运行时换算

眼在手上标定本身给出的是相机到手腕的固定外参。运行时若需要
`torso_link` 坐标：

```text
T_torso_camera
= T_torso_right_wrist_yaw_link
  × T_cam2hand
```

```text
P_torso = T_torso_right_wrist_yaw_link * T_cam2hand * P_camera
```

其中 `T_torso_right_wrist_yaw_link` 必须用当前关节角实时 FK / TF 获取，
不能写成固定矩阵。

## 4. 质量指标

固定关系 `target2base`（此处 base 实际为 `torso_link`）的平移标准差：

```text
x = 3.277 mm
y = 4.192 mm
z = 2.626 mm
```

综合平移波动约 `5.9 mm`。24 个样本的旋转一致性：

```text
平均误差约 0.40 deg
最大误差约 0.84 deg
```

相对此前失败批次，结果已明显改善，可用于近距离目标粗定位与验证。
毫米级精操作前仍建议做实体点 / 已知目标实测。

## 5. 使用注意

1. 正式结果优先使用：
   `eye_in_hand_20260726_115420_npy/T_cam2hand.npy`
2. 该外参相对 `right_wrist_yaw_link` 固定；若改用 `R_ee` 或其他虚拟 TCP，
   需额外叠乘 `wrist -> TCP` 固定变换。
3. 运行时必须使用序列号 `346522074739` 的右腕相机，以及同一次
   `1280x720` 腕部内参。
4. 不要与头部相机眼在手外结果
   `eye_to_hand_20260724_101155` 混用。
5. 替换旧正式结果
   `outputs/eye_in_hand_20260630_150210.json`
   前，先完成 ROS / 感知链路实测验证。
