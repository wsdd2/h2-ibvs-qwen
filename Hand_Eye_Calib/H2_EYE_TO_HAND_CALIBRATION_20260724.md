# H2 头部 RealSense 眼在手外标定成果（2026-07-24）

## 1. 标定配置

- 模式：`eye-to-hand`
- 移动标定板手臂：右臂
- 参考坐标系：`torso_link`
- 末端 FK 坐标系：`right_wrist_yaw_link`
- 相机名称：`h2_head_d435i`
- RealSense 序列号：`347622074029`
- 彩色流：`1280x720 @ 30 FPS`
- 棋盘格：`8x5` 内角点，方格边长 `20 mm`
- 手眼算法：OpenCV Park
- 原始样本：26
- 筛选方式：按单帧 `target_reprojection_rms_px` 排除最高的 6 帧
- 最终样本：20

相机内参：

```text
/home/unitree/MscapeTech/Hand_Eye_Calib/eye_to_hand/camera_intrinsics/
  camera_intrinsics_20260723_163546_npy/camera_matrix.npy
  camera_intrinsics_20260723_163546_npy/dist_coeffs.npy
```

采集数据：

```text
/home/unitree/MscapeTech/Hand_Eye_Calib/eye_to_hand/data/
  right_arm_20260724_095657/h2_head_d435i/
```

最终结果：

```text
/home/unitree/MscapeTech/Hand_Eye_Calib/eye_to_hand/outputs/
  eye_to_hand_20260724_101155.json
  eye_to_hand_20260724_101155_npy/
```

## 2. 最终外参

结果 JSON 中：

```text
reference_frame = torso_link
cam2reference = cam2base
reference2cam = base2cam
```

`cam2reference` 是 `T_torso_camera`，将相机坐标中的点变换到
`torso_link`：

```text
P_torso = T_torso_camera * P_camera
```

```text
T_torso_camera =
[-0.0251505932, -0.0833582372,  0.9962022144,  0.0959500780]
[-0.9964258323,  0.0824761609, -0.0182549618,  0.0161696056]
[-0.0806412327, -0.9931007437, -0.0851346253,  0.0736794749]
[ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000]
```

对应平移：

```text
[0.0959500780, 0.0161696056, 0.0736794749] m
```

逆变换 `reference2cam` 是 `T_camera_torso`：

```text
T_camera_torso =
[-0.0251505932, -0.9964258323, -0.0806412327,  0.0244666178]
[-0.0833582372,  0.0824761609, -0.9931007437,  0.0798357637]
[ 0.9962022144, -0.0182549618, -0.0851346253, -0.0890178302]
[ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000]
```

## 3. 质量指标

固定关系 `target2hand` 的平移标准差：

```text
x = 1.597 mm
y = 4.141 mm
z = 3.223 mm
```

综合平移波动约 `5.5 mm`。20 个样本的旋转一致性：

```text
平均误差约 0.60 deg
最大误差约 1.67 deg
```

该结果可用于柜体、面板等目标的粗定位。用于毫米级末端精操作前，应进行
已知点或实体目标验证；必要时继续按手眼闭环残差筛选样本并复算。

## 4. 使用注意

1. `cam2reference` 才是相机点到 `torso_link` 的转换矩阵，不要误用其逆矩阵。
2. `cam2base` 是兼容字段名，本次实际 reference 是 `torso_link`。
3. 该固定外参仅在头部相机相对 `torso_link` 的安装关系不变时成立。
4. 如果 H2 的头部 pitch/yaw 会运动，需锁定在标定姿态，或改用动态
   `torso -> head` 与固定 `head -> camera` 的变换链。
5. 运行时必须使用序列号 `347622074029` 对应的相机，以及本记录中的
   `1280x720` 内参文件。
