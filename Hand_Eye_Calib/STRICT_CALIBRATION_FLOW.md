# H2 右腕 D435 严格内外参标定流程

本流程保留 `capture_handeye.py` 的硬件采集能力，但不再把同批闭环
`PASS` 直接视为可部署。最终结果必须通过：

- 内参画面覆盖与随机子集稳定性；
- 图像/FK同步、关节完整性和静止门禁；
- Park/Horaud/Tsai 交叉验证；
- 时间顺序前后半、奇偶半和随机子集稳定性；
- 可选的跨会话基线；
- 未参与求解的新姿态绝对验证。

## 0. 上传并部署到Orin数据盘

禁止把新代码、图片、NPY和标定结果放入1.25系统盘。本文统一使用：

```text
Orin宿主机数据盘：/mnt/data/Repos/yck_calibration
humble_v8容器映射：/root/Repos/yck_calibration
```

### 0.1 从Windows的WSL创建数据盘目录

在WSL执行：

```bash
ssh user@192.168.1.25
```

SSH密码在交互提示中输入：`admin`。不要把密码写入脚本或命令参数。

登录Orin宿主机后执行：

```bash
mkdir -p \
  /mnt/data/Repos/yck_calibration/Hand_Eye_Calib \
  /mnt/data/Repos/yck_calibration/robot_kinematics \
  /mnt/data/Repos/yck_calibration/urdf \
  /mnt/data/Repos/yck_calibration/artifacts

df -h /mnt/data
exit
```



### 0.2 从WSL上传代码

以下命令均在WSL执行，每次提示密码时输入 `admin`：

```bash
rsync -avP \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".git/" \
  --exclude "eye_in_hand/" \
  --exclude "outputs/" \
  /mnt/e/MscapeTech/Hand_Eye_Calib/ \
  user@192.168.1.25:/mnt/data/Repos/yck_calibration/Hand_Eye_Calib/

rsync -avP \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  /mnt/e/MscapeTech/robot_kinematics/ \
  user@192.168.1.25:/mnt/data/Repos/yck_calibration/robot_kinematics/

rsync -avP \
  /mnt/e/MscapeTech/unitree_ros/robots/h2_description/H2.urdf \
  user@192.168.1.25:/mnt/data/Repos/yck_calibration/urdf/H2.urdf
```

不要上传到 `/home/user/MscapeTech/Hand_Eye_Calib`，也不要使用 `docker cp`
复制到容器可写层；两者都会继续占用系统盘。

### 0.3 进入容器并初始化ROS环境

```bash
ssh user@192.168.1.25
```

SSH密码输入 `admin`，然后：

```bash
sudo docker ps --filter name=humble_v8
sudo docker exec -it humble_v8 bash
```

sudo密码输入 `admin`。进入容器后执行：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
unset ftp_proxy FTP_PROXY PIP_PROXY
export NO_PROXY='*'
export no_proxy='*'

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ros/humble/bin
unset PYTHONPATH
hash -r

source /opt/ros/humble/setup.bash
source /root/Repos/robotics/install/setup.bash
source /root/Repos/yck_data/ws/install/local_setup.bash

export ROS_DOMAIN_ID=60
unset ROS_LOCALHOST_ONLY
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source /root/Repos/yck_cabinet/.venv/bin/activate

export CALIB_ROOT=/root/Repos/yck_calibration
export CODE_ROOT="$CALIB_ROOT/Hand_Eye_Calib"
export ARTIFACT_ROOT="$CALIB_ROOT/artifacts"
cd "$CODE_ROOT"
```

确认代码和输出都位于数据盘：

```bash
df -h / /root/Repos
python strict_calibration_pipeline.py --help
python -m unittest discover -s tests -v
```

如果 `/root/Repos` 没有映射到宿主机 `/mnt/data/Repos`，必须先修正Docker挂载，
不得继续向容器根文件系统写入标定数据。

## 1. 机械前提

- 相机、Dex1-1基座、右腕之间必须刚性固定，并做好线缆应力释放。
- 手眼采集的整个会话期间，双脚、pelvis、腰部、torso和棋盘均不得移动。
- 只允许右臂关节运动。
- 相机若固定在Dex1-1主壳体上，可以继续以
`right_wrist_yaw_link` 为 hand frame；该固定安装偏移会被外参吸收。



## 2. 采集内参图片

```bash
export CALIB_ROOT=/root/Repos/yck_calibration
export CODE_ROOT="$CALIB_ROOT/Hand_Eye_Calib"
export ARTIFACT_ROOT="$CALIB_ROOT/artifacts"
cd "$CODE_ROOT"
source /root/Repos/yck_cabinet/.venv/bin/activate

INTR_IMAGES="$ARTIFACT_ROOT/intrinsics/strict_images_$(date +%Y%m%d_%H%M%S)"

python -m handeye_calib.chessboard \
  --task camera_calib \
  --cam-serial 346522074739 \
  --camera-backend realsense \
  --color-only \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --cols 11 \
  --rows 8 \
  --gamma 1.0 \
  --output-dir "$INTR_IMAGES" \
  --stream-debug \
  --stream-host 0.0.0.0 \
  --stream-port 8080 \
  --headless
```

至少保存30张。棋盘角点集合必须覆盖画面中心、四边和四角，并包含多个倾角；
不要只在画面中心改变距离。

## 3. 严格求解内参

```bash
python strict_calibration_pipeline.py intrinsics \
  --image-dir "$INTR_IMAGES" \
  --output-root "$ARTIFACT_ROOT/intrinsics" \
  --cols 8 \
  --rows 5 \
  --square-mm 20 \
  --gamma 1.0 \
  --camera-serial 346522074739 \
  --camera-model "RealSense D435" \
  --stream-name color \
  --min-images 24 \
  --max-rms-px 0.35 \
  --max-view-rms-px 0.50
```

只有 `[INTRINSICS][PASS]` 目录才会生成 `intrinsics_manifest.json`。
FAIL目录只保留候选NPY和诊断报告，禁止用于手眼采集。

## 4. 采集 eye-in-hand 外参数据

把 `INTR_DIR` 改为上一步的PASS目录：

```bash
export INTR_DIR="<camera_intrinsics_strict_*_npy PASS目录>"
SESSION="right_wrist_torso_strict_$(date +%Y%m%d_%H%M%S)"

python capture_handeye.py \
  --mode eye-in-hand \
  --strict-capture \
  --robot-model "Unitree H2" \
  --arm-side right \
  --fk-base-link torso_link \
  --hand-frame right_wrist_yaw_link \
  --fk-target right_wrist_yaw_link \
  --fk-urdf "$CALIB_ROOT/urdf/H2.urdf" \
  --fk-state-source ros-joint-states \
  --fk-joint-state-topic /joint_states \
  --fk-sync-max-delta-sec 0.03 \
  --fk-max-state-age-sec 0.10 \
  --capture-max-joint-velocity-rad-s 0.01 \
  --cam-serial 346522074739 \
  --no-cam-fallback \
  --camera-name right_wrist_d435 \
  --camera-mount wrist \
  --color-only \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --cols 8 \
  --rows 5 \
  --square-mm 20 \
  --gamma 1.0 \
 --camera-matrix-npy "$INTR_DIR/camera_matrix.npy" \
--dist-coeffs-npy "$INTR_DIR/dist_coeffs.npy" \
--data-root "$ARTIFACT_ROOT/eye_in_hand/data" \
--output-dir "$ARTIFACT_ROOT/eye_in_hand/outputs" \
--session-name "$SESSION" \
--stream-debug \
--stream-fk \
--headless \
--stream-host 0.0.0.0 \
--stream-port 8081 \
--min-samples 30 \
--capture-max-reproj-rms-px 0.30 \
--handeye-method park
```

每次移动完成后等待1–2秒再Save。采集30–40个姿态，必须包含肩、肘、腕多轴
旋转。网页中的旧Solve按钮只用于快速诊断，部署结果必须由下一步生成。

## 5. 严格求解手眼外参

```bash
DATA_DIR="$ARTIFACT_ROOT/eye_in_hand/data/$SESSION/right_wrist_d435"

python strict_calibration_pipeline.py handeye \
  --data-dir "$DATA_DIR" \
  --output-dir "$ARTIFACT_ROOT/eye_in_hand/outputs" \
  --mode eye-in-hand \
  --method park \
  --camera-serial 346522074739 \
  --width 1280 \
  --height 720 \
  --base-frame torso_link \
  --hand-frame right_wrist_yaw_link \
  --urdf-sha256 e757acabca57429e1888e1e4847b2382cf05dfeaad87d0359da1af689f7d0922 \
  --min-samples 24 \
  --max-reprojection-rms-px 0.30 \
  --max-sync-delta-sec 0.03 \
  --max-state-age-sec 0.10 \
  --max-joint-velocity-rad-s 0.01 \
  --subset-trials 100
```

只有输出JSON同时满足以下条件才可部署：

```text
deployable: true
strict_quality.deployable: true
```



## 6. 重复会话验证

不触碰相机、支架、棋盘和torso，重新采集第二个完全独立会话，再比较：

```bash
python strict_calibration_pipeline.py compare \
  --first <第一次strict PASS.json> \
  --second <第二次strict PASS.json> \
  --mode eye-in-hand \
  --max-translation-mm 5 \
  --max-rotation-deg 0.5
```

也可以在第二次严格求解时直接指定：

```bash
--baseline-json <第一次strict PASS.json>
```



## 7. 未参与求解的绝对验证

另采至少8个新姿态，禁止复用求解样本。独立测量棋盘参考点在torso下的真值：

```bash
python validate_eye_in_hand_absolute.py \
  --validation-data-dir <独立验证数据目录> \
  --handeye-json <strict PASS.json> \
  --truth-xyz-m <X> <Y> <Z> \
  --reference-point inner-grid-center \
  --min-samples 8 \
  --max-reprojection-rms-px 0.30 \
  --pass-rmse-mm 5 \
  --pass-p95-mm 8 \
  --pass-max-mm 10
```

未通过独立绝对验证的结果不得用固定平移或单点补偿伪装为合格外参。

## 8. 将严格PASS外参部署到柜检sidecar

不要把结果复制回 `/home/user`。直接使用数据盘上的最终JSON：

```bash
export STRICT_RESULT="<绝对验证通过的 eye_in_hand_strict_*.json>"

python3 - "$STRICT_RESULT" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
assert payload.get("deployable") is True, "top-level deployable is not true"
assert payload.get("strict_quality", {}).get("deployable") is True, \
    "strict_quality.deployable is not true"
assert "cam2hand" in payload and "transform" in payload["cam2hand"], \
    "cam2hand.transform missing"
print("STRICT HANDEYE DEPLOY CHECK PASSED:", path)
PY
```

启动sidecar时直接替换手眼和URDF参数：

```bash
ros2 launch /root/Repos/yck_data/ws/src/vision_manager/launch/cabinet_sidecar_launch.py \
  cabinet_yoloseg_launch_file:=/root/Repos/yck_data/ws/src/cabinet_perception/launch/cabinet_yoloseg_launch.py \
  cabinet_params_file:=/root/Repos/yck_data/ws/src/cabinet_perception/config/cabinet_orin_3d.yaml \
  model_path:=/root/Repos/yck_cabinet/models/yolo26_seg_cabinet_best.pt \
  enable_live_fk:=true \
  handeye_path:="$STRICT_RESULT" \
  urdf_path:=/root/Repos/yck_calibration/urdf/H2.urdf
```

其余残差和业务参数按现场命令追加。启动日志必须显示：

```text
[SELF-TEST][cabinet_projector] handeye PASSED path=/root/Repos/yck_calibration/...
[SELF-TEST][cabinet_projector] FK PASSED urdf=/root/Repos/yck_calibration/urdf/H2.urdf
```

代码、图片、内参、外参和验证报告全都保留在 `/root/Repos/yck_calibration`
挂载盘中；容器重建或系统盘清理时不会丢失。