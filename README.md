# IBVS_PAD_TOUCH

宇树 H2 右臂 + 闭合 Dex1-1 + **胸部 D435i**。面板固定，相机固定。本仓库先在图像里定位整块面板，再用比例尺把 Button 1/2/3 投到画面上（黄色质心只做修正）。有指尖观测之后，才输出指尖到按钮的相机系 `dx, dy, dz`。不控夹爪、不控上身、不走 ROS、**不做手眼标定**。

## 正确顺序

1. **定位整块面板**（四角 / 米色壳体）
2. **比例尺为主**：把尺量模型通过 Homography 投到当前图像，得到 Button 1/2/3
3. **视觉为辅**：只在预测框里用黄条质心微调，不靠颜色认“这是几号键”
4. **再研究指尖**：同一相机里看到闭合 Dex1 尖端后，`delta = P_button − P_tip`（相机系）
5. **先左右 / 上下对准开关**：只执行相机 `dx, dy`，深度不动
6. **再把深度顶到开关面板**：开关对准后才执行相机 `dz`，目标是黄键所在面板，不是后面的墙
7. 当前阶段的增量直接给运控当笛卡尔增量

胸部相机是眼在手外。手臂移动时，**按钮在画面里几乎不动**，被伺服的是指尖去就按钮，不是把按钮“吸”到画面某个固定像素。

## 为什么不用手眼标定

这里不做 PBVS：不重建面板在机器人基座下的位姿，也不算 `T_base_cam * P_button`。

每周期只告诉运控：**当前相机画面里，指尖还差按钮多少米**。约定胸部 D435i 的轴就是运控笛卡尔轴：

- X 右
- Y 下
- Z 光轴向前（靠近面板为正）

这是安装轴向约定，不是 Tsai 手眼。相机若明显俯仰/滚转，只允许以后做一次固定轴置换，仍不在本仓库做手眼标定。

## 按钮

| 名称 | 面板键 | 说明 |
|---|---|---|
| Button 1 | `<` | 右上组合黄键左半 |
| Button 2 | `>` | 同一颗键右半 |
| Button 3 | OK | 正下方 |

面板 **300 × 130 mm**。Button 1/2 中心 `(238.75, 30)` / `(256.25, 30)` mm，Button 3 `(247.50, 50)` mm。

## 阶段

| 阶段 | 输出 | 现状 |
|---|---|---|
| 匹配面板 | `SEEK_PANEL` / `LOCALIZED`，不动 | 先认整块面板和三颗键 |
| 左右 / 上下 | `ALIGN_XY`，只有相机 `dx, dy` | 先对准开关，不进深 |
| 深度 | `APPROACH_Z`，只有相机 `dz` | 对准后再顶到开关面板，不到墙 |
| 到位 | `HOLD` | 指尖停在黄键上 |
| 运控 | 按同一组轴执行增量 | Isaac 已接；实机未接 |

实机胸部相机暂时看不到指尖时，`--no-tip` / 默认实机路径只做定位，不会假装已经在戳。

## 实机网页监视（sim2real 第一步）

H2 是 **Ubuntu 22.04**。SSH 上车后用 bash 启动 Flask；本机浏览器再打开车上的地址。  
`unitree@192.168.1.3`，胸口相机序列号 `347622074029`。**Button 1 / 2 / 3 / Cycle / Start / Stop / Reset 已预留，这一步不发 `arm_sdk`。**

```bash
ssh unitree@192.168.1.3
cd ~/IBVS_PAD_TOUCH
chmod +x scripts/*.sh
./scripts/start_web_remote.sh 8080
```

日志打在这个 SSH 窗口里，**Ctrl+C** 同时停掉网页和关节sidecar。不要在 WSL 里跑这条（`~/IBVS_PAD_TOUCH` 会变成 `/home/wsdd2/...`）。

然后在你自己的电脑浏览器打开：

```
http://192.168.1.3:8080/
```

## 运行

开发机默认是 **WSL2 Ubuntu**；上车是 **H2 原生 Ubuntu 22.04**。两边都不依赖 X11：OpenCV 用 headless，结果写 `outputs/`。

### WSL2（本机）

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh
./scripts/run_ibvs.sh --source synthetic --target button_1 --no-tip
./scripts/run_ibvs.sh --source synthetic --target button_1
```

指定解释器（和 `mt_act_panel` 一样会优先 `~/miniconda3/envs/rmb`）：

```bash
export IBVS_PYTHON="$HOME/miniconda3/envs/rmb/bin/python"
```

WSL 里 RealSense USB 转发不稳定，本机只跑合成 / 静态图。`--show` 没有 `DISPLAY` 时会自动改成只写图。

Windows 里也能直接：

```powershell
cd E:\MscapeTech\IBVS_PAD_TOUCH
pip install -r requirements.txt
python -m pytest tests -q
python scripts\run_ibvs.py --smoke
python scripts\annotate_photos.py
```

实拍标注图在 `outputs/annotate/`。白墙上先找高饱和黄键：最上条左右对半是 Button 1/2，下一条是 Button 3。

### 上传到 H2（Ubuntu 22.04）

```bash
export IBVS_ROBOT_HOST=unitree@192.168.1.3
export IBVS_ROBOT_DIR=IBVS_PAD_TOUCH
./scripts/sync_to_robot.sh
```

不要写未加引号的 `~/IBVS_PAD_TOUCH`：会在本机 WSL 用户下展开成 `/home/你的用户名/...`，新车上没有这个目录。路径相对车上 `unitree` 的家目录。新机器会先 `mkdir` 再传。

车上：

```bash
cd ~/IBVS_PAD_TOUCH
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh --robot          # 可选再加 --apt
# 当前用户要能读 D435i：sudo usermod -aG video $USER
./scripts/run_ibvs.sh --source realsense --target button_1
```

`--robot` 会尝试 `pip install pyrealsense2`。H2 上如果已经有宇树 / librealsense 环境，用 `IBVS_PYTHON` 指过去即可，不必再装一份。
