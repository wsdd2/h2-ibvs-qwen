# Qwen 末端微调：启动与部署

胸口 PBVS 做粗定位，末端由这台电脑上的 Qwen 看右腕画面发小步位移。网页在 T40 上跑，右手串口在本体上。模型约 30GB，只留在这台 4090 上，不要上传。

三台机器：

| 机器 | 地址 | 负责 |
| --- | --- | --- |
| 本机 4090 | 局域网 IP，端口 8000 | Qwen / vLLM |
| T40 算力背包 | `mscape@192.168.1.40` | 两路相机、网页、YOLO、向 Qwen 要微调 |
| 机器人本体 | `unitree@192.168.1.3` | 右手 Revo2 串口桥 |

相机接在 T40 上。本体拉不起 RealSense，也不要在本体上跑 `start_web_remote.sh`。网页读的是本机 `/tmp` 里的 JPEG，所以相机进程和网页必须在同一台机器上。

人手离开手臂工作区，机器人先进入 AI 站立。这条网页和 Isaac 里的 `h2-uni-agent qwen` 都会占手臂，只开其中一条。

## 1. 本机上传

在 WSL 里执行。`IBVS_ROBOT_DIR` 是对方家目录下的相对路径。

T40 上能开相机的目录是 `~/MscapeTech/IBVS_PAD_TOUCH`。不要把网页同步到 `~/IBVS_PAD_TOUCH` 再从那里启动，那份没有这块板的 RealSense 和 RKNN 环境。

只把这次改过的网页文件覆盖进原目录：

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
rsync -az \
  ibvs_pad_touch/web/llm_fine.py \
  ibvs_pad_touch/web/arm_motion.py \
  ibvs_pad_touch/web/dashboard.py \
  mscape@192.168.1.40:MscapeTech/IBVS_PAD_TOUCH/ibvs_pad_touch/web/
```

本体只要右手串口桥，同一份代码再传一份：

```bash
cd /mnt/e/MscapeTech/IBVS_PAD_TOUCH
export IBVS_ROBOT_HOST=unitree@192.168.1.3
export IBVS_ROBOT_DIR=IBVS_PAD_TOUCH
./scripts/sync_to_robot.sh
```

同步会带上本机已有的 `Hand_Eye_Calib/handeye_calib/h2_ee_ik.py` 和 `h2_arm_sdk.py`，并去掉 Windows 换行。它排除 `outputs/`，所以 YOLO 权重要单独传。权重若已经在本体上：

```bash
ssh mscape@192.168.1.40 "mkdir -p IBVS_PAD_TOUCH/outputs/runs"
rsync -az --info=progress2 \
  unitree@192.168.1.3:IBVS_PAD_TOUCH/outputs/runs/ \
  mscape@192.168.1.40:IBVS_PAD_TOUCH/outputs/runs/
```

网页要读到这两个文件：

```text
~/IBVS_PAD_TOUCH/outputs/runs/yolo12_seg_20261014_1045/weights/best.pt
~/IBVS_PAD_TOUCH/outputs/runs/yolo26s_pose_20261103_1130/weights/best.pt
```

不要上传 `/mnt/e/MscapeTech/models/Qwen3.8-27B-FP8`。

## 2. 本机启动 Qwen

窗口留着。vLLM 听 `0.0.0.0:8000`。T40 要连的是这台电脑无线网的 `192.168.1.18`。WSL 里的 `hostname -I` 给出的是 `172.28.x`，背包访问不到。

然后启动模型：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -m vllm.entrypoints.cli.main serve /mnt/e/MscapeTech/models/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85
```

日志出现 `Application startup complete` 后再开 T40。Qwen 跑在 WSL 里，默认不会出现在 `192.168.1.18` 上。这一步只能在 Windows 里做端口转发，因为转发的是 Windows 网卡而不是 WSL 网卡。管理员 PowerShell：

```powershell
netsh interface portproxy add v4tov4 listenaddress=192.168.1.18 listenport=8000 connectaddress=172.28.44.172 connectport=8000
```

`172.28.44.172` 是当前 WSL 地址，WSL 重启后会变，需要重做这一条。Windows 防火墙还要放行 TCP 8000。T40 上可以先确认：

```bash
curl -sS --max-time 5 http://192.168.1.18:8000/v1/models
```

这块服务只做推理。`h2-uni-agent.exe qwen` 仍然去连 Isaac 的 `127.0.0.1:8767`，真机这条链路不要开它。

## 3. 本体启动右手串口

T40 上原来的网页使用 `192.168.1.3:8092`。本体上的桥要听 8092，不要再开一个 8091。已经在跑就不要重启：

```bash
ssh unitree@192.168.1.3
cd ~/IBVS_PAD_TOUCH
python3 scripts/revo2_bridge.py --host 0.0.0.0 --port 8092
```

这个进程占右手串口。`brainco_hand_service` 不要同时打开这只手。窗口留着。

## 4. T40 启动相机和网页

这台电脑的无线网地址是 `192.168.1.18`。地址两边不要加尖括号，bash 会把 `<...>` 当成文件重定向。网页进程在 T40 上，`127.0.0.1:8000` 指的是背包自己，不是这台 4090。

先停掉刚才从 `~/IBVS_PAD_TOUCH` 拉起的进程，它占着 8080，而且没有打开相机。然后用原来的目录和参数。`wlan0` 是 `192.168.1.40`，用来访问这台电脑的 Qwen 和本体的 `192.168.1.3:8092`。这块 T40 上没有 `eth2`。`gmac0` 和 `eth0` 没有网线，唯一已经连通的有线口是 `eth1`。CycloneDDS 只接受有 IPv4 地址的网卡，启动网页前先给 `eth1` 配上 `192.168.123.166/24`，并且不要把它设成默认路由。右手桥是 `8092`，检测走 RKNN，手眼偏置保持原来的数值。只多两行 LLM 地址：

```bash
sudo ip addr add 192.168.123.166/24 dev eth1
ping -c 2 -I eth1 192.168.123.162
```

```bash
ssh mscape@192.168.1.40
cd ~/MscapeTech/IBVS_PAD_TOUCH
export H2_LLM_URL=http://192.168.1.18:8000/v1
export H2_LLM_MODEL=Qwen3.8-27B-FP8
./scripts/start_web_remote.sh 8080 \
  --iface eth1 \
  --right-ee revo2 \
  --revo2-bridge 192.168.1.3:8092 \
  --direct-pbvs-poke \
  --direct-pbvs-max-m 1.20 \
  --direct-pbvs-max-joint-deg 120 \
  --rknn \
  --eye-to-hand-x-bias-m 0.04 \
  --eye-to-hand-y-bias-m -0.065 \
  --eye-to-hand-z-bias-m 0.035
```

这一条命令里会起来：

| 进程 | 作用 |
| --- | --- |
| `camera_sidecar.py --dual` | 同一进程打开胸口 `347622074029` 和右腕 `233722073797`，写 `/tmp/ibvs_cam_chest.jpg`、`/tmp/ibvs_cam_d435i.jpg` |
| `joint_sidecar.py` | 经 `eth1` 读 `rt/lowstate`，写 `/tmp/ibvs_joints.json` |
| `run_web.py` | 网页 `http://192.168.1.40:8080/`，RKNN 做检测，粗定位后向 `H2_LLM_URL` 要末端位移 |

不要再单独开一个相机进程，两路相机只能由这一个进程占用。日志里应出现 `dds iface=eth1`，并且不能再出现 `eth1: does not match an available interface`。`--iface` 只选择 DDS 网卡，不改变右手桥的 TCP 地址。深度图 `/depth/chest.png` 返回 503 时，是 `/tmp/ibvs_depth_chest.bin` 还没写出来，相机进程没有起来。机器人要开机并进入运控，才会在 `eth1` 上发布 `rt/lowstate`。`ip addr add` 重启后会丢，需要长期保留时在 T40 上执行 `sudo ./scripts/set_orin_h2_ip.sh eth1`。

`Ctrl+C` 会先放下右臂，大约 8 秒，放下过程中不要再按一次中断。

页面上的 Service 会在 T40 上启动 `h2_arm_service_node`，读取 `mscape` 家里的 `~/h2arm_control_ws`。这份工作空间如果只在本体的 `unitree` 家里，页面能看画面，手臂服务起不来。

## 5. 网页上启动 LLM Hybrid

浏览器打开 `http://192.168.1.40:8080/`。

单次：点 `LLM Hybrid`，选按键，再点 Start。

组合：在组合区点 `LLM Hybrid`，排好按键，再点「组合启动」。

两种启动都会先访问 `H2_LLM_URL/models`。连接超时 60 秒，这 60 秒只算连通，不算后面看图推理。连不上会停在启动前，手臂不会动。连通之后，胸口 PBVS 粗定位，末端每步向 Qwen 要躯干系 `delta_xyz`，每轴不超过 15 mm。接触仍由现有腕部规则结束。

默认模型名是 `Qwen3.8-27B-FP8`，和 vLLM 的 `--served-model-name` 一致。
