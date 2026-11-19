# h2-ibvs-qwen

宇树 H2 面板点按。胸口相机做 PBVS 粗定位，Qwen 看右腕画面做末端微调。右手末端可以是 Revo2 食指，也可以是 Dex1 夹爪。

人员先离开手臂工作区，机器人进入运控站立。这套网页会占用手臂，不要同时开另一套运控。

躯干坐标：`+X` 朝面板，`+Y` 向左，`+Z` 向上。

## 末端

`--right-ee` 决定点按用哪一个点。视觉和 PBVS 不变，变的是腕部工具原点和右手怎么摆。

| | Revo2 食指 | Dex1 夹爪 |
| --- | --- | --- |
| 启动 | `--right-ee revo2` | `--right-ee dex1`，这是默认值 |
| 点按点 | `right_index_tip_link` | 夹爪尖，腕部偏移 `(0.205, 0, 0)` m |
| 右手 | 本体串口桥 `revo2_bridge.py` | 不需要串口桥 |
| 预备 | 食指伸出，其余手指收拢 | 保持夹爪 |

换一只新的手时，沿用这个分法：在腕坐标系里给出点按点，需要手指动作时再加一条和 `revo2_bridge.py` 一样的本机桥。不要把人类手臂关节角直接映射到 H2。

## 机器

| 机器 | 例子 | 作用 |
| --- | --- | --- |
| 带 4090 的电脑 | `192.168.1.18:8000` | 只跑 Qwen。模型约 30GB，不要放进仓库 |
| 算力背包 T40 | `mscape@192.168.1.40` | 相机、网页、RKNN。工作目录 `~/MscapeTech/IBVS_PAD_TOUCH` |
| 机器人本体 | `unitree@192.168.1.3` | 只有 Revo2 时才需要，跑右手串口桥 |

相机接在 T40 上。本体拉不起 RealSense。胸口 `347622074029`，右腕 `233722073797`。两路相机只能由这一个网页进程打开。

仓库不包含 YOLO 权重和 Qwen 权重。检测权重要另外放到：

```text
outputs/runs/yolo12_seg_20261014_1045/weights/best.rknn
outputs/runs/yolo26s_pose_20261103_1130/weights/best.rknn
```

T40 上用 `--rknn`。没有 NPU 时去掉 `--rknn`，改放同目录的 `best.pt`。

## 1. Qwen

在 4090 的 WSL 里。模型目录按本机实际路径改。

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

出现 `Application startup complete` 后再开 T40。T40 访问的是 Windows 无线网地址，不是 WSL 里的 `172.28.x`。WSL 重启后地址会变，端口转发要重做。管理员 PowerShell：

```powershell
netsh interface portproxy add v4tov4 listenaddress=192.168.1.18 listenport=8000 connectaddress=<WSL地址> connectport=8000
```

防火墙放行 TCP 8000。T40 上确认：

```bash
curl -sS --max-time 5 http://192.168.1.18:8000/v1/models
```

## 2. 本体（仅 Revo2）

Dex1 跳过这一节。

```bash
ssh unitree@192.168.1.3
cd ~/IBVS_PAD_TOUCH
python3 scripts/revo2_bridge.py --host 0.0.0.0 --port 8092
```

这个窗口留着。`brainco_hand_service` 不要同时打开这只手。

## 3. T40

这块板上没有 `eth2`。有线口是 `eth1`，先给它机器人网段的地址，不要设成默认路由。

```bash
ssh mscape@192.168.1.40
sudo ip addr add 192.168.123.166/24 dev eth1
ping -c 2 -I eth1 192.168.123.162
```

Revo2：

```bash
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

Dex1：去掉 `--right-ee` 和 `--revo2-bridge` 两行，其余相同。

```bash
./scripts/start_web_remote.sh 8080 \
  --iface eth1 \
  --direct-pbvs-poke \
  --direct-pbvs-max-m 1.20 \
  --direct-pbvs-max-joint-deg 120 \
  --rknn \
  --eye-to-hand-x-bias-m 0.04 \
  --eye-to-hand-y-bias-m -0.065 \
  --eye-to-hand-z-bias-m 0.035
```

浏览器打开 `http://192.168.1.40:8080/`。单次或组合都先选 **LLM Hybrid**，再选按键，然后 Start。启动前会检查 Qwen 连通，60 秒只算连接，不算推理。连不上则手臂不动。

`--iface` 只选 DDS 网卡。机器人要已经开机并进入运控，`eth1` 上才有 `rt/lowstate`。`ip addr add` 重启后会丢，要保留时在 T40 上执行 `sudo ./scripts/set_orin_h2_ip.sh eth1`。

现场偏置按这次标定修改。上面三个数是当前这块面板的值。

更细的步骤和故障对照在 `Qwen启动与部署.md`，控制链路在 `控制链路.md`。
