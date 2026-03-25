# football-stream-mvp

这个工程已经进入第二阶段：在可运行 MVP 基础上，加入了更接近生产方案的低延迟解码、轨迹稳定化、消息录制与离线评估。

## 当前能力

1. 视频接入：摄像头 / 文件 / RTSP / HLS
2. 检测：`YOLO (ultralytics)`
3. 跟踪：`ByteTrack (supervision)`
4. 坐标映射：线性映射
5. 轨迹后处理：EMA 平滑 + 短时 ReID（减少 ID 跳变）
6. 解码稳态：解码线程与处理线程分离、可配置缓冲丢帧策略
7. 数据闭环：JSONL 实时录制 + 离线评估脚本

## 目录结构

- `src/server.py`: 服务入口与 CLI 参数
- `src/runtime.py`: 组件工厂与运行时配置
- `src/video_reader.py`: 第二阶段解码模块（缓冲、丢帧策略、可选 pyav）
- `src/pipeline.py`: 核心处理流水线
- `src/postprocess.py`: 轨迹平滑与短时 ReID
- `src/recorder.py`: JSONL 录制器
- `src/eval_replay.py`: 离线回放评估
- `src/detector.py`: `YoloDetector`
- `src/tracker.py`: `ByteTrackAdapter`
- `src/projector.py`: `LinearProjector`
- `web/`: 前端可视化

## 安装

统一依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 启动示例

### 1) 默认主链路

```bash
source .venv/bin/activate
python -m src.server --source 0 --fps 15
```

### 2) 第二阶段主流链路（低延迟 + 稳定轨迹）

```bash
source .venv/bin/activate
python -m src.server \
  --source rtsp://user:pass@ip:554/stream \
  --fps 20 \
  --decode-backend pyav \
  --decode-buffer-size 6 \
  --decode-drop-policy drop_oldest \
  --prefer-latest-frame \
  --smooth-alpha 0.35 \
  --reid-ttl-ms 1500 \
  --reid-distance-px 90
```

### 3) 跨平台性能预设（NVIDIA / CPU）

自动识别本机加速器并应用推荐参数：

```bash
source .venv/bin/activate
python -m src.server \
  --source ./data/football.mp4 \
  --profile auto
```

手工指定（机器上有 NVIDIA GPU 时）：

```bash
source .venv/bin/activate
python -m src.server \
  --source ./data/football.mp4 \
  --profile nvidia
```

服务启动后访问：`http://127.0.0.1:8000`

## 录制与离线评估

### 录制

```bash
source .venv/bin/activate
python -m src.server \
  --source rtsp://user:pass@ip:554/stream \
  --record-path ./runs/session-001.jsonl
```

### 评估

```bash
source .venv/bin/activate
python -m src.eval_replay --path ./runs/session-001.jsonl
```

可输出 JSON：

```bash
source .venv/bin/activate
python -m src.eval_replay --path ./runs/session-001.jsonl --json
```

## 训练与数据扩充

新增训练入口：`python -m src.train`，支持：

1. 从图片目录收集样本
2. 从视频目录按间隔抽帧
3. 按 YOLO 标签自动对齐并划分 train/val
4. 直接调用 Ultralytics 继续训练

### 1) 仅准备数据（抽帧 + 划分，不训练）

```bash
source .venv/bin/activate
python -m src.train \
  --images-dir ./data/images \
  --videos-dir ./data/videos \
  --labels-dir ./data/labels \
  --dataset-dir ./datasets/soccer_train \
  --sample-every 30 \
  --prepare-only
```

### 2) 直接训练

```bash
source .venv/bin/activate
python -m src.train \
  --images-dir ./data/images \
  --videos-dir ./data/videos \
  --labels-dir ./data/labels \
  --dataset-dir ./datasets/soccer_train \
  --model ./models/yolo-v8-football-players-best.pt \
  --epochs 80 \
  --imgsz 960 \
  --batch 8 \
  --device auto
```

说明：

- 标签格式需为 YOLO txt（每张图一个同名 `.txt`）。
- 默认类别：`ball,goalkeeper,player,referee`，可用 `--class-names` 或 `--classes-file` 覆盖。
- 默认 `copy-mode=symlink`（节省磁盘空间），可切换 `--copy-mode copy`。

## 关键参数

- `--profile`: `auto | nvidia | cpu`
- `--yolo-device`: 例如 `cpu | cuda:0`
- `--yolo-half` / `--no-yolo-half`: 是否启用 FP16（仅 CUDA 生效）
- `--yolo-conf`: YOLO 置信度阈值
- `--yolo-imgsz`: YOLO 推理输入尺寸
- `--decode-backend`: `opencv | pyav`
- `--decode-buffer-size`: 解码缓冲区大小
- `--decode-drop-policy`: `drop_oldest | drop_newest`
- `--bytetrack-track-activation-threshold`: ByteTrack 激活阈值
- `--prefer-latest-frame` / `--no-prefer-latest-frame`: 是否优先处理最新帧
- `--smooth-alpha`: 轨迹平滑系数（0~1）
- `--reid-ttl-ms`: 短时 ReID 保留时间
- `--reid-distance-px`: 短时 ReID 像素距离阈值
- `--record-path`: JSONL 录制路径

## 输出消息（节选）

```json
{
  "schema_version": "1.5",
  "seq": 123,
  "frame_ts_ms": 1739271234567,
  "entities": [
    {"id": 7, "type": "player", "team": "A", "x": 31.44, "y": 22.11, "vx": 0.82, "vy": -0.14, "conf": 0.62}
  ],
  "meta": {
    "detector": "yolo",
    "tracker": "bytetrack",
    "projector": "linear",
    "decode_backend": "pyav",
    "decode_buffered": 1,
    "decode_dropped": 42,
    "detect_ms": 12.4,
    "track_ms": 1.1,
    "post_ms": 0.6,
    "process_ms": 14.1
  }
}
```
