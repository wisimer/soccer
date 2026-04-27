

### 1) 默认主链路

```bash
source .venv/bin/activate
python -m src.server --source 0 --fps 15
```

电脑需要配置摄像头，否则会报错：
```
WARNING src.video_reader: failed to open source with OpenCV: 0
[ERROR:0@175.817] global obsensor_uvc_stream_channel.cpp:158 cv::obsensor::getStreamChannelGroup Camera index out of range
```

### 3) 跨平台性能预设（NVIDIA / CPU）

手工指定（机器上有 NVIDIA GPU 时）：

```bash
source .venv/bin/activate
python -m src.server \
  --source ./data/football.mp4 \
  --profile nvidia
```

报错：
```
python -m src.server    --source ./data/HBTFZwMdcCw.mp4   --profile nvidia 启动之后报错 FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instea ，并且 http://127.0.0.1:8000无法访问
```

文档和代码中的端口不一致，修改访问端口为 8765:  
http://127.0.0.1:8765/


### 录制

```bash
source .venv/bin/activate
python -m src.server \
  --source rtsp://user:pass@ip:554/stream \
  --record-path ./runs/session-001.jsonl
```
录制的作用：

这个“录制”指的是在你跑 `src.server` 实时处理视频/RTSP 的同时，把**每一帧的推理/跟踪结果按行写成 JSONL 文件**（一行一个 JSON），方便后续离线分析、复现问题、做性能对比。

- 开关是 `--record-path ./runs/session-001.jsonl`（见 [server.py:parse_args](file:///d:/GCZM/soccer/src/server.py#L1024-L1047)），启用后会创建 [JsonlRecorder](file:///d:/GCZM/soccer/src/recorder.py#L1-L26) 并在流水线每处理完一帧就 `write()` 一行（见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L365-L412)）。
- 录下来的内容不仅有 `entities`（球员/球的 ID、位置等），还有 `detections`、`tracks`、`frame` 信息，以及 `meta` 里的耗时与解码统计（`detect_ms/track_ms/post_ms/process_ms/decode_dropped` 等，见 [pipeline.py:_build_payload/_build_meta](file:///d:/GCZM/soccer/src/pipeline.py#L174-L226)）。
- 主要用途：
  - 给你刚才问的“评估”用：`python -m src.eval_replay --path ...` 会读这个 JSONL 来汇总性能与稳定性指标（见 [eval_replay.py](file:///d:/GCZM/soccer/src/eval_replay.py#L23-L104)）。
  - 复现/排查：线上某段 RTSP 出问题时，保留当时的输出结果，之后不用再连摄像头也能分析（至少能看输出、耗时、丢帧、ID 行为）。
  - 回归对比：改了模型或参数后，用同一段录制文件对比 `process_ms_p95`、`decode_dropped_max`、`unique_ids/id_births` 等是否变差。

README 里也有一段“输出消息（节选）”展示了录制文件每行大概长什么样（见 [README.md:L168-L195](file:///d:/GCZM/soccer/README.md#L168-L195)）。



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
评估的作用：


这里的“评估”指的是**对录制出来的回放 JSONL（`--record-path` 生成）做离线统计汇总**，用来快速看这次运行的**吞吐/延迟、丢帧、跟踪 ID 稳定性**等健康指标——不是检测精度（mAP/IoU）那种“算法准确率评估”。

- 评估脚本入口是 [eval_replay.py](file:///d:/GCZM/soccer/src/eval_replay.py#L1-L122)，它逐行读取 `--path` 指定的 `.jsonl`，每行代表一帧的处理结果。
- 这些 JSONL 行内容来自服务端流水线写盘：在 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L174-L226) 里组装 payload，并通过 [recorder.py](file:///d:/GCZM/soccer/src/recorder.py#L1-L26) 的 `JsonlRecorder.write()` 写入。

它最终输出一份 summary（`--json` 时输出 JSON），主要指标含义如下（见 [eval_replay.py](file:///d:/GCZM/soccer/src/eval_replay.py#L23-L104)）：

- `frames`：回放里总帧数（JSONL 行数）。
- `duration_s`：用 `frame_ts_ms` 的首尾差估算的时长（秒）。
- `effective_fps`：`frames / duration_s`，等价于这段回放的“有效帧率”（更像输入时间轴上的帧密度）。
- `avg_entities_per_frame`：每帧平均实体数（entities 数量均值）。
- `unique_ids` / `id_births`：出现过的跟踪 ID 数、以及“首次出现的新 ID”数量（用来粗看 ID 是否频繁重生/抖动）。
- `process_ms_avg` / `process_ms_p95`：单帧总耗时的平均值与 P95（`process_ms = detect_ms + track_ms + post_ms`，来自 [pipeline.py:_build_meta](file:///d:/GCZM/soccer/src/pipeline.py#L202-L226)）。
- `detect_ms_avg` / `track_ms_avg` / `post_ms_avg`：检测/跟踪/后处理分阶段平均耗时。
- `decode_dropped_max`：解码累计丢帧计数的最大值（来自 `decode_stats["dropped_frames"]`，写到 `meta.decode_dropped`，见 [pipeline.py:_build_meta](file:///d:/GCZM/soccer/src/pipeline.py#L202-L226) 和 [video_reader.py:stats](file:///d:/GCZM/soccer/src/video_reader.py#L111-L122)）。

所以，这个“评估”更像是**离线性能/稳定性体检**：你改了模型、解码参数、跟踪参数后，用同一份 session 回放跑一下，就能对比 FPS、P95 延迟、丢帧、ID 变化是否变差/变好。


---

离线视频目标检测核心流程


          
下面按“服务启动 → 解码 → 目标检测 → 跟踪/后处理 → 组包 → WebSocket 推送”的真实调用链，把后端流程串起来（并标出关键代码位置），最后给出流程图。

---

**1) 服务启动后发生了什么（创建流水线 + 启动两条后台任务）**

入口是 `python -m src.server ...`，最终走到 [server.py:main](file:///d:/GCZM/soccer/src/server.py#L1040-L1071)：

- 解析 CLI（`--source ./data/football.mp4 --profile nvidia` 等）→ 组装运行时配置 `RuntimeSettings`（CLI/env/YAML/profile 合并）  
  见 [server.py:build_runtime_settings](file:///d:/GCZM/soccer/src/server.py#L476-L736) + [runtime.py:profile_defaults/resolve_effective_profile](file:///d:/GCZM/soccer/src/runtime.py#L240-L306)
- 创建 FastAPI app，并在 `lifespan()` 里启动处理链路：  
  见 [server.py:create_app/lifespan](file:///d:/GCZM/soccer/src/server.py#L193-L244)

`lifespan()` 里做的关键事：

- `build_detector / build_tracker / build_projector / build_decode_config / build_postprocessor / build_recorder`  
  见 [runtime.py:build_*](file:///d:/GCZM/soccer/src/runtime.py#L308-L470)
- 创建 `StreamProcessor(...)`（核心流水线对象）并启动两个 asyncio task：  
  - `processor.run(queue)`：不断产出每帧（或每 batch）的检测/跟踪结果 payload  
  - `_forward_loop(queue, hub)`：把 payload 广播给所有 WebSocket 客户端  
  见 [server.py:create_app/lifespan](file:///d:/GCZM/soccer/src/server.py#L199-L241) + [server.py:_forward_loop](file:///d:/GCZM/soccer/src/server.py#L86-L90)

前端连上 WebSocket `/ws` 后，后端只是把连接加入 `ConnectionHub`，并不需要前端发任何业务消息；前端定时 `ping` 只是为了保活/探测断开：  
见 [server.py:/ws](file:///d:/GCZM/soccer/src/server.py#L331-L343) + [web/app.js:connect](file:///d:/GCZM/soccer/web/app.js#L808-L867)

---

**2) 目标检测主流程（StreamProcessor.run）**

核心循环在 [pipeline.py:StreamProcessor.run](file:///d:/GCZM/soccer/src/pipeline.py#L332-L426)。它做的事情可以拆成 7 步：

1. **启动解码线程**
   - `self.reader.start()` 启动 `BufferedVideoReader` 的后台线程持续解码并写入 buffer  
     见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L338-L339) + [video_reader.py:BufferedVideoReader.start](file:///d:/GCZM/soccer/src/video_reader.py#L55-L63)

2. **按 target_fps 节流**
   - 用 `min_interval_s = 1/target_fps` 控制处理侧节奏（不是解码侧）  
     见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L335-L346)

3. **从解码缓冲中取 frame（支持 batch）**
   - `detect_batch_size = detector.batch_size`，然后 `reader.pop_frames(batch_size, prefer_latest_frame)`  
   - `prefer_latest_frame=True` 时会把 buffer 清空，只拿最新 N 帧，降低端到端延迟  
     见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L335-L351) + [video_reader.py:pop_frames](file:///d:/GCZM/soccer/src/video_reader.py#L81-L110)

4. **目标检测（YOLO，支持 batched inference）**
   - `detections_batch = detector.detect_many(frames)`  
     见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L355-L360)

   检测器默认是 `YoloDetector` 或 `HybridDetector(球员模型 + 足球模型)`：  
   - 构建逻辑见 [runtime.py:build_detector](file:///d:/GCZM/soccer/src/runtime.py#L308-L368)
   - YOLO 推理见 [detector.py:YoloDetector.detect_many](file:///d:/GCZM/soccer/src/detector.py#L400-L408)

   这里 `--profile nvidia` 的影响主要是把默认参数切到更偏 GPU/低延迟（例如 `yolo_device=cuda:0, yolo_half=True, yolo_imgsz=768, yolo_batch_size=2, decode_backend=pyav`）：  
   见 [runtime.py:profile_defaults](file:///d:/GCZM/soccer/src/runtime.py#L254-L305)

5. **跟踪（ByteTrack + 全局运动补偿 GMC）**
   - 对每一帧：`raw_tracks = tracker.update(detections, ts_ms, frame)`  
   - `ByteTrackAdapter` 内部会：
     - 把检测框转成 supervision 的 `Detections`
     - 用光流估计相机运动 `(dx,dy)` 并平移 track（GMC）
     - 更新 ByteTrack，并把结果转成自定义 `TrackState`（包含 vx/vy、missed_frames 等）  
     见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L373-L376) + [tracker.py:ByteTrackAdapter.update](file:///d:/GCZM/soccer/src/tracker.py#L374-L482) + [tracker.py:_SparseOptFlowMotionCompensator](file:///d:/GCZM/soccer/src/tracker.py#L38-L115)

6. **轨迹后处理（平滑 + 短时 ReID，减少 ID 跳变）**
   - `tracks = postprocessor.update(raw_tracks, ts_ms)`  
   - 主要做两件事：
     - EMA/自适应 alpha 平滑位置与速度（球/人参数不同）
     - 如果 ByteTrack 出现短时断裂，尝试把新 raw_id 重新绑定到旧的 stable_id（TTL + 距离门限）  
     见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L373-L377) + [postprocess.py:TrackPostProcessor](file:///d:/GCZM/soccer/src/postprocess.py#L28-L281)

7. **序列化 + 组包 + 推送**
   - `detections` → `payload.detections`（像素框）  
     见 [pipeline.py:_serialize_detection](file:///d:/GCZM/soccer/src/pipeline.py#L94-L104)
   - `tracks` → 同时生成：
     - `payload.tracks`：跟踪框（像素 xywh）
     - `payload.entities`：映射到球场坐标（米制 x/y + vx/vy）  
       映射使用 `LinearProjector`：`x_px/frame_width * 105m`，`y_px/frame_height * 68m`  
       见 [pipeline.py:_serialize_track](file:///d:/GCZM/soccer/src/pipeline.py#L105-L135) + [projector.py:LinearProjector](file:///d:/GCZM/soccer/src/projector.py#L26-L50)
   - 还会做一个“球短时丢失的预测注入”（连续性兜底）：  
     如果当前帧没有 ball entity，但最近窗口内见过球，就用上次球的速度外推一个 predicted ball（并衰减 conf）  
     见 [pipeline.py:_apply_ball_continuity](file:///d:/GCZM/soccer/src/pipeline.py#L281-L331)
   - 生成 `meta`（性能与解码统计）与 `game_payload`（事件/比分等），最后合成总 payload  
     见 [pipeline.py:_build_meta](file:///d:/GCZM/soccer/src/pipeline.py#L202-L227) + [pipeline.py:_build_payload](file:///d:/GCZM/soccer/src/pipeline.py#L174-L201) + [game_engine.py:update_from_frame](file:///d:/GCZM/soccer/src/game_engine.py#L98-L133)
   - 可选录制：`JsonlRecorder.write(payload)`  
     见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L414-L416)
   - 把 payload 放入 queue；如果 queue 满了（maxsize=5），先丢掉最旧的一条，保证前端拿到“最新帧”  
     见 [pipeline.py](file:///d:/GCZM/soccer/src/pipeline.py#L417-L422)

`_forward_loop` 会从 queue 取 payload，然后 `hub.broadcast(payload)` 广播给所有 ws 客户端：  
见 [server.py:_forward_loop](file:///d:/GCZM/soccer/src/server.py#L86-L90) + [server.py:ConnectionHub.broadcast](file:///d:/GCZM/soccer/src/server.py#L53-L68)


几个关键环节：
1、创建 StreamProcessor(...)（核心流水线对象）并启动两个 asyncio task：
processor.run(queue)：不断产出每帧（或每 batch）的检测/跟踪结果 payload
_forward_loop(queue, hub)：把 payload 广播给所有 WebSocket 客户端
2、启动解码线程
self.reader.start() 启动 BufferedVideoReader 的后台线程持续解码并写入 buffer
3、目标检测（YOLO，支持 batched inference）
多帧同时检测
4、跟踪（ByteTrack + 全局运动补偿 GMC）
5、序列化 + 组包 + 推送

---

**3) YOLO 检测器内部细节（你在前端看到的 det/tracks 是怎么来的）**

`YoloDetector`（[detector.py](file:///d:/GCZM/soccer/src/detector.py)）的关键点：

- **后端选择**
  - 默认走 `ultralytics.YOLO`；如果模型名里包含 `yolov5` 或显式指定，则走 yolov5 runner  
    见 [detector.py:_resolve_backend](file:///d:/GCZM/soccer/src/detector.py#L478-L486)
- **输出转 Detection**
  - 从 `xyxy/conf/cls` 构建 `Detection(kind,x,y,w,h,confidence,team)`  
  - `kind` 推断：类名包含 `ball/football`→ball；包含 `person/player/referee...`→player；否则忽略  
    见 [detector.py:_resolve_kind](file:///d:/GCZM/soccer/src/detector.py#L516-L538)
- **球过滤**
  - 对 ball bbox 做面积/长宽比门限（避免把噪声当球）  
    见 [detector.py:_accept_ball_bbox](file:///d:/GCZM/soccer/src/detector.py#L471-L477)
  - `ball_max_detections` 限制每帧最多保留几个球（按 conf 排序截断）  
    见 [detector.py](file:///d:/GCZM/soccer/src/detector.py#L459-L468)
- **队伍识别（player 的 team 字段）**
  - 把 player 框上半身 ROI 转 HSV 做颜色判别：auto 模式用 hue 粗分 A/B；manual 模式按配置颜色比例判别  
    见 [detector.py:infer_team_from_bbox](file:///d:/GCZM/soccer/src/detector.py#L314-L352)

---

**4) WebSocket 消息长什么样（前端如何渲染）**

后端每次推送的是 `payload: dict`，骨架由 [pipeline.py:_build_payload](file:///d:/GCZM/soccer/src/pipeline.py#L174-L201) 决定，前端在 [web/app.js message handler](file:///d:/GCZM/soccer/web/app.js#L840-L867) 里读取：

- `detections[]`：原始检测框（像素坐标）
  - `{type, team, x, y, w, h, conf}`
- `tracks[]`：跟踪框（像素坐标，ID 更稳定）
  - `{id, type, team, x, y, w, h, conf, predicted?}`
- `entities[]`：球场坐标（米制），用于右侧“球场视角”渲染/逻辑
  - `{id, type, team, x, y, vx, vy, conf, predicted?}`
- `frame`：宽高、帧序号、时间戳（前端用 `source_ts_ms` 做视频对齐）
- `meta`：检测/跟踪/后处理耗时、解码丢帧数等（页面底部显示）

---

**5) 流程图（后端端到端）**

```mermaid
flowchart TD
  A[启动 python -m src.server\n--source ... --profile nvidia] --> B[解析参数/合并配置\nbuild_runtime_settings]
  B --> C[FastAPI create_app + lifespan]
  C --> D[构建组件\nbuild_detector/tracker/projector\nbuild_decode_config/postprocessor/recorder]
  D --> E[创建 StreamProcessor]
  E --> F1[Task1: StreamProcessor.run(queue)]
  E --> F2[Task2: forward_loop(queue, hub)]
  C --> G[WebSocket /ws 接入\nhub.connect]

  subgraph Decode[解码线程 BufferedVideoReader]
    H1[reader.start()] --> H2[decode_loop\npyav 或 opencv]
    H2 --> H3[push FramePacket 到 buffer\n满了按 drop_policy 丢帧]
  end

  F1 --> I[按 target_fps 节流]
  I --> J[pop_frames(batch_size, prefer_latest)\n从 buffer 取帧/可清空取最新]
  J --> K[detector.detect_many(frames)\nYOLO 推理 + team/ball过滤]
  K --> L[tracker.update(detections, ts, frame)\nByteTrack + 光流GMC]
  L --> M[postprocessor.update(raw_tracks)\n平滑 + ReID 稳定ID]
  M --> N[projector.to_pitch + serialize\nentities/tracks/detections]
  N --> O[ball continuity 兜底\n短时丢球预测]
  O --> P[game_engine.update_from_frame\n生成 game_* 字段]
  P --> Q[build payload + meta\n可选 recorder.write]
  Q --> R{queue 满?}
  R -- 是 --> S[丢掉最旧一条]
  R -- 否 --> T[queue.put(payload)]
  S --> T
  T --> F2
  F2 --> U[hub.broadcast(payload)\n推送给所有 ws 客户端]
  U --> V[前端渲染\n视频overlay + 球场视角]
```

如果你还想把“检测（YOLO）输出的类别/ID 与前端画框颜色、球员阵营（A/B）显示规则”也一并对上，我可以继续按 [web/app.js](file:///d:/GCZM/soccer/web/app.js) 里 `drawVideoOverlay()/drawPitchScene()` 的绘制逻辑，把每个字段如何映射到 UI 讲清楚。



