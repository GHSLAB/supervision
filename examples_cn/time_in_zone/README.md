# time in zone（区域停留时长）

[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=hAWpsIuem10)

## 👋 你好

本示例演示如何利用计算机视觉分析视频中目标在预定义区域内停留或等待的时长，非常适合零售分析或交通管理等应用场景。

https://github.com/roboflow/supervision/assets/26109316/d051cc8a-dd15-41d4-aa36-d38b86334c39

## 💻 安装

- 克隆仓库并进入示例目录

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
    cd supervision/examples/time_in_zone
    ```

- （可选）创建并激活 Python 虚拟环境

    ```bash
    uv venv
    source .venv/bin/activate
    ```

- 安装所需依赖

    ```bash
    uv pip install -r requirements.txt
    ```

## 🛠 脚本

### `download_from_youtube`

此脚本用于从 YouTube 下载视频。

- `--url`：待下载 YouTube 视频的完整 URL。
- `--output_path`（可选）：视频保存的目标目录。
- `--file_name`（可选）：保存的视频文件名。

```bash
python scripts/download_from_youtube.py \
    --url "https://www.youtube.com/watch?v=-8zyEwAa50Q" \
    --output_path "data/checkout" \
    --file_name "video.mp4"
```

```bash
python scripts/download_from_youtube.py \
    --url "https://www.youtube.com/watch?v=MNn9qKG2UFI" \
    --output_path "data/traffic" \
    --file_name "video.mp4"
```

### `stream_from_file`

此脚本用于将目录下的视频文件作为流推送，是本地模拟实时视频流的好方式。视频会以循环方式在 `rtsp://localhost:8554/live0.stream` URL 上推送。此脚本需要安装 Docker。

- `--video_directory`：包含待推送视频文件的目录。
- `--number_of_streams`：要推送的视频文件数量。

```bash
python scripts/stream_from_file.py \
    --video_directory "data/checkout" \
    --number_of_streams 1
```

```bash
python scripts/stream_from_file.py \
    --video_directory "data/traffic" \
    --number_of_streams 1
```

### `draw_zones`

如果你想在自己的视频上测试停留时长分析，可以使用此脚本设计自定义区域，并将结果保存为 JSON 文件。脚本会打开一个窗口，你可以在源图像或视频上绘制多边形，绘制完成的多边形将保存为 JSON 文件。

- `--source_path`：用于绘制多边形的源图像或视频文件路径。
- `--zone_configuration_path`：多边形标注结果保存为 JSON 文件的路径。
- `enter` — 完成当前多边形的绘制。
- `escape` — 取消当前多边形的绘制。
- `q` — 退出绘制窗口。
- `s` — 将区域配置保存为 JSON 文件。

```bash
python scripts/draw_zones.py \
    --source_path "data/checkout/video.mp4" \
    --zone_configuration_path "data/checkout/config.json"
```

```bash
python scripts/draw_zones.py \
    --source_path "data/traffic/video.mp4" \
    --zone_configuration_path "data/traffic/config.json"
```

https://github.com/roboflow/supervision/assets/26109316/9d514c9e-2a61-418b-ae49-6ac1ad6ae5ac

## 🎬 视频与流处理

### `inference_file_example`

使用 Roboflow Inference 模型对视频文件运行目标检测的脚本。

- `--zone_configuration_path`：区域配置 JSON 文件路径。
- `--source_video_path`：源视频文件路径。
- `--model_id`：Roboflow 模型 ID。
- `--classes`：要跟踪的目标类别 ID 列表。若为空，则跟踪所有类别。
- `--confidence_threshold`：检测的置信度阈值（`0` 到 `1`），默认为 `0.3`。
- `--iou_threshold`：非极大值抑制的 IOU 阈值，默认为 `0.7`。

```bash
python inference_file_example.py \
    --zone_configuration_path "data/checkout/config.json" \
    --source_video_path "data/checkout/video.mp4" \
    --model_id "rfdetr-medium" \
    --classes "[0]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7 \
    --roboflow_api_key "ROBOFLOWS_API_KEY"
```

https://github.com/roboflow/supervision/assets/26109316/d051cc8a-dd15-41d4-aa36-d38b86334c39

```bash
python inference_file_example.py \
    --zone_configuration_path "data/traffic/config.json" \
    --source_video_path "data/traffic/video.mp4" \
    --model_id "rfdetr-medium" \
    --classes "[2, 5, 6, 7]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7 \
    --roboflow_api_key "ROBOFLOWS_API_KEY"
```

https://github.com/roboflow/supervision/assets/26109316/5ec896d7-4b39-4426-8979-11e71666878b

### `inference_stream_example`

使用 Roboflow Inference 模型对 RTSP 流运行目标检测的脚本。

- `--zone_configuration_path`：区域配置 JSON 文件路径。
- `--rtsp_url`：视频流的完整 RTSP URL。
- `--model_id`：Roboflow 模型 ID。
- `--classes`：要跟踪的目标类别 ID 列表。若为空，则跟踪所有类别。
- `--confidence_threshold`：检测的置信度阈值（`0` 到 `1`），默认为 `0.3`。
- `--iou_threshold`：非极大值抑制的 IOU 阈值，默认为 `0.7`。

```bash
python inference_stream_example.py \
    --zone_configuration_path "data/checkout/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --model_id "rfdetr-medium" \
    --classes "[0]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

```bash
python inference_stream_example.py \
    --zone_configuration_path "data/traffic/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --model_id "rfdetr-medium" \
    --classes "[2, 5, 6, 7]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

### `rfdeter_file_example`

使用 RF-DETR 模型对视频文件运行目标检测的脚本。

- `--zone_configuration_path`：区域配置 JSON 文件路径。
- `--source_video_path`：源视频文件路径。
- `--model_size`：RF-DETR 模型尺寸（'nano'、'small'、'medium'、'base' 或 'large'），默认为 'medium'。
- `--device`：计算设备（'cpu'、'mps' 或 'cuda'），默认为 'cpu'。
- `--classes`：要跟踪的目标类别 ID 列表。若为空，则跟踪所有类别。
- `--confidence_threshold`：检测的置信度阈值（`0` 到 `1`），默认为 `0.3`。
- `--iou_threshold`：非极大值抑制的 IOU 阈值，默认为 `0.7`。
- `--resolution`：模型输入分辨率，默认为 `640`。

```bash
python rfdetr_file_example.py \
    --zone_configuration_path "data/checkout/config.json" \
    --source_video_path "data/checkout/video.mp4" \
    --model_size "medium" \
    --device="cpu" \
    --classes "[1]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7 \
    --resolution 640
```

```bash
python rfdetr_file_example.py \
    --zone_configuration_path "data/traffic/config.json" \
    --source_video_path "data/traffic/video.mp4" \
    --model_size "medium" \
    --device="cpu" \
    --classes "[3, 6, 7, 8]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7 \
    --resolution 640
```

### `rfdeter_stream_example`

使用 RF-DETR 模型对 RTSP 流运行目标检测的脚本。

- `--zone_configuration_path`：定义多边形区域的区域配置 JSON 文件路径。
- `--rtsp_url`：实时视频流的完整 RTSP URL。
- `--model_size`：要加载的 RF-DETR 主干尺寸，可选 'nano'、'small'、'medium'、'base' 或 'large'（默认 'medium'）。
- `--device`：运行模型的计算设备（'cpu'、'mps' 或 'cuda'；默认 'cpu'）。
- `--classes`：要跟踪的目标类别 ID 列表（以空格分隔）。留空则跟踪所有类别。
- `--confidence_threshold`：保留检测所需的最低置信度，取值范围 0-1（默认 0.3）。
- `--iou_threshold`：非极大值抑制中的 IOU 阈值（默认 0.7）。
- `--resolution`：输入到模型的最短边分辨率，脚本会将其四舍五入到最近的有效倍数（默认 640）。

```bash
python rfdetr_stream_example.py \
    --zone_configuration_path "data/checkout/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --model_size "medium" \
    --device "cpu" \
    --classes "[1]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7 \
    --resolution 640
```

```bash
python rfdetr_stream_example.py \
    --zone_configuration_path "data/traffic/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --model_size "medium" \
    --device "cpu" \
    --classes "[3, 6, 7, 8]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7 \
    --resolution 640
```

### `ultralytics_file_example`

使用 Ultralytics YOLOv8 模型对视频文件运行目标检测的脚本。

- `--zone_configuration_path`：区域配置 JSON 文件路径。
- `--source_video_path`：源视频文件路径。
- `--weights`：模型权重文件路径，默认为 `'yolov8s.pt'`。
- `--device`：计算设备（`'cpu'`、`'mps'` 或 `'cuda'`），默认为 `'cpu'`。
- `--classes`：要跟踪的目标类别 ID 列表。若为空，则跟踪所有类别。
- `--confidence_threshold`：检测的置信度阈值（`0` 到 `1`），默认为 `0.3`。
- `--iou_threshold`：非极大值抑制的 IOU 阈值，默认为 `0.7`。

```bash
python ultralytics_file_example.py \
    --zone_configuration_path "data/checkout/config.json" \
    --source_video_path "data/checkout/video.mp4" \
    --weights "yolov8x.pt" \
    --device "cpu" \
    --classes "[0]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

```bash
python ultralytics_file_example.py \
    --zone_configuration_path "data/traffic/config.json" \
    --source_video_path "data/traffic/video.mp4" \
    --weights "yolov8x.pt" \
    --device "cpu" \
    --classes "[2, 5, 6, 7]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

### `ultralytics_stream_example`

使用 Ultralytics YOLOv8 模型对视频流运行目标检测的脚本。

- `--zone_configuration_path`：区域配置 JSON 文件路径。
- `--rtsp_url`：视频流的完整 RTSP URL。
- `--weights`：模型权重文件路径，默认为 `'yolov8s.pt'`。
- `--device`：计算设备（`'cpu'`、`'mps'` 或 `'cuda'`），默认为 `'cpu'`。
- `--classes`：要跟踪的目标类别 ID 列表。若为空，则跟踪所有类别。
- `--confidence_threshold`：检测的置信度阈值（`0` 到 `1`），默认为 `0.3`。
- `--iou_threshold`：非极大值抑制的 IOU 阈值，默认为 `0.7`。

```bash
python ultralytics_stream_example.py \
    --zone_configuration_path "data/checkout/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --weights "yolov8x.pt" \
    --device "cpu" \
    --classes "[0]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

```bash
python ultralytics_stream_example.py \
    --zone_configuration_path "data/traffic/config.json" \
    --rtsp_url "rtsp://localhost:8554/live0.stream" \
    --weights "yolov8x.pt" \
    --device "cpu" \
    --classes "[2, 5, 6, 7]" \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

</details>

## © 许可证

本示例集成了两个主要组件，各自具有不同的许可证：

- ultralytics：本示例中使用的目标检测模型 YOLOv8 基于 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 发布。更多详细信息请参见此许可证。

- supervision：本示例中支撑区域分析的分析代码基于 Supervision 库，其使用 [MIT 许可证](https://github.com/roboflow/supervision/blob/develop/LICENSE.md) 发布。这意味着 Supervision 部分的代码完全开源，可在你的项目中自由使用。
