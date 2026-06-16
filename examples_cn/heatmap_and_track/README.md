# heatmap and tracking（热力图与跟踪）

## 👋 你好

本脚本使用 YOLOv8 目标检测方法与 ByteTrack（一种简单而有效的在线多目标跟踪方法）进行热力图和跟踪分析。它借助 supervision 包完成绘制热力图标注、跟踪目标等多项任务。

## 💻 安装

- 克隆仓库并进入示例目录

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
    cd supervision/examples/heatmap_and_track
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

## 🛠️ 脚本参数

- `--source_weights_path`：必填。指定 YOLO 模型的权重文件路径。该文件包含目标检测所需的训练模型数据。
- `--source_video_path`（可选）：待分析的源视频文件路径，将在该视频上进行人群分析。若未指定，则默认使用 supervision 资源中的 `people-walking.mp4`。
- `--target_video_path`（可选）：带标注结果的目标 `.mp4` 视频保存路径。
- `--confidence_threshold`（可选）：设置 YOLO 模型过滤检测结果的置信度阈值，默认为 `0.3`。该值决定模型需要在多大程度上确认视频中检测到的目标。
- `--iou_threshold`（可选）：指定模型的 IOU（交并比）阈值，默认为 `0.7`。该参数用于管理目标检测精度，尤其是在区分不同目标时。
- `--heatmap_alpha`（可选）：叠加遮罩的不透明度，取值范围 0 到 1。
- `--radius`（可选）：热力圆点的半径。
- `--track_activation_threshold`（可选）：激活跟踪所需的检测置信度阈值。
- `--track_seconds`（可选）：当跟踪目标丢失时，缓存跟踪信息的秒数。
- `--minimum_matching_threshold`（可选）：将跟踪与检测结果进行匹配时的阈值。

## ⚙️ 运行

```bash
python script.py \
    --source_weights_path weight.pt \
    --source_video_path  input_video.mp4 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.5 \
    --target_video_path  output_video.mp4
```

## © 许可证

本示例集成了两个主要组件，各自具有不同的许可证：

- ultralytics：本示例中使用的目标检测模型 YOLOv8 基于 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 发布。更多详细信息请参见此许可证。

- supervision：本示例中支撑区域分析的分析代码基于 Supervision 库，其使用 [MIT 许可证](https://github.com/roboflow/supervision/blob/develop/LICENSE.md) 发布。这意味着 Supervision 部分的代码完全开源，可在你的项目中自由使用。
