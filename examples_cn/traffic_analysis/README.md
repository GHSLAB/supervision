# traffic analysis（交通分析）

## 👋 你好

本脚本使用 YOLOv8 目标检测方法与 ByteTrack（一种简单而有效的在线多目标跟踪方法）进行交通流分析。它借助 supervision 包完成跟踪、标注等多项任务。

https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900

## 💻 安装

- 克隆仓库并进入示例目录

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
    cd supervision/examples/traffic_analysis
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

- 下载 `traffic_analysis.pt` 和 `traffic_analysis.mov` 文件

    ```bash
    ./setup.sh
    ```

## 🛠️ 脚本参数

- ultralytics

    - `--source_weights_path`：必填。指定 YOLO 模型的权重文件路径，是目标检测流程的关键。该文件包含模型用于识别视频中目标的数据。

    - `--source_video_path`：必填。待分析的源视频文件路径，将在该视频上进行交通流分析。

    - `--target_video_path`（可选）：带标注结果视频的保存路径。若未指定，处理后的视频将实时显示而不保存。

    - `--confidence_threshold`（可选）：设置 YOLO 模型过滤检测结果的置信度阈值，默认为 `0.3`。该值决定模型需要在多大程度上确认视频中检测到的目标。

    - `--iou_threshold`（可选）：指定模型的 IOU（交并比）阈值，默认为 `0.7`。该参数用于管理目标检测精度，尤其是在区分不同目标时。

- inference

    - `--roboflow_api_key`（可选）：Roboflow 服务的 API 密钥。如未直接传入，脚本会尝试从 `ROBOFLOW_API_KEY` 环境变量读取。获取方式请参考[此指南](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)。

    - `--model_id`（可选）：指定要使用的 Roboflow 模型 ID，默认值为 `"vehicle-count-in-drone-video/6"`。

    - `--source_video_path`：必填。待分析的源视频文件路径，将在该视频上进行交通流分析。

    - `--target_video_path`（可选）：带标注结果视频的保存路径。若未指定，处理后的视频将实时显示而不保存。

    - `--confidence_threshold`（可选）：设置 YOLO 模型过滤检测结果的置信度阈值，默认为 `0.3`。该值决定模型需要在多大程度上确认视频中检测到的目标。

    - `--iou_threshold`（可选）：指定模型的 IOU（交并比）阈值，默认为 `0.7`。该参数用于管理目标检测精度，尤其是在区分不同目标时。

## ⚙️ 运行示例

- ultralytics

    ```bash
    python ultralytics_example.py \
        --source_weights_path data/traffic_analysis.pt \
        --source_video_path data/traffic_analysis.mov \
        --confidence_threshold 0.3 \
        --iou_threshold 0.5 \
        --target_video_path data/traffic_analysis_result.mov
    ```

- inference

    ```bash
    python inference_example.py \
        --roboflow_api_key "ROBOFLOW_API_KEY" \
        --source_video_path data/traffic_analysis.mov \
        --confidence_threshold 0.3 \
        --iou_threshold 0.5 \
        --target_video_path data/traffic_analysis_result.mov
    ```

## 🚗 增强版：`ultralytics_enhanced.py`

在原示例基础上叠加了三个能力：

- **车辆类型归一化**：从 `Detections.data["class_name"]` 映射到 `car / truck / bus / van / motorcycle / bicycle / other`，并按类型上色与打标签（`#ID car`），未知类别统一归为 `other`。可通过 `vehicle_type_aliases` 字典扩展。
- **车辆轨迹记录**：按 `tracker_id` 维护轨迹点序列（`frame_index` / `timestamp` / `xyxy`），在 `process_video` 结束后可调用 `get_trajectories()` 得到长表，适合画轨迹 / 热力图。
- **路口间车流统计表**：维护 `in_zone_id × out_zone_id × vehicle_type` 的去重计数（按 tracker id），并落盘为 CSV / JSON / Parquet：

  - `traffic_analysis_od_matrix.csv` — 长表（in/out/type/count）
  - `traffic_analysis_od_matrix.json` — 嵌套结构（含 `od_matrix_pivot`）
  - `traffic_analysis_od_matrix.parquet` — 列存版（需要 `pyarrow`）
  - `traffic_analysis_tracks_summary.csv` — 每辆车的汇总
  - `traffic_analysis_trajectories.csv` — 每条轨迹点

这些 DataFrame / 嵌套 JSON 都是下一步可视化（matplotlib、seaborn、Plotly、Streamlit 等）开箱即用的数据结构。

### 运行示例

```bash
python ultralytics_enhanced.py \
    --source_weights_path data/traffic_analysis.pt \
    --source_video_path data/traffic_analysis.mov \
    --target_video_path data/traffic_analysis_enhanced.mov \
    --output_dir data/enhanced_output \
    --confidence_threshold 0.3 \
    --iou_threshold 0.5
```

> 提示：示例权重 `traffic_analysis.pt` 仅检测 `vehicle` 一类，因此会全部归一化为 `other`。
> 换成多类车辆检测模型（如 `yolov8n.pt`）后，类型归一化与按类型统计会自动生效。

## © 许可证

本示例集成了两个主要组件，各自具有不同的许可证：

- ultralytics：本示例中使用的目标检测模型 YOLOv8 基于 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 发布。更多详细信息请参见此许可证。

- supervision：本示例中支撑区域分析的分析代码基于 Supervision 库，其使用 [MIT 许可证](https://github.com/roboflow/supervision/blob/develop/LICENSE.md) 发布。这意味着 Supervision 部分的代码完全开源，可在你的项目中自由使用。
