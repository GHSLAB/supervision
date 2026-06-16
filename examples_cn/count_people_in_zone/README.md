# count people in zone（区域人数统计）

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-detect-and-count-objects-in-polygon-zone.ipynb) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=l_kf9CfZ_8M)

## 👋 你好

本示例是一个视频分析工具，用于统计并高亮视频中特定区域内出现的目标。每个区域及其内部目标使用不同颜色进行标记，便于直观地观察和统计每个区域中的目标数量。该工具可以将处理后的视频保存下来，也可以实时显示在屏幕上。

https://github.com/roboflow/supervision/assets/26109316/f84db7b5-79e2-4142-a1da-64daa43ce667

## 💻 安装

- 克隆仓库并进入示例目录

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
    cd supervision/examples/count_people_in_zone
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

    - `--source_weights_path`（可选）：YOLO 模型的权重文件路径。若未指定，默认值为 `"yolov8x.pt"`。

    - `--zone_configuration_path`：包含区域配置的 JSON 文件路径。该文件定义了视频中用于目标统计的多边形区域。

    - `--source_video_path`：待分析的源视频文件路径。

    - `--target_video_path`（可选）：带标注结果视频的保存路径。若未提供，处理后的视频将实时显示。

    - `--confidence_threshold`（可选）：YOLO 模型过滤检测结果的置信度阈值，默认为 `0.3`。

    - `--iou_threshold`（可选）：模型的 IOU（交并比）阈值，默认为 `0.7`。

- inference

    - `--roboflow_api_key`（可选）：Roboflow 服务的 API 密钥。如未直接传入，脚本会尝试从 `ROBOFLOW_API_KEY` 环境变量读取。获取方式请参考[此指南](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)。

    - `--model_id`（可选）：指定要使用的 Roboflow 模型 ID，默认值为 `"yolov8x-1280"`。

    - `--zone_configuration_path`：包含区域配置的 JSON 文件路径。该文件定义了视频中用于目标统计的多边形区域。

    - `--source_video_path`：待分析的源视频文件路径。

    - `--target_video_path`（可选）：带标注结果视频的保存路径。若未提供，处理后的视频将实时显示。

    - `--confidence_threshold`（可选）：YOLO 模型过滤检测结果的置信度阈值，默认为 `0.3`。

    - `--iou_threshold`（可选）：模型的 IOU（交并比）阈值，默认为 `0.7`。

## 📌 区域配置

- `horizontal-zone-config.json`：以水平方向划分多个区域。
- `multi-zone-config.json`：配置多个具有自定义形状和位置的多边形区域。
- `quarters-zone-config.json`：将画面等分为四个区域。
- `vertical-zone-config.json`：将画面按等宽划分为多个垂直区域。

## ⚙️ 运行示例

- ultralytics

    ```bash
    python ultralytics_example.py \
        --zone_configuration_path data/multi-zone-config.json \
        --source_video_path data/market-square.mp4 \
        --confidence_threshold 0.3 \
        --iou_threshold 0.5
    ```

- inference

    ```bash
    python inference_example.py \
        --roboflow_api_key "ROBOFLOW_API_KEY" \
        --zone_configuration_path data/multi-zone-config.json \
        --source_video_path data/market-square.mp4 \
        --confidence_threshold 0.3 \
        --iou_threshold 0.5
    ```

## © 许可证

本示例集成了两个主要组件，各自具有不同的许可证：

- ultralytics：本示例中使用的目标检测模型 YOLOv8 基于 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 发布。更多详细信息请参见此许可证。

- supervision：本示例中支撑区域分析的分析代码基于 Supervision 库，其使用 [MIT 许可证](https://github.com/roboflow/supervision/blob/develop/LICENSE.md) 发布。这意味着 Supervision 部分的代码完全开源，可在你的项目中自由使用。
