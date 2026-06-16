# tracking（目标跟踪）

## 👋 你好

本脚本使用 YOLOv8 进行目标检测，并结合 Supervision 完成跟踪与标注，提供视频处理能力。

## 💻 安装

- 克隆仓库并进入示例目录

    ```bash
    git clone --depth 1 -b develop https://github.com/roboflow/supervision.git
    cd supervision/examples/tracking
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

- ultralytics

    - `--source_weights_path`：必填。指定 YOLO 模型的权重文件路径，是目标检测流程的关键。该文件包含模型用于识别视频中目标的数据。

    - `--source_video_path`：必填。待处理的源视频文件路径，将在该视频上进行目标检测与标注。

    - `--target_video_path`：必填。处理后视频的保存路径，输出文件将包含标注结果。

    - `--confidence_threshold`（可选）：设置模型识别视频中目标的置信度阈值，默认为 `0.3`。阈值越高，模型对目标的筛选越严格；阈值越低，则更容易识别为有效目标。

    - `--iou_threshold`（可选）：指定模型的 IOU（交并比）阈值，默认为 `0.7`。该参数用于区分不同目标，在拥挤场景中尤为关键。

- inference

    - `--roboflow_api_key`（可选）：Roboflow 服务的 API 密钥。如未直接传入，脚本会尝试从 `ROBOFLOW_API_KEY` 环境变量读取。获取方式请参考[此指南](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)。

    - `--model_id`（可选）：指定要使用的 Roboflow 模型 ID，默认值为 `"yolov8x-1280"`。

    - `--source_video_path`：必填。待处理的源视频文件路径，将在该视频上进行目标检测与标注。

    - `--target_video_path`：必填。处理后视频的保存路径，输出文件将包含标注结果。

    - `--confidence_threshold`（可选）：设置模型识别视频中目标的置信度阈值，默认为 `0.3`。阈值越高，模型对目标的筛选越严格；阈值越低，则更容易识别为有效目标。

    - `--iou_threshold`（可选）：指定模型的 IOU（交并比）阈值，默认为 `0.7`。该参数用于区分不同目标，在拥挤场景中尤为关键。

## ⚙️ 运行示例

- inference

    ```bash
    python inference_example.py \
        --roboflow_api_key "ROBOFLOW_API_KEY" \
        --source_video_path input.mp4 \
        --target_video_path tracking_result.mp4
    ```

- ultralytics

    ```bash
    python ultralytics_example.py \
        --source_weights_path yolov8s.pt \
        --source_video_path input.mp4 \
        --target_video_path tracking_result.mp4
    ```

## © 许可证

本示例集成了两个主要组件，各自具有不同的许可证：

- ultralytics：本示例中使用的目标检测模型 YOLOv8 基于 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 发布。更多详细信息请参见此许可证。

- supervision：本示例中支撑区域分析的分析代码基于 Supervision 库，其使用 [MIT 许可证](https://github.com/roboflow/supervision/blob/develop/LICENSE.md) 发布。这意味着 Supervision 部分的代码完全开源，可在你的项目中自由使用。
