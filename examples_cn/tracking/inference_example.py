import os

from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv


def main(
    source_video_path: str,
    target_video_path: str,
    roboflow_api_key: str,
    model_id: str = "yolov8x-1280",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """
    使用 Inference 和 ByteTrack 进行视频处理。

    Args:
        source_video_path: 源视频文件路径
        target_video_path: 目标（输出）视频文件路径
        roboflow_api_key: Roboflow API key
        model_id: Roboflow 模型 ID
        confidence_threshold: 模型的置信度阈值
        iou_threshold: 模型的 IOU 阈值
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", roboflow_api_key)
    if api_key is None:
        raise ValueError(
            "缺少 Roboflow API key。请通过参数传入或设置 ROBOFLOW_API_KEY 环境变量。"
        )

    model = get_roboflow_model(model_id=model_id, api_key=api_key)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results = model.infer(
                frame, confidence=confidence_threshold, iou_threshold=iou_threshold
            )[0]
            detections = sv.Detections.from_inference(results)
            detections = tracker.update_with_detections(detections)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            sink.write_frame(frame=annotated_labeled_frame)


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
