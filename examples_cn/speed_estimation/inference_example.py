import os
from collections import defaultdict, deque

import cv2
import numpy as np
from inference.models.utils import get_roboflow_model

import supervision as sv

# 透视变换：源（视频画面）→ 目标（鸟瞰图矩形）
# 这些坐标针对特定的交通摄像头角度，使用前必须根据你的视角调整
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    """基于 OpenCV 透视变换的 2D 坐标映射工具。"""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """将一组 (N, 2) 点变换到目标坐标系。"""
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def main(
    source_video_path: str,
    target_video_path: str,
    model_id: str = "yolov8x-640",
    roboflow_api_key: str | None = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
):
    """
    使用 Inference 和 Supervision 进行车辆测速。

    Args:
        source_video_path: 源视频文件路径
        target_video_path: 目标（输出）视频文件路径
        model_id: Roboflow 模型 ID
        roboflow_api_key: Roboflow API KEY
        confidence_threshold: 模型的置信度阈值
        iou_threshold: 模型的 IOU 阈值
    """
    api_key = roboflow_api_key
    api_key = os.environ.get("ROBOFLOW_API_KEY", api_key)
    if api_key is None:
        raise ValueError(
            "缺少 Roboflow API key。请通过参数传入或设置 ROBOFLOW_API_KEY 环境变量。"
        )
    roboflow_api_key = api_key

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    model = get_roboflow_model(model_id=model_id, api_key=roboflow_api_key)

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # 为每个跟踪器维护一个滑动窗口，存储最近 N 帧的 y 坐标
    coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            results = model.infer(
                frame, confidence=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_inference(results)
            # 仅保留在多边形区域内的检测
            detections = detections[polygon_zone.trigger(detections)]
            detections = byte_track.update_with_detections(detections=detections)

            # 获取底部中心锚点，并投影到鸟瞰图坐标
            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    # 历史数据不足，仅显示跟踪器 ID
                    labels.append(f"#{tracker_id}")
                else:
                    # 使用窗口首尾位置的差值估算速度
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
