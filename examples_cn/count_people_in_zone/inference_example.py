import json
import os

import cv2
import numpy as np
from inference.core.models.roboflow import RoboflowInferenceModel
from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv

COLORS = sv.ColorPalette.DEFAULT


def load_zones_config(file_path: str) -> list[np.ndarray]:
    """
    从 JSON 文件中加载多边形区域配置。

    该函数读取包含多边形坐标的 JSON 文件，并将其转换为一个 NumPy 数组列表。
    每个多边形由一个 NumPy 坐标数组表示。

    Args:
        file_path (str): JSON 配置文件的路径。

    Returns:
        list[np.ndarray]: 多边形列表，每个多边形由一个 NumPy 数组表示。
    """
    with open(file_path) as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data["polygons"]]


def initiate_annotators(
    polygons: list[np.ndarray], resolution_wh: tuple[int, int]
) -> tuple[list[sv.PolygonZone], list[sv.PolygonZoneAnnotator], list[sv.BoxAnnotator]]:
    """根据多边形与分辨率初始化多边形区域、区域标注器和框标注器。"""
    line_thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    zones = []
    zone_annotators = []
    box_annotators = []

    for index, polygon in enumerate(polygons):
        zone = sv.PolygonZone(polygon=polygon)
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=COLORS.by_idx(index),
            thickness=line_thickness,
            text_thickness=line_thickness * 2,
            text_scale=text_scale * 2,
        )
        box_annotator = sv.BoxAnnotator(
            color=COLORS.by_idx(index), thickness=line_thickness
        )
        zones.append(zone)
        zone_annotators.append(zone_annotator)
        box_annotators.append(box_annotator)

    return zones, zone_annotators, box_annotators


def detect(
    frame: np.ndarray,
    model: RoboflowInferenceModel,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.7,
) -> sv.Detections:
    """
    使用 Inference 模型在帧中检测目标，并按类别 ID 和置信度进行过滤。

    Args:
        frame (np.ndarray): 待处理的帧，应为 NumPy 数组。
        model (RoboflowInferenceModel): 用于处理帧的 Inference 模型。
        confidence_threshold (float): 用于过滤检测结果的置信度阈值。
        iou_threshold (float): 非极大值抑制的 IOU 阈值。

    Returns:
        sv.Detections: 使用 Inference 模型处理帧后过滤得到的检测结果。

    Note:
        该函数针对 Inference 模型定制，假设使用类别 ID 0 进行过滤。
    """
    results = model.infer(frame, confidence=confidence_threshold, iou=iou_threshold)[0]
    detections = sv.Detections.from_inference(results)
    filter_by_class = detections.class_id == 0
    filter_by_confidence = detections.confidence > confidence_threshold
    return detections[filter_by_class & filter_by_confidence]


def annotate(
    frame: np.ndarray,
    zones: list[sv.PolygonZone],
    zone_annotators: list[sv.PolygonZoneAnnotator],
    box_annotators: list[sv.BoxAnnotator],
    detections: sv.Detections,
) -> np.ndarray:
    """
    根据给定的检测结果，在帧上绘制区域和检测框标注。

    Args:
        frame (np.ndarray): 原始待标注帧。
        zones (list[sv.PolygonZone]): 用于检测的多边形区域列表。
        zone_annotators (list[sv.PolygonZoneAnnotator]): 用于绘制区域标注的标注器列表。
        box_annotators (list[sv.BoxAnnotator]): 用于绘制检测框标注的标注器列表。
        detections (sv.Detections): 用于标注的检测结果。

    Returns:
        np.ndarray: 标注后的帧。
    """
    annotated_frame = frame.copy()
    for zone, zone_annotator, box_annotator in zip(
        zones, zone_annotators, box_annotators
    ):
        detections_in_zone = detections[zone.trigger(detections=detections)]
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections_in_zone
        )
    return annotated_frame


def main(
    zone_configuration_path: str,
    source_video_path: str,
    model_id: str = "yolov8x-1280",
    roboflow_api_key: str | None = None,
    target_video_path: str | None = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
):
    """
    使用 Inference 和 Supervision 在区域中统计人数。

    Args:
        zone_configuration_path: 区域配置 JSON 文件路径
        source_video_path: 源视频文件路径
        model_id: Roboflow 模型 ID
        roboflow_api_key: Roboflow API KEY
        target_video_path: 目标（输出）视频文件路径
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

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    polygons = load_zones_config(zone_configuration_path)
    zones, zone_annotators, box_annotators = initiate_annotators(
        polygons=polygons, resolution_wh=video_info.resolution_wh
    )

    model = get_roboflow_model(model_id=model_id, api_key=roboflow_api_key)

    frames_generator = sv.get_video_frames_generator(source_video_path)
    if target_video_path is not None:
        # 有输出路径时：写入视频文件
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames):
                detections = detect(frame, model, confidence_threshold, iou_threshold)
                annotated_frame = annotate(
                    frame=frame,
                    zones=zones,
                    zone_annotators=zone_annotators,
                    box_annotators=box_annotators,
                    detections=detections,
                )
                sink.write_frame(annotated_frame)
    else:
        # 无输出路径时：实时显示
        for frame in tqdm(frames_generator, total=video_info.total_frames):
            detections = detect(frame, model, confidence_threshold, iou_threshold)
            annotated_frame = annotate(
                frame=frame,
                zones=zones,
                zone_annotators=zone_annotators,
                box_annotators=box_annotators,
                detections=detections,
            )
            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
