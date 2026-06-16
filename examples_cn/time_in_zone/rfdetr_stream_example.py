from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from utils.general import find_in_list, load_zones_config
from utils.timers import ClockBasedTimer

import supervision as sv


class ModelSize(Enum):
    """RF-DETR 支持的模型尺寸枚举。"""

    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    BASE = "base"
    LARGE = "large"

    @classmethod
    def list(cls):
        """返回所有可用的尺寸名称列表。"""
        return [c.value for c in cls]

    @classmethod
    def from_value(cls, value: ModelSize | str) -> ModelSize:
        """从枚举值或字符串中解析出 ModelSize，字符串不区分大小写。"""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            try:
                return cls(value)
            except ValueError as exc:
                raise ValueError(
                    f"非法的模型尺寸 '{value}'。必须是 {cls.list()} 之一。"
                ) from exc
        raise ValueError(
            f"非法的值类型 '{type(value)}'。应为 str 或 ModelSize。"
        )


def load_model(checkpoint: ModelSize | str, device: str, resolution: int):
    """根据 checkpoint 名称加载对应尺寸的 RF-DETR 模型。"""
    checkpoint = ModelSize.from_value(checkpoint)
    if checkpoint == ModelSize.NANO:
        return RFDETRNano(device=device, resolution=resolution)
    if checkpoint == ModelSize.SMALL:
        return RFDETRSmall(device=device, resolution=resolution)
    if checkpoint == ModelSize.MEDIUM:
        return RFDETRMedium(device=device, resolution=resolution)
    if checkpoint == ModelSize.BASE:
        return RFDETRBase(device=device, resolution=resolution)
    if checkpoint == ModelSize.LARGE:
        return RFDETRLarge(device=device, resolution=resolution)
    raise RuntimeError("未处理的 checkpoint 类型。")


def adjust_resolution(checkpoint: ModelSize | str, resolution: int) -> int:
    """将分辨率对齐到模型所要求的有效倍数（取最接近的一侧）。"""
    checkpoint = ModelSize.from_value(checkpoint)
    divisor = (
        32 if checkpoint in {ModelSize.NANO, ModelSize.SMALL, ModelSize.MEDIUM} else 56
    )
    remainder = resolution % divisor
    if remainder == 0:
        return resolution
    lower = resolution - remainder
    upper = lower + divisor
    return lower if resolution - lower < upper - resolution else upper


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class CustomSink:
    """InferencePipeline 自定义输出：跟踪 + 停留时长标注。"""

    def __init__(self, zone_configuration_path: str, classes: list[int]):
        self.classes = classes
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.8)
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]

    def on_prediction(self, detections: sv.Detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps
        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update_with_detections(detections)
        annotated_frame = frame.image.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )
        for idx, zone in enumerate(self.zones):
            # 绘制区域多边形
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone.polygon,
                color=COLORS.by_idx(idx),
            )
            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)
            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            # 标签格式：#<跟踪器 ID> <mm:ss>
            labels = [
                f"#{tracker_id} {int(t // 60):02d}:{int(t % 60):02d}"
                for tracker_id, t in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )
        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)


def main(
    rtsp_url: str,
    zone_configuration_path: str,
    resolution: int,
    model_size: str = "small",
    device: str = "cpu",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    classes: list[int] = [],
) -> None:
    """
    在区域中计算检测目标的停留时长（使用 RTSP 流，InferencePipeline + RF-DETR）。

    Args:
        rtsp_url: 视频流的完整 RTSP URL
        zone_configuration_path: 区域配置 JSON 文件路径
        resolution: 模型输入分辨率
        model_size: RF-DETR 模型尺寸（'nano'、'small'、'medium'、'base' 或 'large'）
        device: 计算设备（'cpu'、'mps' 或 'cuda'）
        confidence_threshold: 检测的置信度阈值（0 到 1）
        iou_threshold: 非极大值抑制的 IOU 阈值
        classes: 要跟踪的目标类别 ID 列表。留空则跟踪所有类别
    """
    resolution = adjust_resolution(checkpoint=model_size, resolution=resolution)
    model = load_model(checkpoint=model_size, device=device, resolution=resolution)

    def inference_callback(frames: list[VideoFrame]) -> list[sv.Detections]:
        dets = model.predict(frames[0].image, threshold=confidence_threshold)
        return [dets.with_nms(threshold=iou_threshold)]

    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes)
    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=rtsp_url,
        on_video_frame=inference_callback,
        on_prediction=sink.on_prediction,
    )
    pipeline.start()
    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
