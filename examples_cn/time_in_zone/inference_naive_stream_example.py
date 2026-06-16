import cv2
import numpy as np
from inference import get_model
from utils.general import find_in_list, get_stream_frames_generator, load_zones_config
from utils.timers import ClockBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


def main(
    zone_configuration_path: str,
    rtsp_url: str,
    model_id: str = "rfdetr-medium",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    classes: list[int] = [],
    roboflow_api_key: str = "",
) -> None:
    """
    在区域中计算检测目标的停留时长（使用 RTSP 流，naive 方式）。

    Args:
        zone_configuration_path: 区域配置 JSON 文件路径
        rtsp_url: 视频流的完整 RTSP URL
        model_id: Roboflow 模型 ID
        confidence_threshold: 检测的置信度阈值（0 到 1）
        iou_threshold: 非极大值抑制的 IOU 阈值
        classes: 要跟踪的目标类别 ID 列表。留空则跟踪所有类别
        roboflow_api_key: 用于访问私有模型的 Roboflow API key
    """
    model = get_model(model_id=model_id, api_key=roboflow_api_key)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    frames_generator = get_stream_frames_generator(rtsp_url=rtsp_url)
    fps_monitor = sv.FPSMonitor()

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    # naive 流处理使用基于时钟的计时器
    timers = [ClockBasedTimer() for _ in zones]

    for frame in frames_generator:
        fps_monitor.tick()
        fps = fps_monitor.fps

        results = model.infer(
            frame, confidence=confidence_threshold, iou_threshold=iou_threshold
        )[0]
        detections = sv.Detections.from_inference(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(zones):
            # 绘制区域多边形
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            # 标签格式：#<跟踪器 ID> <mm:ss>
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
