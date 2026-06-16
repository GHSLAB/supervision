from typing import Optional

import cv2
from ultralytics import YOLO

import supervision as sv
from supervision.assets import VideoAssets, download_assets


def download_video() -> str:
    """下载默认的示例视频（people-walking.mp4）。"""
    download_assets(VideoAssets.PEOPLE_WALKING)
    return VideoAssets.PEOPLE_WALKING.value


def main(
    source_weights_path: str,
    source_video_path: Optional[str] = None,
    target_video_path: str = "output.mp4",
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    heatmap_alpha: float = 0.5,
    radius: int = 25,
    track_activation_threshold: float = 0.35,
    track_seconds: int = 5,
    minimum_matching_threshold: float = 0.99,
) -> None:
    """
    使用 Supervision 进行热力图与目标跟踪。

    Args:
        source_weights_path: 源权重文件路径
        source_video_path: 源视频文件路径
        target_video_path: 目标（输出）视频文件路径
        confidence_threshold: 模型的置信度阈值
        iou_threshold: 模型的 IOU 阈值
        heatmap_alpha: 叠加遮罩的不透明度（0 到 1）
        radius: 热力圆点的半径
        track_activation_threshold: 激活跟踪所需的检测置信度阈值
        track_seconds: 跟踪丢失时缓存的秒数
        minimum_matching_threshold: 将跟踪与检测进行匹配时的阈值
    """
    ### 实例化模型
    model = YOLO(source_weights_path)
    source_video_path = source_video_path or download_video()

    ### 热力图配置
    heat_map_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=heatmap_alpha,
        radius=radius,
        kernel_size=25,
        top_hue=0,
        low_hue=125,
    )

    ### 标注器配置
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    ### 获取视频的 FPS
    cap = cv2.VideoCapture(source_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    ### 跟踪器配置
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=track_seconds * fps,
        minimum_matching_threshold=minimum_matching_threshold,
        frame_rate=fps,
    )

    ### 视频配置
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=1
    )

    ### 检测、跟踪、标注、保存
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in frames_generator:
            result = model(
                source=frame,
                classes=[0],  # 仅 person 类别
                conf=confidence_threshold,
                iou=iou_threshold,
                # show_conf = True,
                # save_txt = True,
                # save_conf = True,
                # save = True,
                device=None,  # None = CPU, 0 = 单 GPU, 或 [0,1] = 双 GPU
            )[0]

            detections = sv.Detections.from_ultralytics(result)  # 获取检测结果

            detections = byte_tracker.update_with_detections(
                detections
            )  # 更新跟踪器

            ### 绘制热力图
            annotated_frame = heat_map_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            ### 绘制来自 detections 对象的其他属性
            labels = [
                f"#{tracker_id}"
                for class_id, tracker_id in zip(
                    detections.class_id, detections.tracker_id
                )
            ]

            label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
