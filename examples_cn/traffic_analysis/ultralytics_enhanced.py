from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

# ----- 区域定义（与原示例保持一致） -------------------------------------------

COLORS = sv.ColorPalette.from_hex(
    [
        "#E6194B",  # 红
        "#3CB44B",  # 绿
        "#FFE119",  # 黄
        "#3C76D1",  # 蓝
    ]
)

ZONE_IN_POLYGONS: list[np.ndarray] = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
]

ZONE_OUT_POLYGONS: list[np.ndarray] = [
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
]

# ----- 车辆类型归一化 ---------------------------------------------------------

#: 默认类型色板。键为归一化后的类型，值为 :class:`sv.Color`，便于在标注中保持一致。
VEHICLE_TYPE_COLORS: dict[str, sv.Color] = {
    "car": sv.Color.from_hex("#34B44B"),          # 绿
    "truck": sv.Color.from_hex("#D63C45"),        # 红
    "bus": sv.Color.from_hex("#FFE119"),          # 黄
    "van": sv.Color.from_hex("#3C76D1"),          # 蓝
    "motorcycle": sv.Color.from_hex("#9E2FA5"),   # 紫
    "bicycle": sv.Color.from_hex("#F08A2A"),      # 橙
    "other": sv.Color.from_hex("#C8C8C8"),        # 灰
}

#: 把模型输出的原始类别名映射到归一化车辆类型。
#: 兼容 Ultralytics YOLO（car/truck/bus/motorcycle/bicycle）与本仓库示例权重
#: （traffic_analysis.pt 仅输出 vehicle 类，会被映射为 "other"，方便复用）。
DEFAULT_VEHICLE_TYPE_ALIASES: dict[str, str] = {
    "car": "car",
    "sedan": "car",
    "suv": "car",
    "taxi": "car",
    "truck": "truck",
    "pickup": "truck",
    "bus": "bus",
    "van": "van",
    "minivan": "van",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "vehicle": "other",
    "vehicles": "other",
    "other": "other",
}


def normalize_vehicle_type(
    raw_name: str,
    aliases: dict[str, str] = DEFAULT_VEHICLE_TYPE_ALIASES,
) -> str:
    """把模型原始类别名归一化为车辆类型；未匹配则返回 'other'."""
    if raw_name is None:
        return "other"
    key = str(raw_name).strip().lower()
    return aliases.get(key, "other")


# ----- 自定义 Detections.data 字段名 ------------------------------------------

#: ``Detections.data`` 中存储归一化车辆类型的字段名。
VEHICLE_TYPE_DATA_FIELD: str = "vehicle_type"
#: ``Detections.data`` 中存储该帧时间戳（秒）的字段名。
FRAME_TIMESTAMP_DATA_FIELD: str = "frame_timestamp"
#: ``Detections.data`` 中存储该帧帧号的字段名。
FRAME_INDEX_DATA_FIELD: str = "frame_index"


# ----- 数据结构 --------------------------------------------------------------


@dataclass
class TrajectoryPoint:
    """单帧上的目标位置。"""

    frame_index: int
    timestamp: float
    xyxy: tuple[float, float, float, float]


@dataclass
class VehicleTrack:
    """单个跟踪 ID 的完整轨迹与汇总信息。"""

    tracker_id: int
    vehicle_type: str = "other"
    class_name: str = ""
    points: list[TrajectoryPoint] = field(default_factory=list)
    in_zone_id: int = -1
    out_zone_id: int = -1
    first_seen_frame: int = -1
    last_seen_frame: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker_id": self.tracker_id,
            "vehicle_type": self.vehicle_type,
            "class_name": self.class_name,
            "in_zone_id": self.in_zone_id,
            "out_zone_id": self.out_zone_id,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "n_points": len(self.points),
            "trajectory": [
                {
                    "frame_index": p.frame_index,
                    "timestamp": round(p.timestamp, 4),
                    "x1": p.xyxy[0],
                    "y1": p.xyxy[1],
                    "x2": p.xyxy[2],
                    "y2": p.xyxy[3],
                }
                for p in self.points
            ],
        }


class TrajectoryStore:
    """按 ``tracker_id`` 维护的轨迹仓库。"""

    def __init__(self, max_points: int = 10_000) -> None:
        self._max_points = max_points
        self._tracks: dict[int, VehicleTrack] = {}

    def update(
        self,
        detections: sv.Detections,
        frame_index: int,
        timestamp: float,
        in_zone_assignments: dict[int, int] | None = None,
        out_zone_assignments: dict[int, int] | None = None,
    ) -> None:
        if detections.tracker_id is None or len(detections) == 0:
            return
        vehicle_types = detections.data.get(VEHICLE_TYPE_DATA_FIELD)
        class_names = detections.data.get(CLASS_NAME_DATA_FIELD)
        in_zone_assignments = in_zone_assignments or {}
        out_zone_assignments = out_zone_assignments or {}
        for i, tracker_id in enumerate(detections.tracker_id):
            tid = int(tracker_id)
            track = self._tracks.get(tid)
            vehicle_type = (
                str(vehicle_types[i]) if vehicle_types is not None else "other"
            )
            class_name = (
                str(class_names[i]) if class_names is not None else ""
            )
            in_zone = int(in_zone_assignments.get(tid, -1))
            out_zone = int(out_zone_assignments.get(tid, -1))
            if track is None:
                track = VehicleTrack(
                    tracker_id=tid,
                    vehicle_type=vehicle_type,
                    class_name=class_name,
                    first_seen_frame=frame_index,
                    in_zone_id=in_zone,
                    out_zone_id=out_zone,
                )
                self._tracks[tid] = track
            else:
                # 以首次观测的类型为主；只在未知时更新
                if track.vehicle_type == "other" and vehicle_type != "other":
                    track.vehicle_type = vehicle_type
                if not track.class_name and class_name:
                    track.class_name = class_name
                if track.in_zone_id < 0 and in_zone >= 0:
                    track.in_zone_id = in_zone
                if track.out_zone_id < 0 and out_zone >= 0:
                    track.out_zone_id = out_zone
            x1, y1, x2, y2 = detections.xyxy[i]
            track.points.append(
                TrajectoryPoint(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
            if len(track.points) > self._max_points:
                track.points = track.points[-self._max_points :]
            track.last_seen_frame = frame_index

    def get(self, tracker_id: int) -> VehicleTrack | None:
        return self._tracks.get(int(tracker_id))

    def all(self) -> list[VehicleTrack]:
        return list(self._tracks.values())

    def summary_frame(self) -> pd.DataFrame:
        """每辆车一行的汇总表，便于可视化与下游分析。"""
        rows = [
            {
                "tracker_id": t.tracker_id,
                "vehicle_type": t.vehicle_type,
                "class_name": t.class_name,
                "in_zone_id": None if t.in_zone_id < 0 else t.in_zone_id,
                "out_zone_id": None if t.out_zone_id < 0 else t.out_zone_id,
                "first_seen_frame": t.first_seen_frame,
                "last_seen_frame": t.last_seen_frame,
                "n_points": len(t.points),
            }
            for t in self._tracks.values()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["in_zone_id"] = df["in_zone_id"].astype("Int64")
            df["out_zone_id"] = df["out_zone_id"].astype("Int64")
        return df

    def trajectories_frame(self) -> pd.DataFrame:
        """每条轨迹点一行的长表，适合画热力图/轨迹图。"""
        rows: list[dict[str, Any]] = []
        for t in self._tracks.values():
            for p in t.points:
                rows.append(
                    {
                        "tracker_id": t.tracker_id,
                        "vehicle_type": t.vehicle_type,
                        "frame_index": p.frame_index,
                        "timestamp": round(p.timestamp, 4),
                        "x1": p.xyxy[0],
                        "y1": p.xyxy[1],
                        "x2": p.xyxy[2],
                        "y2": p.xyxy[3],
                    }
                )
        return pd.DataFrame(rows)


class DetectionsManager:
    """聚合 ``in -> out`` 流量，并按车辆类型细分。"""

    def __init__(self) -> None:
        self.tracker_id_to_zone_id: dict[int, int] = {}
        self.tracker_id_to_vehicle_type: dict[int, str] = {}
        # 嵌套：counts[out_zone_id][in_zone_id][vehicle_type] -> set[tracker_id]
        self.counts: dict[int, dict[int, dict[str, set[int]]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: list[sv.Detections],
        detections_out_zones: list[sv.Detections],
    ) -> sv.Detections:
        # 1) 记录每个 tracker 首次进入的入口区域与车辆类型
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            if detections_in_zone.tracker_id is None:
                continue
            types = detections_in_zone.data.get(VEHICLE_TYPE_DATA_FIELD)
            for i, tracker_id in enumerate(detections_in_zone.tracker_id):
                tracker_id = int(tracker_id)
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
                if types is not None:
                    self.tracker_id_to_vehicle_type.setdefault(
                        tracker_id, str(types[i])
                    )

        # 2) 出现于出口区域时，登记一次 (out, in, type) 通行
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            if detections_out_zone.tracker_id is None:
                continue
            types = detections_out_zone.data.get(VEHICLE_TYPE_DATA_FIELD)
            for i, tracker_id in enumerate(detections_out_zone.tracker_id):
                tracker_id = int(tracker_id)
                if tracker_id not in self.tracker_id_to_zone_id:
                    continue
                zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                vehicle_type = (
                    str(types[i])
                    if types is not None
                    else self.tracker_id_to_vehicle_type.get(tracker_id, "other")
                )
                bucket = self.counts.setdefault(zone_out_id, {})
                bucket_in = bucket.setdefault(zone_in_id, {})
                bucket_type = bucket_in.setdefault(vehicle_type, set())
                bucket_type.add(tracker_id)

        # 3) 重写 class_id 为入口区域 id，方便过滤与按区域上色
        if len(detections_all) > 0 and detections_all.tracker_id is not None:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(int(x), -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]

    def od_dataframe(self) -> pd.DataFrame:
        """长表：每一行是一次 in->out 通行（去重，按 tracker）。"""
        rows: list[dict[str, Any]] = []
        for out_zone_id, in_map in self.counts.items():
            for in_zone_id, type_map in in_map.items():
                for vehicle_type, ids in type_map.items():
                    rows.append(
                        {
                            "in_zone_id": in_zone_id,
                            "out_zone_id": out_zone_id,
                            "vehicle_type": vehicle_type,
                            "count": len(ids),
                        }
                    )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(
            ["in_zone_id", "out_zone_id", "vehicle_type"]
        ).reset_index(drop=True)


# ----- 多边形区域工具 ---------------------------------------------------------


def initiate_polygon_zones(
    polygons: list[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = (sv.Position.CENTER,),
) -> list[sv.PolygonZone]:
    return [
        sv.PolygonZone(polygon=p, triggering_anchors=triggering_anchors)
        for p in polygons
    ]


# ----- 主处理流水线 -----------------------------------------------------------


class VideoProcessor:
    """交通流量分析流水线（增强版）。"""

    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str | None = None,
        target_csv_path: str | None = None,
        target_json_path: str | None = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        trajectory_max_points: int = 10_000,
        vehicle_type_aliases: dict[str, str] | None = None,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.target_csv_path = target_csv_path
        self.target_json_path = target_json_path
        self.vehicle_type_aliases = (
            vehicle_type_aliases or DEFAULT_VEHICLE_TYPE_ALIASES
        )

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, [sv.Position.CENTER]
        )

        # 标注器：Box/Label 改为按 vehicle_type 上色，需要自定义 color_lookup
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS,
            position=sv.Position.CENTER,
            trace_length=100,
            thickness=2,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.type_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )

        self.detections_manager = DetectionsManager()
        self.trajectory_store = TrajectoryStore(max_points=trajectory_max_points)
        self._frame_index: int = 0

    # --- 工具方法 ----------------------------------------------------------

    def _vehicle_type_lookup(self, detections: sv.Detections) -> np.ndarray:
        """根据 ``data[CLASS_NAME_DATA_FIELD]`` 生成车辆类型数组。"""
        if len(detections) == 0:
            return np.empty(0, dtype=object)
        class_names = detections.data.get(CLASS_NAME_DATA_FIELD)
        if class_names is None:
            return np.array(["other"] * len(detections), dtype=object)
        return np.array(
            [normalize_vehicle_type(n, self.vehicle_type_aliases) for n in class_names],
            dtype=object,
        )

    def _type_color_lookup(self, vehicle_types: np.ndarray) -> np.ndarray:
        """把车辆类型数组转换成 ``ColorLookup.INDEX`` 用的索引数组。"""
        unique = sorted({t for t in vehicle_types})
        index_of = {t: i for i, t in enumerate(unique)}
        return np.array([index_of[t] for t in vehicle_types], dtype=int)

    def _per_frame_csv_row(self) -> dict[str, Any] | None:
        """导出当前帧每个 tracker 的扁平表行，便于后续时序可视化。"""
        rows: list[dict[str, Any]] = []
        for t in self.trajectory_store.all():
            if t.last_seen_frame != self._frame_index or not t.points:
                continue
            p = t.points[-1]
            rows.append(
                {
                    "frame_index": self._frame_index,
                    "timestamp": round(p.timestamp, 4),
                    "tracker_id": t.tracker_id,
                    "vehicle_type": t.vehicle_type,
                    "class_name": t.class_name,
                    "in_zone_id": t.in_zone_id if t.in_zone_id is not None else -1,
                    "out_zone_id": t.out_zone_id if t.out_zone_id is not None else -1,
                    "x1": p.xyxy[0],
                    "y1": p.xyxy[1],
                    "x2": p.xyxy[2],
                    "y2": p.xyxy[3],
                }
            )
        return rows

    # --- 帧处理 ----------------------------------------------------------

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        # 轨迹（按 track 着色）
        annotated_frame = self.trace_annotator.annotate(
            annotated_frame, detections
        )

        # 按车辆类型上色：构造 ColorLookup.INDEX 所需的索引
        vehicle_types = detections.data.get(VEHICLE_TYPE_DATA_FIELD)
        if vehicle_types is None or len(vehicle_types) == 0:
            vehicle_types = np.array([], dtype=object)
        if len(vehicle_types) > 0:
            color_lookup = self._type_color_lookup(vehicle_types)
        else:
            color_lookup = np.array([], dtype=int)

        annotated_frame = self.box_annotator.annotate(
            annotated_frame,
            detections,
            custom_color_lookup=color_lookup,
        )

        # 标签：ID + 类型
        if len(detections) > 0:
            labels = [
                f"#{tid} {vtype}"
                for tid, vtype in zip(
                    detections.tracker_id, vehicle_types
                )
            ]
        else:
            labels = []
        annotated_frame = self.type_annotator.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )

        # 出口区域附近显示按入口分组、按类型分色的计数
        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id not in self.detections_manager.counts:
                continue
            counts = self.detections_manager.counts[zone_out_id]
            for i, zone_in_id in enumerate(counts):
                type_map = counts[zone_in_id]
                offset = 0
                for vehicle_type, ids in type_map.items():
                    text = f"{vehicle_type}:{len(ids)}"
                    text_anchor = sv.Point(
                        x=zone_center.x,
                        y=zone_center.y + 22 * offset,
                    )
                    color = VEHICLE_TYPE_COLORS.get(
                        vehicle_type, VEHICLE_TYPE_COLORS["other"]
                    )
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=text,
                        text_anchor=text_anchor,
                        background_color=color,
                    )
                    offset += 1

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        timestamp = self._frame_index / max(self.video_info.fps, 1e-6)
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        # 重新计算 vehicle_type 后再跟踪，确保类型字段在更新中保持对齐
        detections.data[VEHICLE_TYPE_DATA_FIELD] = self._vehicle_type_lookup(
            detections
        )
        detections.data[FRAME_INDEX_DATA_FIELD] = np.array(
            [self._frame_index] * len(detections)
        )
        detections.data[FRAME_TIMESTAMP_DATA_FIELD] = np.array(
            [timestamp] * len(detections)
        )

        detections = self.tracker.update_with_detections(detections)
        # `update_with_detections` 会通过 `Detections.__getitem__` 对 `data` 同步子集化，
        # 因此 `vehicle_type` 等对齐字段无需手动重新计算。

        # 入口/出口区域过滤
        detections_in_zones: list[sv.Detections] = []
        detections_out_zones: list[sv.Detections] = []
        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zones.append(
                detections[zone_in.trigger(detections=detections)]
            )
            detections_out_zones.append(
                detections[zone_out.trigger(detections=detections)]
            )

        # 累计通行量
        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )

        # 当前帧的入/出口区域分配，用于回写到轨迹
        in_zone_assignments: dict[int, int] = {}
        for zone_in_id, dets in enumerate(detections_in_zones):
            if dets.tracker_id is None:
                continue
            for tid in dets.tracker_id:
                in_zone_assignments.setdefault(int(tid), zone_in_id)
        out_zone_assignments: dict[int, int] = {}
        for zone_out_id, dets in enumerate(detections_out_zones):
            if dets.tracker_id is None:
                continue
            for tid in dets.tracker_id:
                out_zone_assignments.setdefault(int(tid), int(zone_out_id))

        # 累计轨迹
        self.trajectory_store.update(
            detections,
            self._frame_index,
            timestamp,
            in_zone_assignments=in_zone_assignments,
            out_zone_assignments=out_zone_assignments,
        )

        self._frame_index += 1
        return self.annotate_frame(frame, detections)

    def process_video(self) -> None:
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )
        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(
                    frame_generator, total=self.video_info.total_frames
                ):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    # --- 导出 ------------------------------------------------------------

    def get_od_dataframe(self) -> pd.DataFrame:
        """in->out 通行量长表，列为 ``in_zone_id / out_zone_id / vehicle_type / count``."""
        return self.detections_manager.od_dataframe()

    def get_trajectory_summary(self) -> pd.DataFrame:
        return self.trajectory_store.summary_frame()

    def get_trajectories(self) -> pd.DataFrame:
        return self.trajectory_store.trajectories_frame()

    def export_results(
        self,
        output_dir: str,
        base_name: str = "traffic_analysis",
    ) -> dict[str, str]:
        """把统计数据落盘，返回写入的文件路径字典。

        生成：

        - ``<base_name>_od_matrix.csv`` —— OD 长表
        - ``<base_name>_od_matrix.json`` —— 嵌套 JSON（含 OD 矩阵与按类型分项）
        - ``<base_name>_od_matrix.parquet`` —— Parquet 版
        - ``<base_name>_tracks_summary.csv`` —— 每辆车的汇总
        - ``<base_name>_trajectories.csv`` —— 每条轨迹点
        """
        os.makedirs(output_dir, exist_ok=True)
        od = self.get_od_dataframe()
        od_csv = os.path.join(output_dir, f"{base_name}_od_matrix.csv")
        od_json = os.path.join(output_dir, f"{base_name}_od_matrix.json")
        od_parquet = os.path.join(output_dir, f"{base_name}_od_matrix.parquet")
        tracks_csv = os.path.join(output_dir, f"{base_name}_tracks_summary.csv")
        traj_csv = os.path.join(output_dir, f"{base_name}_trajectories.csv")

        if not od.empty:
            od.to_csv(od_csv, index=False)
            try:
                od.to_parquet(od_parquet, index=False)
            except Exception:
                # pyarrow 未安装时跳过，不影响 CSV
                od_parquet_unset = True
            else:
                od_parquet_unset = False
        else:
            pd.DataFrame(
                columns=["in_zone_id", "out_zone_id", "vehicle_type", "count"]
            ).to_csv(od_csv, index=False)
            od_parquet_unset = True

        # 嵌套 JSON：保留 OD 矩阵和按类型分项，便于直接喂给前端
        nested: dict[str, Any] = {
            "video": {
                "path": self.source_video_path,
                "fps": self.video_info.fps,
                "width": self.video_info.width,
                "height": self.video_info.height,
            },
            "zones": {
                "in": [poly.tolist() for poly in ZONE_IN_POLYGONS],
                "out": [poly.tolist() for poly in ZONE_OUT_POLYGONS],
            },
            "vehicle_types": sorted(
                {t for tracks in self.trajectory_store.all() for t in [tracks.vehicle_type]}
            ),
            "od_matrix": od.to_dict(orient="records"),
            "od_matrix_pivot": _pivot_to_nested(od),
        }
        with open(od_json, "w", encoding="utf-8") as fp:
            json.dump(nested, fp, ensure_ascii=False, indent=2)

        self.get_trajectory_summary().to_csv(tracks_csv, index=False)
        self.get_trajectories().to_csv(traj_csv, index=False)

        return {
            "od_csv": od_csv,
            "od_json": od_json,
            "od_parquet": od_parquet if not od.empty and not od_parquet_unset else "",
            "tracks_csv": tracks_csv,
            "trajectories_csv": traj_csv,
        }


def _pivot_to_nested(od: pd.DataFrame) -> dict[str, Any]:
    """把 OD 长表转成 ``{in_zone_id: {out_zone_id: {type: count}}}`` 嵌套结构。"""
    if od.empty:
        return {}
    result: dict[str, dict[str, dict[str, int]]] = {}
    for row in od.itertuples(index=False):
        result.setdefault(str(row.in_zone_id), {}).setdefault(
            str(row.out_zone_id), {}
        )[row.vehicle_type] = int(row.count)
    return result


# ----- CLI 入口 ---------------------------------------------------------------


def main(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str | None = None,
    output_dir: str | None = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """使用 YOLO 和 ByteTrack 进行交通流量分析（增强版）。

    Args:
        source_weights_path: YOLO 权重路径
        source_video_path: 待分析视频路径
        target_video_path: 可选，标注结果视频的保存路径
        output_dir: 可选，统计 CSV / JSON / Parquet 输出目录
        confidence_threshold: 模型置信度阈值
        iou_threshold: 模型 IOU 阈值
    """
    processor = VideoProcessor(
        source_weights_path=source_weights_path,
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    processor.process_video()
    if output_dir:
        paths = processor.export_results(output_dir)
        for name, path in paths.items():
            if path:
                print(f"[export] {name}: {path}")


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
