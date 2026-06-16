from datetime import datetime

import numpy as np

import supervision as sv


class FPSBasedTimer:
    """
    基于 FPS（每秒帧数）的计时器，用于计算每个目标被检测到的持续时间。

    Attributes:
        fps (float): 视频流的帧率，用于计算时间长度。
        frame_id (int): 当前帧在序列中的编号。
        tracker_id2frame_id (dict[int, int]): 将每个跟踪器的 ID 映射到其
            首次被检测到的帧编号。
    """

    def __init__(self, fps: float = 30) -> None:
        """使用指定的帧率初始化 FPSBasedTimer。

        Args:
            fps (float): 视频流的帧率，默认为 30。
        """
        self.fps = fps
        self.frame_id = 0
        self.tracker_id2frame_id: dict[int, int] = {}

    def tick(self, detections: sv.Detections) -> np.ndarray:
        """处理当前帧，更新每个跟踪器的时间长度。

        Args:
            detections: 当前帧的检测结果，包含跟踪器 ID。

        Returns:
            np.ndarray: 每个被检测跟踪器自首次检测以来的时间长度（秒）。
        """
        self.frame_id += 1
        times = []

        for tracker_id in detections.tracker_id:
            self.tracker_id2frame_id.setdefault(tracker_id, self.frame_id)

            start_frame_id = self.tracker_id2frame_id[tracker_id]
            time_duration = (self.frame_id - start_frame_id) / self.fps
            times.append(time_duration)

        return np.array(times)


class ClockBasedTimer:
    """
    基于系统时钟的计时器，用于计算每个目标被检测到的持续时间。

    Attributes:
        tracker_id2start_time (dict[int, datetime]): 将每个跟踪器的 ID 映射到
            其首次被检测到的 datetime。
    """

    def __init__(self) -> None:
        """初始化 ClockBasedTimer。"""
        self.tracker_id2start_time: dict[int, datetime] = {}

    def tick(self, detections: sv.Detections) -> np.ndarray:
        """处理当前帧，更新每个跟踪器的时间长度。

        Args:
            detections: 当前帧的检测结果，包含跟踪器 ID。

        Returns:
            np.ndarray: 每个被检测跟踪器自首次检测以来的时间长度（秒）。
        """
        current_time = datetime.now()
        times = []

        for tracker_id in detections.tracker_id:
            self.tracker_id2start_time.setdefault(tracker_id, current_time)

            start_time = self.tracker_id2start_time[tracker_id]
            time_duration = (current_time - start_time).total_seconds()
            times.append(time_duration)

        return np.array(times)
