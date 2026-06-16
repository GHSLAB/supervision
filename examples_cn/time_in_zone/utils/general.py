import json
from collections.abc import Generator

import cv2
import numpy as np


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
        return [np.array(polygon, np.int32) for polygon in data]


def find_in_list(array: np.ndarray, search_list: list[int]) -> np.ndarray:
    """判断 NumPy 数组中的元素是否出现在列表中。

    Args:
        array (np.ndarray): 待检查的整数 NumPy 数组。
        search_list (list[int]): 用于搜索的整数列表。

    Returns:
        np.ndarray: 布尔值 NumPy 数组，每个布尔值表示 `array` 中对应元素
            是否在 `search_list` 中。
    """
    if not search_list:
        return np.ones(array.shape, dtype=bool)
    else:
        return np.isin(array, search_list)


def get_stream_frames_generator(rtsp_url: str) -> Generator[np.ndarray, None, None]:
    """
    从 RTSP 流中逐帧生成的生成器函数。

    Args:
        rtsp_url (str): RTSP 视频流的 URL。

    Yields:
        np.ndarray: 视频流中的下一帧。
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise Exception("错误：无法打开视频流。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流已结束或读取帧时出错。")
                break
            yield frame
    finally:
        cap.release()
