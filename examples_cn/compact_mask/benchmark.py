"""CompactMask 演示与基准测试。

演示 ``CompactMask`` 是 ``supervision.Detections`` 中稠密 ``(N, H, W)`` 布尔数组的
即插即用替代方案，能显著降低内存占用并加速标注。

运行方式：
    uv run python examples/compact_mask/benchmark.py

无需 GPU 或真实模型——所有内容均由 NumPy 合成。掩码复杂度由 ``num_vertices``
控制：顶点数更多的随机多边形会产生更粗糙的边界，从而在每行中产生更多的 RLE 段。
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

import supervision as sv
from supervision.detection.compact_mask import CompactMask

console = Console(width=240, force_terminal=True)

REPETITIONS = 4
# time_reps 中并发运行的重复次数。每个线程独立计时，结果取平均。
# Numpy 在 C 层释放 GIL，因此线程在多核机器上能真正并行运行。
# 设为 1 即可关闭并发，恢复为串行计时循环。
PARALLEL = 3
# 当稠密 (N,H,W) 数组超过此阈值时跳过其计时——避免极端场景下的 OOM
# 或换页抖动，但仍会报告理论内存占用。
DENSE_SKIP_GB = 16.0
# 超过此阈值时跳过稠密 IoU *与 NMS* 计时：成对 (N,H,W) AND 极其昂贵——
# NMS 内部会调用 IoU，因此两者使用同一阈值。
IOU_DENSE_SKIP_GB = 1.0
# 稠密 IoU/NMS 的重复次数——单次执行就可能耗时数秒。
IOU_NMS_REPS = 2


# ══════════════════════════════════════════════════════════════════════════════
# 结果容器
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScenarioResult:
    name: str
    resolution: str  # 例如 "1920x1080"
    num_objects: int
    fill_name: str  # 例如 "5%"
    num_vertices: int  # 多边形顶点数——复杂度代理
    # 内存（理论值：原始 numpy 字节数）
    dense_bytes: int
    compact_bytes_theoretical: int
    # 内存（实际值：tracemalloc 峰值；当 dense_skipped=True 时 dense_bytes_actual=0）
    dense_bytes_actual: int
    compact_bytes_actual: int
    # 压缩往返开销——转换的绝对时间（始终测量）
    encode_s: float  # CompactMask.from_dense()  dense → compact
    decode_s: float  # compact_mask.to_dense()   compact → dense
    # 计时（当 dense_skipped=True 时为 nan）
    dense_area_s: float
    compact_area_s: float
    dense_filter_s: float
    compact_filter_s: float
    dense_annot_s: float
    compact_annot_s: float
    # 流水线阶段（相应 skip 标志为 True 时为 nan）
    dense_iou_s: float  # 当 iou_dense_skipped 时为 nan
    compact_iou_s: float
    dense_nms_s: float  # 当 dense_skipped 时为 nan
    compact_nms_s: float
    dense_merge_s: float  # 当 dense_skipped 时为 nan
    compact_merge_s: float
    dense_offset_s: float  # 当 dense_skipped 时为 nan
    compact_offset_s: float
    dense_centroids_s: float  # 当 dense_skipped 时为 nan
    compact_centroids_s: float
    # 正确性（阶段被跳过时为 None）
    pixel_perfect: bool | None
    areas_match: bool | None
    roundtrip_ok: bool | None
    iou_ok: bool | None
    nms_ok: bool | None
    nms_mismatch_count: (
        int  # 两条路径中 NMS 决策不同的检测数量（dense_skipped 时为 0）
    )
    merge_ok: bool | None
    offset_ok: bool | None
    centroids_ok: bool | None
    dense_resize_s: float  # dense_skipped 时为 nan
    compact_resize_s: float
    resize_ok: bool | None
    # skip 标志
    dense_skipped: bool = field(default=False)
    iou_dense_skipped: bool = field(default=False)



# ══════════════════════════════════════════════════════════════════════════════
# 合成数据辅助函数
# ══════════════════════════════════════════════════════════════════════════════


def make_scene(image_height: int, image_width: int) -> np.ndarray:
    """生成随机的 BGR 图像。"""
    return np.random.default_rng(42).integers(
        0, 255, (image_height, image_width, 3), dtype=np.uint8
    )


def _make_polygon_mask(
    image_height: int,
    image_width: int,
    center_x: int,
    center_y: int,
    axis_x: int,
    axis_y: int,
    rng: np.random.Generator,
    num_vertices: int,
) -> np.ndarray:
    """生成随机多边形掩码。

    *num_vertices* 直接用作复杂度代理：顶点数越多 → 半径采样越独立 →
    边界越粗糙 → 每行的 RLE 段数越多。这里不应用平滑，因此该关系是单调的。
    """
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
    radii = rng.uniform(0.3, 1.0, num_vertices)
    pts_x = np.clip(
        (center_x + axis_x * radii * np.cos(angles)).astype(np.int32),
        0,
        image_width - 1,
    )
    pts_y = np.clip(
        (center_y + axis_y * radii * np.sin(angles)).astype(np.int32),
        0,
        image_height - 1,
    )
    pts = np.column_stack([pts_x, pts_y]).reshape(-1, 1, 2)
    canvas = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas.astype(bool)


def make_detections(
    num_objects: int,
    image_height: int,
    image_width: int,
    fill_fraction: float,
    num_vertices: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(xyxy, masks_dense, class_ids)``，包含随机多边形掩码。

    *num_vertices* 控制掩码复杂度：顶点数越多，边界越粗糙。
    """
    rng = np.random.default_rng(seed)
    half = max(
        2,
        int(
            (image_height * image_width * fill_fraction / (np.pi * num_objects)) ** 0.5
        ),
    )
    xyxy_list = []
    masks = np.zeros((num_objects, image_height, image_width), dtype=bool)
    for index in range(num_objects):
        center_x = int(rng.integers(half + 1, image_width - half - 1))
        center_y = int(rng.integers(half + 1, image_height - half - 1))
        axis_x = int(rng.integers(max(2, half // 2), half * 2 + 1))
        axis_y = int(rng.integers(max(2, half // 2), half * 2 + 1))
        masks[index] = _make_polygon_mask(
            image_height,
            image_width,
            center_x,
            center_y,
            axis_x,
            axis_y,
            rng,
            num_vertices,
        )
        xyxy_list.append(
            [
                max(0, center_x - axis_x),
                max(0, center_y - axis_y),
                min(image_width - 1, center_x + axis_x),
                min(image_height - 1, center_y + axis_y),
            ]
        )
    xyxy = np.array(xyxy_list, dtype=np.float32)
    class_ids = rng.integers(0, 10, num_objects, dtype=int)
    return xyxy, masks, class_ids


# ══════════════════════════════════════════════════════════════════════════════
# 内存辅助函数
# ══════════════════════════════════════════════════════════════════════════════


def dense_memory_bytes(masks: np.ndarray) -> int:
    """理论稠密占用：原始 numpy 缓冲区大小。"""
    return int(masks.nbytes)


def compact_memory_bytes_theoretical(compact_mask: CompactMask) -> int:
    """理论 compact 占用：所有内部 numpy 缓冲区大小之和。"""
    return int(
        compact_mask._crop_shapes.nbytes
        + compact_mask._offsets.nbytes
        + sum(rle.nbytes for rle in compact_mask._rles),
    )


def measure_peak_bytes(func: Callable[[], object]) -> int:
    """在 tracemalloc 下运行 *func* 并返回峰值分配字节数。

    tracemalloc 捕获每一处 Python 层的分配——numpy 缓冲区、列表节点、
    对象头——给出 *func* 所构建对象的真实堆开销。*func* 的返回值会被丢弃，
    以避免对象继续存活。
    """
    tracemalloc.start()
    tracemalloc.clear_traces()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def dense_memory_bytes_actual(
    num_objects: int, image_height: int, image_width: int
) -> int:
    """实际稠密占用：(N, H, W) 布尔数组分配过程中的峰值字节数。"""
    return measure_peak_bytes(
        lambda: np.zeros((num_objects, image_height, image_width), dtype=bool),
    )


def compact_memory_bytes_actual(
    masks_dense: np.ndarray,
    xyxy: np.ndarray,
    image_shape: tuple[int, int],
) -> int:
    """实际 compact 占用：CompactMask.from_dense() 过程中的峰值字节数。"""
    return measure_peak_bytes(
        lambda: CompactMask.from_dense(masks_dense, xyxy, image_shape=image_shape),
    )


def time_reps(
    func: Callable[[], object],
    repeats: int = REPETITIONS,
    parallel: int = PARALLEL,
) -> float:
    """运行 *func* 多次并返回每次调用的平均墙钟时间（秒）。

    当 ``parallel > 1`` 时，最多 ``parallel`` 个调用会在线程中并发执行。
    Numpy 和 OpenCV 会在其 C 层工作中释放 GIL，因此线程在多核机器上
    可以真正并发执行。每个线程记录自己的耗时；最终返回所有 *reps* 的
    平均值。

    当 ``parallel == 1`` 时使用原始的串行循环，规避线程调度开销，对
    廉价的函数有更好的测量精度。

    在计时前会运行一次完整的 GC，确保前面阶段累积的垃圾不会在测量中途
    被回收而抬高结果。
    """
    gc.collect()
    if parallel <= 1:
        t0 = time.perf_counter()
        for _ in range(repeats):
            func()
        return (time.perf_counter() - t0) / repeats

    def _timed() -> float:
        t0 = time.perf_counter()
        func()
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=min(parallel, repeats)) as pool:
        timings = list(pool.map(lambda _: _timed(), range(repeats)))
    return sum(timings) / repeats


# ══════════════════════════════════════════════════════════════════════════════
# 基准测试阶段
# ══════════════════════════════════════════════════════════════════════════════


def stage_build(
    num_objects: int,
    image_height: int,
    image_width: int,
    fill_fraction: float,
    num_vertices: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CompactMask]:
    """合成多边形掩码并构建 CompactMask。"""
    xyxy, masks_dense, class_ids = make_detections(
        num_objects, image_height, image_width, fill_fraction, num_vertices
    )
    compact_mask = CompactMask.from_dense(
        masks_dense, xyxy, image_shape=(image_height, image_width)
    )
    return xyxy, masks_dense, class_ids, compact_mask


def _resize_dense_to_shape(masks: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """将 (N, H, W) 布尔掩码使用最近邻缩放到 (N, new_h, new_w)。

    使用 floor-division 索引（``arange * src // dst``）与 ``_rle_resize`` 中
    的策略保持一致，从而在 :func:`stage_resize` 的正确性比较中实现像素级
    精确对等。
    """
    orig_h, orig_w = masks.shape[1], masks.shape[2]
    x = np.arange(new_w) * orig_w // new_w
    y = np.arange(new_h) * orig_h // new_h
    xv, yv = np.meshgrid(x, y)
    return masks[:, yv, xv]


def stage_encode(
    masks_dense: np.ndarray,
    xyxy: np.ndarray,
    image_height: int,
    image_width: int,
) -> float:
    """单掩码编码时间：逐个掩码编码并对 N 取平均。

    一次只对一个掩码调用 from_dense（而不是一次性批处理所有 N），可以隔离
    每个形状的开销——每个多边形的 RLE 段数不同，因此平均值反映了真实的
    形状差异。
    """
    num_masks = len(masks_dense)
    image_shape = (image_height, image_width)

    def _encode_each() -> None:
        for i in range(num_masks):
            CompactMask.from_dense(
                masks_dense[i : i + 1], xyxy[i : i + 1], image_shape=image_shape
            )

    return time_reps(_encode_each) / max(num_masks, 1)


def stage_decode(compact_mask: CompactMask) -> float:
    """单掩码解码时间：逐个掩码解码并对 N 取平均。

    通过 compact_mask[i] 构建列表的方式独立解码每个裁剪，
    从而给出将单个 RLE 物化为稠密数组的逐掩码开销。
    """
    num_masks = len(compact_mask)
    return time_reps(lambda: [compact_mask[i] for i in range(num_masks)]) / max(
        num_masks, 1
    )


def stage_area(
    det_dense: sv.Detections, det_compact: sv.Detections
) -> tuple[float, float]:
    """对两种表示分别计时 .area。"""
    return (
        time_reps(lambda: det_dense.area),
        time_reps(lambda: det_compact.area),
    )


def stage_filter(
    det_dense: sv.Detections, det_compact: sv.Detections
) -> tuple[float, float]:
    """对两种表示分别计时布尔索引过滤（每隔一个保留）。"""
    keep = np.arange(len(det_dense)) % 2 == 0
    return (
        time_reps(lambda: det_dense[keep]),
        time_reps(lambda: det_compact[keep]),
    )


def stage_annotate(
    scene: np.ndarray, det_dense: sv.Detections, det_compact: sv.Detections
) -> tuple[float, float]:
    """对两种表示分别计时 MaskAnnotator。"""
    annotator = sv.MaskAnnotator(opacity=0.5)
    return (
        time_reps(lambda: annotator.annotate(scene.copy(), det_dense)),
        time_reps(lambda: annotator.annotate(scene.copy(), det_compact)),
    )


def stage_correctness(
    scene: np.ndarray,
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    det_dense: sv.Detections,
    det_compact: sv.Detections,
) -> tuple[bool, bool, bool]:
    """返回 (pixel_perfect, areas_match, roundtrip_ok)。"""
    annotator = sv.MaskAnnotator(opacity=0.5)
    out_dense = annotator.annotate(scene.copy(), det_dense)
    out_compact = annotator.annotate(scene.copy(), det_compact)
    pixel_perfect = bool(np.array_equal(out_dense, out_compact))
    areas_match = bool(np.allclose(det_dense.area, det_compact.area))
    roundtrip_ok = bool(np.array_equal(compact_mask.to_dense(), masks_dense))
    return pixel_perfect, areas_match, roundtrip_ok


def stage_iou(
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    iou_dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """对成对自 IoU 进行计时：稠密路径使用 (N,H,W) AND，compact 路径使用裁剪过滤。

    为保持运行迅速，无论是否跳过完整稠密 IoU 计时，都仅在
    前 10 个掩码上检查正确性。
    """
    correct_n = min(len(compact_mask), 10)
    iou_compact_small = sv.mask_iou_batch(
        compact_mask[:correct_n], compact_mask[:correct_n]
    )
    iou_dense_small = sv.mask_iou_batch(
        masks_dense[:correct_n], masks_dense[:correct_n]
    )
    iou_ok = bool(np.allclose(iou_dense_small, iou_compact_small, atol=1e-4))

    compact_iou_s = time_reps(lambda: sv.mask_iou_batch(compact_mask, compact_mask))
    if iou_dense_skipped:
        dense_iou_s = math.nan
    else:
        dense_iou_s = time_reps(
            lambda: sv.mask_iou_batch(masks_dense, masks_dense),
            repeats=IOU_NMS_REPS,
        )
    return dense_iou_s, compact_iou_s, iou_ok


def stage_nms(
    xyxy: np.ndarray,
    confidence: np.ndarray,
    class_ids: np.ndarray,
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    dense_skipped: bool,
    iou_dense_skipped: bool,
) -> tuple[float, float, bool | None, int]:
    """对掩码 NMS 进行计时。稠密路径在 IoU 前会缩放到 640；compact 路径使用精确裁剪 IoU。

    Compact NMS 严格来说比稠密更准确：它在全分辨率 RLE 裁剪上直接计算像素级 IoU，
    而不是有损的 640 像素下采样近似。对于真实 IoU 接近 0.5 阈值的检测对，
    稠密路径的缩放步骤可能会翻转保留/抑制决策。

    ``n_diff`` 统计两条路径决策不同的检测数量。``nms_ok`` 在 ``n_diff == 0``
    时为 True。

    当 ``dense_skipped`` *或* ``iou_dense_skipped`` 为 True 时跳过稠密 NMS：
    NMS 内部调用 mask_iou_batch，开销与 IoU 相同。

    Returns:
        ``(dense_nms_s, compact_nms_s, nms_ok, n_diff)`` 元组。
    """
    predictions = np.c_[xyxy, confidence, class_ids.astype(float)]

    compact_nms_s = time_reps(
        lambda: sv.mask_non_max_suppression(predictions, compact_mask)
    )
    if dense_skipped or iou_dense_skipped:
        return math.nan, compact_nms_s, None, 0

    keep_dense = sv.mask_non_max_suppression(predictions, masks_dense)
    keep_compact = sv.mask_non_max_suppression(predictions, compact_mask)
    n_diff = int(np.sum(keep_dense != keep_compact))
    nms_ok = n_diff == 0
    dense_nms_s = time_reps(
        lambda: sv.mask_non_max_suppression(predictions, masks_dense),
        repeats=IOU_NMS_REPS,
    )
    return dense_nms_s, compact_nms_s, nms_ok, n_diff


def stage_merge(
    det_dense: sv.Detections | None,
    det_compact: sv.Detections,
    dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """对两个半切分进行 Detections.merge 计时。

    稠密路径：np.vstack；compact 路径：RLE 拼接。
    切分预先计算，因此被计时的 lambda 只测量 merge 本身。
    """
    half = len(det_compact) // 2
    compact_a, compact_b = det_compact[:half], det_compact[half:]

    compact_merge_s = time_reps(lambda: sv.Detections.merge([compact_a, compact_b]))
    if dense_skipped or det_dense is None:
        return math.nan, compact_merge_s, None

    dense_a, dense_b = det_dense[:half], det_dense[half:]
    merged_d = sv.Detections.merge([dense_a, dense_b])
    merged_c = sv.Detections.merge([compact_a, compact_b])
    merge_ok = bool(np.allclose(merged_d.area, merged_c.area))
    dense_merge_s = time_reps(lambda: sv.Detections.merge([dense_a, dense_b]))
    return dense_merge_s, compact_merge_s, merge_ok


def stage_offset(
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    image_height: int,
    image_width: int,
    dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """对掩码偏移进行计时：move_masks (N,H,W) 复制 vs O(N) 偏移更新。"""
    dx, dy = 10, 10
    # 通过偏移量扩展画布，避免任何偏移后的裁剪越界。
    # move_masks 和 with_offset.to_dense() 都在相同的空间上工作。
    new_h, new_w = image_height + dy, image_width + dx
    new_shape = (new_h, new_w)

    compact_offset_s = time_reps(
        lambda: compact_mask.with_offset(dx, dy, new_image_shape=new_shape)
    )
    if dense_skipped:
        return math.nan, compact_offset_s, None

    moved_dense = sv.move_masks(
        masks_dense, np.array([dx, dy]), resolution_wh=(new_w, new_h)
    )
    moved_compact = compact_mask.with_offset(
        dx, dy, new_image_shape=new_shape
    ).to_dense()
    offset_ok = bool(np.array_equal(moved_dense, moved_compact))
    dense_offset_s = time_reps(
        lambda: sv.move_masks(
            masks_dense, np.array([dx, dy]), resolution_wh=(new_w, new_h)
        )
    )
    return dense_offset_s, compact_offset_s, offset_ok


def stage_centroids(
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """对质心计算进行计时：稠密路径在完整堆上使用 np.tensordot，compact 路径按裁剪进行。"""
    compact_centroids_s = time_reps(lambda: sv.calculate_masks_centroids(compact_mask))
    if dense_skipped:
        return math.nan, compact_centroids_s, None

    c_dense = sv.calculate_masks_centroids(masks_dense)
    c_compact = sv.calculate_masks_centroids(compact_mask)
    # 1 像素容差
    centroids_ok = bool(np.allclose(c_dense, c_compact, atol=1.0))
    dense_centroids_s = time_reps(lambda: sv.calculate_masks_centroids(masks_dense))
    return dense_centroids_s, compact_centroids_s, centroids_ok


def stage_resize(
    masks_dense: np.ndarray,
    compact_mask: CompactMask,
    image_height: int,
    image_width: int,
    dense_skipped: bool,
) -> tuple[float, float, bool | None]:
    """对缩放到半分辨率进行计时，并检查像素级正确性。

    稠密路径使用 numpy 高级索引（``_resize_dense_to_shape``）。
    Compact 路径对 ``CompactMask.resize()`` 计时：对稀疏掩码（低于
    ``_L3_DENSITY_THRESHOLD``）使用直接 RLE 算术，对稠密掩码则回退到
    ``cv2.INTER_NEAREST`` 解码/缩放/重编码。两种最近邻策略在 bbox 边界
    上可能相差 1 像素，因此正确性检查使用 1 像素容差。
    """
    new_h, new_w = image_height // 2, image_width // 2
    new_shape = (new_h, new_w)

    # 使用 parallel=1 避免嵌套的 ThreadPoolExecutor 争用：
    # CompactMask.resize() 本身会为 N >= _PARALLEL_THRESHOLD 生成线程池，
    # 而 time_reps 自己的并发外层循环会导致过度订阅。
    compact_resize_s = time_reps(lambda: compact_mask.resize(new_shape), parallel=1)
    if dense_skipped:
        return math.nan, compact_resize_s, None

    resized_dense = _resize_dense_to_shape(masks_dense, new_h, new_w)
    resized_compact = compact_mask.resize(new_shape).to_dense()
    resize_ok = bool(
        np.abs(resized_dense.astype(np.int8) - resized_compact.astype(np.int8)).max()
        <= 1
    )
    dense_resize_s = time_reps(
        lambda: _resize_dense_to_shape(masks_dense, new_h, new_w)
    )
    return dense_resize_s, compact_resize_s, resize_ok



# ══════════════════════════════════════════════════════════════════════════════
# 场景运行器——协调各阶段
# ══════════════════════════════════════════════════════════════════════════════


def run_scenario(
    name: str,
    num_objects: int,
    image_height: int,
    image_width: int,
    fill_fraction: float = 0.10,
    num_vertices: int = 20,
) -> ScenarioResult:
    resolution = f"{image_width}x{image_height}"
    fill_name = f"{fill_fraction:.0%}"
    console.rule(
        f"[bold]{name}[/bold] | {num_objects} objects · {resolution} "
        f"· fill≈{fill_name} · polygon/{num_vertices} vertices"
    )

    xyxy, masks_dense, class_ids, compact_mask = stage_build(
        num_objects, image_height, image_width, fill_fraction, num_vertices
    )
    scene = make_scene(image_height, image_width)

    # ── 内存 ──────────────────────────────────────────────────────────────
    dense_bytes = dense_memory_bytes(masks_dense)
    dense_skipped = dense_bytes > DENSE_SKIP_GB * 1e9
    compact_theoretical = compact_memory_bytes_theoretical(compact_mask)

    # 仅在分配完整数组是安全时才测量稠密路径的 tracemalloc。
    dense_actual = (
        0
        if dense_skipped
        else dense_memory_bytes_actual(num_objects, image_height, image_width)
    )
    compact_actual = compact_memory_bytes_actual(
        masks_dense, xyxy, (image_height, image_width)
    )

    encode_s = stage_encode(masks_dense, xyxy, image_height, image_width)
    decode_s = stage_decode(compact_mask)

    theory_ratio = dense_bytes / max(compact_theoretical, 1)
    if dense_skipped:
        malloc_ratio_str = "[dim]—[/dim]"
        dense_actual_str = "[dim]skipped[/dim]"
    else:
        malloc_ratio = dense_actual / max(compact_actual, 1)
        malloc_ratio_str = _fmt_ratio(malloc_ratio)
        dense_actual_str = f"{dense_actual / 1e6:.1f} MB"
    console.print(
        f"\tmemory >>\n"
        f"\t\ttheory :: dense={dense_bytes / 1e6:.1f} MB "
        f"| compact={compact_theoretical / 1e3:.0f} KB "
        f"\t{_fmt_ratio(theory_ratio)}\n"
        f"\t\tmalloc :: dense={dense_actual_str} "
        f"| compact={compact_actual / 1e3:.0f} KB "
        f"\t{malloc_ratio_str}"
    )
    console.print(f"\t<create> encode (from_dense)\t={encode_s * 1e3:.3f} ms/mask")
    console.print(f"\t<export> decode (to_dense)\t={decode_s * 1e3:.3f} ms/mask")

    # ── skip 标志 ──────────────────────────────────────────────────────────
    iou_dense_skipped = dense_bytes > IOU_DENSE_SKIP_GB * 1e9
    if dense_skipped:
        console.print(
            f"\t[yellow]dense array is {dense_bytes / 1e9:.1f} GB "
            f"(>{DENSE_SKIP_GB:.0f} GB threshold) — skipping dense timing"
            f"[/yellow]"
        )
    elif iou_dense_skipped:
        console.print(
            f"\t[yellow]dense IoU skipped (>{IOU_DENSE_SKIP_GB:.0f}GB thr.)[/yellow]"
        )

    confidence = (
        np.random.default_rng(1).uniform(0.3, 0.99, num_objects).astype(np.float32)
    )
    det_compact = sv.Detections(xyxy=xyxy, mask=compact_mask, class_id=class_ids)

    if dense_skipped:
        dense_area_s = dense_filter_s = dense_annot_s = math.nan
        compact_area_s = _time_compact_area(det_compact)
        compact_filter_s = _time_compact_filter(det_compact)
        compact_annot_s = _time_compact_annotate(scene, det_compact)
        pixel_perfect = areas_match = roundtrip_ok = None
        det_dense = None
    else:
        det_dense = sv.Detections(xyxy=xyxy, mask=masks_dense, class_id=class_ids)
        dense_area_s, compact_area_s = stage_area(det_dense, det_compact)
        dense_filter_s, compact_filter_s = stage_filter(det_dense, det_compact)
        dense_annot_s, compact_annot_s = stage_annotate(scene, det_dense, det_compact)
        pixel_perfect, areas_match, roundtrip_ok = stage_correctness(
            scene, masks_dense, compact_mask, det_dense, det_compact
        )

    dense_iou_s, compact_iou_s, iou_ok = stage_iou(
        masks_dense, compact_mask, iou_dense_skipped
    )
    dense_nms_s, compact_nms_s, nms_ok, nms_diff = stage_nms(
        xyxy,
        confidence,
        class_ids,
        masks_dense,
        compact_mask,
        dense_skipped,
        iou_dense_skipped,
    )
    dense_merge_s, compact_merge_s, merge_ok = stage_merge(
        det_dense, det_compact, dense_skipped
    )
    dense_offset_s, compact_offset_s, offset_ok = stage_offset(
        masks_dense, compact_mask, image_height, image_width, dense_skipped
    )
    dense_centroids_s, compact_centroids_s, centroids_ok = stage_centroids(
        masks_dense, compact_mask, dense_skipped
    )
    dense_resize_s, compact_resize_s, resize_ok = stage_resize(
        masks_dense, compact_mask, image_height, image_width, dense_skipped
    )

    def _timing_line(label: str, dense_s: float, compact_s: float) -> str:
        compact_ms = f"{compact_s * 1e3:.2f} ms"
        if math.isnan(dense_s):
            return (
                f"\t{label}\t -> dense=[dim]—[/dim]"
                f"\t\t | compact={compact_ms}\t | speedup=[dim]—[/dim]"
            )
        dense_ms = f"{dense_s * 1e3:.2f} ms"
        speedup = _fmt_ratio(dense_s / max(compact_s, 1e-9))
        return (
            f"\t{label}\t -> dense={dense_ms}\t | "
            f"compact={compact_ms}\t | speedup={speedup}"
        )

    console.print(_timing_line(".area    ", dense_area_s, compact_area_s))
    console.print(_timing_line("annotate ", dense_annot_s, compact_annot_s))
    console.print(_timing_line("centroids", dense_centroids_s, compact_centroids_s))
    console.print(_timing_line("filter   ", dense_filter_s, compact_filter_s))
    console.print(_timing_line("iou      ", dense_iou_s, compact_iou_s))
    console.print(_timing_line("merge    ", dense_merge_s, compact_merge_s))
    console.print(_timing_line("nms      ", dense_nms_s, compact_nms_s))
    console.print(_timing_line("offset   ", dense_offset_s, compact_offset_s))
    console.print(_timing_line("resize   ", dense_resize_s, compact_resize_s))

    checks = {
        "pixel-perfect": pixel_perfect,
        "areas": areas_match,
        "roundtrip": roundtrip_ok,
        "iou": iou_ok,
        "nms": nms_ok,
        "merge": merge_ok,
        "offset": offset_ok,
        "centroids": centroids_ok,
        "resize": resize_ok,
    }
    parts = []
    for k, v in checks.items():
        if k == "nms" and v is False:
            parts.append(f"nms=[red]✗({nms_diff})[/red]")
        else:
            parts.append(
                f"{k}="
                + (
                    "[dim]—[/dim]"
                    if v is None
                    else "[green]✓[/green]"
                    if v
                    else "[red]✗[/red]"
                )
            )
    all_checked = [v for v in checks.values() if v is not None]
    overall = (
        "[green]✓ all correct[/green]"
        if all_checked and all(all_checked)
        else "[red]✗ MISMATCH[/red]"
        if any(v is False for v in checks.values())
        else "[dim]—[/dim]"
    )
    console.print("  correctness >> " + " | ".join(parts) + f" | {overall}")

    return ScenarioResult(
        name=name,
        resolution=resolution,
        num_objects=num_objects,
        fill_name=fill_name,
        num_vertices=num_vertices,
        dense_bytes=dense_bytes,
        compact_bytes_theoretical=compact_theoretical,
        dense_bytes_actual=dense_actual,
        compact_bytes_actual=compact_actual,
        encode_s=encode_s,
        decode_s=decode_s,
        dense_area_s=dense_area_s,
        compact_area_s=compact_area_s,
        dense_filter_s=dense_filter_s,
        compact_filter_s=compact_filter_s,
        dense_annot_s=dense_annot_s,
        compact_annot_s=compact_annot_s,
        dense_iou_s=dense_iou_s,
        compact_iou_s=compact_iou_s,
        dense_nms_s=dense_nms_s,
        compact_nms_s=compact_nms_s,
        dense_merge_s=dense_merge_s,
        compact_merge_s=compact_merge_s,
        dense_offset_s=dense_offset_s,
        compact_offset_s=compact_offset_s,
        dense_centroids_s=dense_centroids_s,
        compact_centroids_s=compact_centroids_s,
        pixel_perfect=pixel_perfect,
        areas_match=areas_match,
        roundtrip_ok=roundtrip_ok,
        iou_ok=iou_ok,
        nms_ok=nms_ok,
        nms_mismatch_count=nms_diff,
        merge_ok=merge_ok,
        offset_ok=offset_ok,
        centroids_ok=centroids_ok,
        dense_resize_s=dense_resize_s,
        compact_resize_s=compact_resize_s,
        resize_ok=resize_ok,
        dense_skipped=dense_skipped,
        iou_dense_skipped=iou_dense_skipped,
    )


def _time_compact_area(det_compact: sv.Detections) -> float:
    """对 compact 检测的 .area 计时（在稠密计时被跳过时使用）。"""
    return time_reps(lambda: det_compact.area)


def _time_compact_filter(det_compact: sv.Detections) -> float:
    """对 compact 检测的布尔索引过滤计时（dense-skip 路径）。"""
    keep = np.arange(len(det_compact)) % 2 == 0
    return time_reps(lambda: det_compact[keep])


def _time_compact_annotate(scene: np.ndarray, det_compact: sv.Detections) -> float:
    """对 compact 检测的 MaskAnnotator 计时（dense-skip 路径）。"""
    annotator = sv.MaskAnnotator(opacity=0.5)
    return time_reps(lambda: annotator.annotate(scene.copy(), det_compact))


# ══════════════════════════════════════════════════════════════════════════════
# Rich 汇总表
# ══════════════════════════════════════════════════════════════════════════════

_OPS = (
    "area",
    "filter",
    "annot",
    "iou",
    "nms",
    "merge",
    "offset",
    "centroids",
    "resize",
)


def _build_summary_df(results: list[ScenarioResult]) -> pd.DataFrame:
    """根据场景结果计算派生的汇总列。

    返回的 DataFrame 包含所有 ScenarioResult 字段以及派生列
    （比值、加速、ok 标志），均为原始 float 值。各列的具体格式化由
    调用方自行处理。
    """
    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    df["ratio_theory"] = df["dense_bytes"] / df["compact_bytes_theoretical"].clip(
        lower=1
    )
    df["ratio_malloc"] = df["dense_bytes_actual"] / df["compact_bytes_actual"].clip(
        lower=1
    )
    # dense_bytes_actual == 0（未测量）当 dense_skipped 时——将这些单元格清空
    df.loc[df["dense_skipped"], "ratio_malloc"] = None
    for op in _OPS:
        df[f"{op}_speedup"] = df[f"dense_{op}_s"] / df[f"compact_{op}_s"].clip(
            lower=1e-9
        )

    check_cols = [
        "pixel_perfect",
        "areas_match",
        "roundtrip_ok",
        "iou_ok",
        "nms_ok",
        "merge_ok",
        "offset_ok",
        "centroids_ok",
        "resize_ok",
    ]
    df["ok"] = df.apply(
        lambda row: (
            False
            if any(row[c] is False for c in check_cols)
            else True
            if any(row[c] is True for c in check_cols)
            else None
        ),
        axis=1,
    )
    return df


def _fmt_ratio(ratio: float) -> str:
    """使用颜色编码格式化加速/压缩比。

    ≥10 → 绿色（大赢），1-10 → 黄色（小幅赢），<1 → 红色（退步）。
    ≥10 显示为整数，否则保留两位小数。
    """
    fmt = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.2f}x"
    if ratio >= 10:
        return f"[green]{fmt}[/green]"
    elif ratio >= 1:
        return f"[yellow]{fmt}[/yellow]"
    else:
        return f"[red]{fmt}[/red]"


def _fmt_speedup(dense_s: float, compact_s: float) -> str:
    if math.isnan(dense_s):
        # 稠密被跳过——显示 compact 绝对时间，使该列不为空。
        return f"[dim]{compact_s * 1e3:.1f} ms[/dim]"
    return _fmt_ratio(dense_s / max(compact_s, 1e-9))


def print_summary(results: list[ScenarioResult]) -> None:
    table = Table(
        title="CompactMask — benchmark summary",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        min_width=console.width,
    )
    table.add_column("Scenario", style="bold", min_width=22)
    table.add_column("Objects", justify="right", min_width=7)
    table.add_column("Resolution", min_width=12, no_wrap=True)
    table.add_column("Fill", justify="right", min_width=5, no_wrap=True)
    table.add_column("Vertices", justify="right", min_width=8, no_wrap=True)
    table.add_column("Dense\ntheory", justify="right", min_width=10)
    table.add_column("Compact\ntheory", justify="right", style="green", min_width=9)
    table.add_column("Ratio\ntheory", justify="right", min_width=7)
    table.add_column("Dense\nmalloc", justify="right", style="cyan", min_width=9)
    table.add_column("Compact\nmalloc", justify="right", style="cyan", min_width=9)
    table.add_column("Ratio\nmalloc", justify="right", min_width=7)
    table.add_column("Encode\n(ms/mask)", justify="right", style="yellow", min_width=7)
    table.add_column("Decode\n(ms/mask)", justify="right", style="yellow", min_width=7)
    table.add_column("Area\natt.", justify="right", min_width=6)
    table.add_column("Filter\nop.", justify="right", min_width=6)
    table.add_column("Annot\nop.", justify="right", min_width=6)
    table.add_column("IoU\nop.", justify="right", min_width=6)
    table.add_column("NMS\nop.", justify="right", min_width=6)
    table.add_column("Merge\nop.", justify="right", min_width=6)
    table.add_column("Offset\nop.", justify="right", min_width=6)
    table.add_column("Resize\nop.", justify="right", min_width=6)
    table.add_column("Centr\nop.", justify="right", min_width=6)
    table.add_column("OK?", justify="center", min_width=4)

    for _, row in _build_summary_df(results).iterrows():
        ok = row["ok"]
        ok_cell = (
            "[red]✗[/red]"
            if ok is False
            else "[green]✓[/green]"
            if ok is True
            else "[dim]—[/dim]"
        )
        dense_malloc_cell = (
            "[dim]—[/dim]"
            if row["dense_skipped"]
            else f"{row['dense_bytes_actual'] / 1e6:.1f} MB"
        )
        malloc_ratio_cell = (
            "[dim]—[/dim]" if row["dense_skipped"] else _fmt_ratio(row["ratio_malloc"])
        )
        table.add_row(
            row["name"],
            str(row["num_objects"]),
            row["resolution"],
            row["fill_name"],
            str(row["num_vertices"]),
            f"{row['dense_bytes'] / 1e6:.1f} MB",
            f"{row['compact_bytes_theoretical'] / 1e3:.0f} KB",
            _fmt_ratio(row["ratio_theory"]),
            dense_malloc_cell,
            f"{row['compact_bytes_actual'] / 1e3:.0f} KB",
            malloc_ratio_cell,
            f"{row['encode_s'] * 1e3:.1f}",
            f"{row['decode_s'] * 1e3:.1f}",
            _fmt_speedup(row["dense_area_s"], row["compact_area_s"]),
            _fmt_speedup(row["dense_filter_s"], row["compact_filter_s"]),
            _fmt_speedup(row["dense_annot_s"], row["compact_annot_s"]),
            _fmt_speedup(row["dense_iou_s"], row["compact_iou_s"]),
            _fmt_speedup(row["dense_nms_s"], row["compact_nms_s"]),
            _fmt_speedup(row["dense_merge_s"], row["compact_merge_s"]),
            _fmt_speedup(row["dense_offset_s"], row["compact_offset_s"]),
            _fmt_speedup(row["dense_resize_s"], row["compact_resize_s"]),
            _fmt_speedup(row["dense_centroids_s"], row["compact_centroids_s"]),
            ok_cell,
        )

    console.print(table)
    console.print(
        "[dim]"
        + "  ·  ".join(
            [
                "Vertices — polygon vertex count "
                "(complexity proxy: more = jaggier boundary)",
                "Dense theory — NxHxW bytes (raw numpy buffer)",
                "Compact theory — sum of internal numpy buffer sizes",
                "Ratio (theory) — dense / compact theoretical ratio",
                "Dense malloc — tracemalloc peak during np.zeros allocation",
                "Compact malloc — tracemalloc peak during .from_dense()",
                "Ratio (malloc) — dense / compact tracemalloc peak ratio",
                "Encode ms/mask — from_dense() / N (dense→compact overhead per mask)",
                "Decode ms/mask — to_dense() / N (compact→dense overhead per mask)",
                "Area x — .area speedup (RLE sum, no materialisation)",
                "Filter x — boolean-index speedup",
                "Annot x — MaskAnnotator speedup (crop-paint vs full-frame alloc)",
                f"IoU x — pairwise self-IoU speedup "
                f"(dense skipped >{IOU_DENSE_SKIP_GB:.0f} GB)",
                "NMS x — mask_non_max_suppression speedup",
                "Merge x — Detections.merge speedup",
                "Offset x — move_masks vs with_offset speedup",
                "Resize x — resize-to-half speedup",
                "Centroids x — calculate_masks_centroids speedup",
                "dim ms — dense skipped, compact absolute time shown",
            ]
        )
        + "[/dim]"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 结果持久化
# ══════════════════════════════════════════════════════════════════════════════


def _append_result(result: ScenarioResult, path: Path) -> None:
    """将单个场景结果以 JSON Lines 格式追加到 *path*。

    ``math.nan``（用于被跳过的稠密计时）会被序列化为 ``null``，使文件
    保持合法 JSON-Lines，可被任何 JSON 解析器读取。
    """
    row = {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in dataclasses.asdict(result).items()
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def save_results_csv(results: list[ScenarioResult], path: Path) -> None:
    """将汇总表写入 *path*，格式为 CSV。

    每行对应 Rich 汇总表的一行：场景元数据、内存比值、编码/解码开销，
    以及每项操作的加速比。被跳过的稠密计时列写为空单元格。
    """
    df = _build_summary_df(results)
    pd.DataFrame(
        {
            "scenario": df["name"],
            "objects": df["num_objects"],
            "resolution": df["resolution"],
            "fill": df["fill_name"],
            "vertices": df["num_vertices"],
            "dense_theory_mb": (df["dense_bytes"] / 1e6).round(1),
            "compact_theory_kb": (df["compact_bytes_theoretical"] / 1e3).round(1),
            "ratio_theory": df["ratio_theory"].round(0),
            "dense_malloc_mb": (df["dense_bytes_actual"] / 1e6)
            .where(~df["dense_skipped"])
            .round(1),
            "compact_malloc_kb": (df["compact_bytes_actual"] / 1e3).round(1),
            "ratio_malloc": df["ratio_malloc"].round(0),
            "encode_ms_per_mask": (df["encode_s"] * 1e3).round(4),
            "decode_ms_per_mask": (df["decode_s"] * 1e3).round(4),
            **{f"{op}_speedup": df[f"{op}_speedup"].round(2) for op in _OPS},
            "resize_ok": df["resize_ok"],
            "ok": df["ok"],
        }
    ).to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 入口点
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # ── 参数矩阵 ──────────────────────────────────────────────────────────────
    # (tier_label, (image_width, image_height), num_objects)
    TIERS: list[tuple[str, tuple[int, int], int]] = [
        ("FHD", (1920, 1080), 100),  # 完整对比  (0.21 GB < 1 GB IoU 阈值)
        ("FHD", (1920, 1080), 200),  # 完整对比  (0.41 GB < 1 GB IoU 阈值)
        ("FHD", (1920, 1080), 400),  # 完整对比  (0.83 GB < 1 GB IoU 阈值)
        ("4K", (3840, 2160), 100),  # 完整对比  (0.83 GB < 1 GB IoU 阈值)
        ("4K", (3840, 2160), 200),  # 排除稠密 IoU/NMS  (1.66 GB > 1 GB 阈值)
        ("SAT", (8192, 8192), 200),  # 排除稠密 IoU/NMS  (13.4 GB > 1 GB 阈值)
    ]
    FILL_FRACTIONS = [0.05, 0.20, 0.50]  # 稀疏 / 中等 / 满填充
    VERTEX_COUNTS = [8, 128, 600]  # 低 / 现实 / YOLOv8-seg 默认

    scenarios = [
        {
            "name": f"{tier}-{num_objects}-{fill_fraction:.0%}-v{num_vertices}",
            "num_objects": num_objects,
            "image_height": img_h,
            "image_width": img_w,
            "fill_fraction": fill_fraction,
            "num_vertices": num_vertices,
        }
        for tier, (img_w, img_h), num_objects in TIERS
        for fill_fraction in FILL_FRACTIONS
        for num_vertices in VERTEX_COUNTS
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).parent / f"results_{timestamp}.jsonl"

    console.print(
        f"[bold]supervision[/bold]"
        f" {sv.__version__}  ·  numpy {np.__version__}  ·  {len(scenarios)} scenarios"
        f"  ·  saving to [dim]{results_path.name}[/dim]"
    )

    results = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("benchmarking…", total=len(scenarios))
        for params in scenarios:
            progress.update(task, description=f"[bold]{params['name']}[/bold]")
            result = run_scenario(**params)
            results.append(result)
            _append_result(result, results_path)
            gc.collect()  # 在下一轮运行前清空场景临时对象
            progress.advance(task)

    print_summary(results)

    csv_path = results_path.with_suffix(".csv")
    save_results_csv(results, csv_path)
    console.print(f"[dim]results saved → {results_path.name}  ·  {csv_path.name}[/dim]")


if __name__ == "__main__":
    main()
