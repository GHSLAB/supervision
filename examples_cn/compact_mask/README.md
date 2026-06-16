# CompactMask — 内存高效的掩码存储

本示例对 `CompactMask` 进行基准测试。`CompactMask` 是 `supervision` 引入的一种新的掩码表示形式，用基于裁剪的游程编码（RLE）取代稠密的 `(N, H, W)` 布尔数组。基准测试展示了完整的 API 兼容性、显著的内存节省以及数量级的标注加速——无需修改你现有的 `Detections` 代码即可使用。

---

## 问题

实例分割模型会为每个检测到的目标返回一个布尔掩码。`supervision` 将这些掩码存储为一个堆叠的 `(N, H, W)` numpy 数组。

对于一张 4K 图像和 1 000 个检测目标：

```
1 000 x 3840 x 2160 x 1 byte = 8.3 GB
```

在这种规模下，典型的流水线在单帧标注完成前就会因 `MemoryError` 而崩溃。航空影像、卫星瓦片以及高密度人群场景都会遇到这堵墙。

---

## 解决方案 — Crop-RLE 存储

`CompactMask` 将每个掩码存储为对其**边界框裁剪区域**而非整张图像画布的游程编码。

```
稠密 (N,H,W) 掩码   →   N × crop_RLE + N × (x1,y1) 偏移量
8.3 GB               →   ~280 KB
```

边界框已经存在于 `Detections.xyxy` 中，因此调用方无需提供额外的元数据。


### 理论分析（4K 场景，80x80 像素目标，每个 bbox 约 65% 填充）



整个 PR 设计分析中使用的假设：



| 参数                  | 取值                      |

| ---------------------- | ------------------------ |

| 图像尺寸               | 4K — 3840x2160 = 8.29 MP |

| 平均边界框             | 80x80 px = 6 400 px²     |

| bbox 内填充比例        | ~65%                     |

| 平均轮廓顶点数         | ~400 点                  |

| 平均每个掩码的 RLE 段数 | ~240（3 段 × 80 行）     |



#### 空间对比



| 格式              | 单目标       | N=100  | N=1 000    | 相对稠密   |

| ----------------- | ------------ | ------ | ---------- | ---------- |

| **Dense**（当前） | 8.29 MB      | 829 MB | **8.3 GB** | 1x         |

| Local Crop + Offset | 6.4 KB     | 640 KB | 6.4 MB     | 1 300x     |

| **Crop-RLE** ✓    | ~2 KB        | 200 KB | **2 MB**   | 4 000x     |

| Polygon ⚠ 有损     | ~3.2 KB     | 320 KB | 3.2 MB     | 2 600x     |

| memmap            | 8.29 MB（磁盘）| 829 MB | 8.3 GB   | 1x（磁盘） |



Crop-RLE 优于 Local Crop，因为前者只编码实际像素段，跳过了每个边界框内约 35% 的背景像素。




#### 编码时间：稠密数组 → 格式

| 格式              | 复杂度                                | N=10    | N=100   | N=1 000   |
| ----------------- | ------------------------------------- | ------- | ------- | --------- |
| Local Crop + Offset | O(A) — 基于 xyxy 的跨步切片        | ~0.1 ms | ~1 ms   | ~10 ms    |
| **Crop RLE**      | O(A) — 扫描裁剪行寻找段               | ~0.2 ms | ~2 ms   | ~20 ms    |
| Polygon           | O(P) — 在裁剪上调用 `cv2.findContours` | ~2 ms   | ~20 ms  | ~200 ms   |
| memmap            | O(I) — 将 8.29 MB 写入磁盘            | ~80 ms  | ~800 ms | ~8 000 ms |

#### 解码时间：格式 → 完整 (H, W) 掩码

`MaskAnnotator`、`mask_iou_batch`、`merge()` 等都需要此操作。在 4K 下，主要开销是**分配并将 8.29 MB 数组清零**，一旦需要完整物化，所有内存格式的开销相同。

| 格式                  | N=10   | N=100   | N=1 000   |
| --------------------- | ------ | ------- | --------- |
| Local Crop / Crop RLE | ~3 ms  | ~30 ms  | ~300 ms   |
| Polygon               | ~5 ms  | ~50 ms  | ~500 ms   |
| memmap                | ~80 ms | ~800 ms | ~8 000 ms |

#### 解码时间：仅裁剪路径（已优化）

当调用方只需要边界框区域时——`MaskAnnotator` 的裁剪绘制路径、`.area`、`contains_holes`、`filter_segments_by_distance`：

| 格式              | 复杂度                                  | N=10     | N=100   | N=1 000   |
| ----------------- | --------------------------------------- | -------- | ------- | --------- |
| Local Crop + Offset | O(1) — 已存储                       | ~0 ms    | ~0 ms   | ~0 ms     |
| **Crop RLE** ✓    | O(A) — 展开约 240 段                    | ~0.02 ms | ~0.2 ms | ~2 ms     |
| Polygon           | O(A) — 在裁剪画布上调用 `fillPoly`      | ~2 ms    | ~20 ms  | ~200 ms   |
| memmap            | N/A — 始终为完整尺寸                    | ~80 ms   | ~800 ms | ~8 000 ms |

Crop RLE 的 `.crop()` 方法支撑了 `MaskAnnotator` 的优化——它从不分配整张图像画布，这也是标注加速的根源。

#### IoU / NMS（1% bbox 重叠率，稀疏航空场景）

| 格式              | 策略                                        | N=1 000    |
| ----------------- | ------------------------------------------- | ---------- |
| Dense（当前）     | 全配对，640² 像素 AND                       | ~10 000 ms |
| Local Crop + Offset | bbox 预过滤 → 像素 IoU                    | **~5 ms**  |
| Crop RLE          | bbox 预过滤 → 展开交集                      | **~15 ms** |

在 1% 重叠的 N=1 000 情况下，bbox 预过滤将 499 500 个候选对减少为约 5 000 个重叠对——像素级工作量减少约 2 000 倍。

---

## 为什么选择 Crop-RLE 而非 Local Crop

两种格式都能极致压缩；选择 Crop-RLE 的决定因素是：

1. 对在自身边界框内本就稀疏的掩码，体积小约 **3 倍**。
2. **COCO RLE 互通**——Crop RLE 采用列优先（F-order）像素扫描，与 `pycocotools` 一致；如需互通，仍需基于裁剪范围的编码构造一个完整图像的 COCO RLE（例如将段补齐/合并到完整画布，或在完整图像中物化裁剪后重新编码）。
3. `.area` 直接由段长度计算得出——无需物化，无需分配。

主要权衡：仅裁剪的解码为 O(A) 而非 O(1)。对于常见的实心填充分割掩码，这个开销可以忽略不计（每个掩码 < 0.1 ms）。

---

## 逐操作加速分析

本节逐个分析每个涉及掩码的 `Detections` 操作，并解释 `CompactMask` 更快的原因。代码片段均取自实际实现。除特别说明外，数字基于 **FHD-200-50%-v600** 场景（1920 x 1080 图像，200 个检测目标，每个掩码覆盖约 50% 画面，600 顶点多边形——一个贴近实际的高填充、复杂边界的硬场景）。

在 50% 填充的 FHD 图像上，每个掩码的边界框覆盖画面的大部分，每行会产生大量 RLE 段。

---

### 内存

稠密路径为每个掩码存储一个完整分辨率的布尔数组：

```
N × H × W × 1 byte
200 × 1080 × 1920 × 1 = 414 MB
```

Compact 路径存储三个轻量结构：

```python
self._rles: list[npt.NDArray[np.int32]]  # N 个 Python 引用，指向较小的 int32 数组
self._crop_shapes: npt.NDArray[np.int32]  # (N, 2) — 每个掩码的裁剪 (h, w)
self._offsets: npt.NDArray[np.int32]  # (N, 2) — 每个掩码的 (x1, y1) 原点
```

在 50% 填充、600 顶点多边形的条件下，每个掩码的 RLE 大小约 4.7 KB（933 KB / 200）。每个掩码稠密大小：1920 × 1080 × 1 = 2.1 MB。单掩码比值：2.1 MB / 4.7 KB = **~445x**。

按 N=200 缩放：200 × 4.7 KB ≈ 933 KB 的 RLE 数据，外加 `_crop_shapes`（1.6 KB）和 `_offsets`（1.6 KB）。Python 列表与数组对象开销在 N 较小时大约会让整体占用翻倍。

| 组件           | Dense      | Compact     | 比值        |
| -------------- | ---------- | ----------- | ----------- |
| 掩码数据       | 414 MB     | ~933 KB     | ~445x       |
| Python 开销    | 可忽略     | ~933 KB     | --          |
| **总计**       | **414 MB** | **~1.9 MB** | **~392x**   |

在 5% 填充、8 顶点多边形条件下，由于裁剪更小、RLE 更短，比值可达 10 000x–20 000x。基准测试中的 4K-200-5%-v8 场景测得 21 786x（理论）/ ~6 000x（malloc）。SAT-200-5%-v8 场景的理论比值达到 62 968x。

---

### `.area`

稠密的 `Detections.area` 读取每个掩码的每个像素：

```python
# detection/core.py — 稠密路径
return np.array([np.sum(mask) for mask in self.mask])
# N 个掩码 × H × W 的布尔求和 = 200 × 210 万 = 4.2 亿次读取
```

Compact 委托给 `_rle_area`，它只对每个 RLE 中奇数下标位置的段长度（即 True 像素段）求和：

```python
# detection/compact_mask.py — _rle_area
return int(np.sum(rle[1::2]))
```

```python
# detection/compact_mask.py — CompactMask.area
return np.array([_rle_area(r) for r in self._rles], dtype=np.int64)
```

在 FHD-200-50%-v600 上，稠密 `.area` 用时 84.66 ms；Compact 用时 0.48 ms——**71 倍加速**。在 SAT-200-20%-v128 上，由于稠密数组高达 13.4 GB 且每次求和都必须扫描整个画布，测得的加速可达 **1 204 倍**。

| 因素                                | 减少量       |
| ----------------------------------- | ------------ |
| RLE 求和 vs 全帧像素读取            | ~4 600x      |
| int32 算术 vs 布尔归约              | ~2x          |
| 无需为每个掩码分配 (H, W)           | 时延         |
| **合计**                            | **~1 000x**  |

---

### `filter` / `__getitem__`（布尔索引）

稠密路径：`masks[bool_array]` 触发 NumPy 高级索引，会分配新的 `(K, H, W)` 布尔数组并复制 K 帧完整数据：

```python
# detection/core.py — Detections.__getitem__
mask = (self.mask[index] if self.mask is not None else None,)
# 对稠密 ndarray，numpy 会分配 (K, 2160, 3840) 并 memcpy K 帧
```

Compact 的 `CompactMask.__getitem__` 将布尔索引转换为整数位置，通过 Python 列表索引和在小 `(N, 2)` 数组上的 NumPy 高级索引构造新的 `CompactMask`：

```python
# detection/compact_mask.py — CompactMask.__getitem__
if isinstance(index, np.ndarray) and index.dtype == bool:
    idx_arr = np.where(index)[0]
# ...
new_rles = [self._rles[int(i)] for i in idx_arr]
new_crop_shapes: npt.NDArray[np.int32] = self._crop_shapes[idx_arr]
new_offsets: npt.NDArray[np.int32] = self._offsets[idx_arr]
return CompactMask(new_rles, new_crop_shapes, new_offsets, self._image_shape)
```

在 FHD-200-50%-v600 上，稠密 `filter` 用时 14.56 ms；Compact 用时 0.03 ms——**500 倍加速**。在 SAT-200-20%-v128 上加速可达 **14 757 倍**。

|              | 稠密                          | Compact                                |
| ------------ | ----------------------------- | -------------------------------------- |
| 复制数据     | K × H × W（完整帧）           | K 个 Python 引用 + K × 8 字节          |
| 分配         | 新的 `(K, H, W)` 数组         | 新的 `CompactMask` 外壳（开销极小）    |
| **加速**     |                               | **数百到数万倍**                       |

---

### `annotate`（`MaskAnnotator`）

稠密路径：对每个掩码，`MaskAnnotator` 索引完整的 `(H, W)` 数组并在整张画面上施加布尔掩码：

```python
# annotators/core.py — 稠密路径
mask = np.asarray(detections.mask[detection_idx], dtype=bool)
colored_mask[mask] = color.as_bgr()
```

对稠密数组而言，每次 `detections.mask[detection_idx]` 都会生成一个完整的 `(H, W)` 视图，布尔索引会扫描所有像素。

Compact 路径：标注器会检测 `CompactMask` 并仅在裁剪区域进行绘制：

```python
# annotators/core.py — compact 路径
x1 = int(compact_mask.offsets[detection_idx, 0])
y1 = int(compact_mask.offsets[detection_idx, 1])
crop_m = compact_mask.crop(detection_idx)
crop_h, crop_w = crop_m.shape
colored_mask[y1 : y1 + crop_h, x1 : x1 + crop_w][crop_m] = color.as_bgr()
```

`compact_mask.crop()` 将 RLE 解码为 `(crop_h, crop_w)` 数组。在 FHD-200-50%-v600 上，稠密 `annotate` 用时 848.95 ms；Compact 用时 32.67 ms——**22 倍加速**。在 SAT-200-20%-v128 上加速可达 **89 倍**。

| 因素                                              | 减少量               |
| ------------------------------------------------- | -------------------- |
| 每个掩码的裁剪解码 vs 全帧布尔索引                | 取决于裁剪尺寸       |
| 无需为每次整数索引分配完整的 `(H, W)`             | 时延                 |
| 乘以 N 个掩码                                     | 累积                 |
| **合计**                                          | **~26 – 400x**       |

---

### IoU（`mask_iou_batch` / `compact_mask_iou_batch`）

稠密 `mask_iou_batch`，N=200，FHD：

```python
# detection/utils/iou_and_nms.py — _mask_iou_batch_split
intersection_area = np.logical_and(masks_true[:, None], masks_detection).sum(
    axis=(2, 3)
)
# 形状为 (200, 200, 1080, 1920) — 约 800 亿次布尔操作
# .sum(axis=(2,3)) 用于求交集计数
# memory_limit 将其切分为不超过 5 GB 暂存的块
```

Compact `compact_mask_iou_batch` —— 三层优化：

**1. 向量化的 bbox 预过滤——O(N²) 数组操作，零解码**

```python
ix1: npt.NDArray[np.int32] = np.maximum(x1a[:, None], x1b[None, :])
iy1: npt.NDArray[np.int32] = np.maximum(y1a[:, None], y1b[None, :])
ix2: npt.NDArray[np.int32] = np.minimum(x2a[:, None], x2b[None, :])
iy2: npt.NDArray[np.int32] = np.minimum(y2a[:, None], y2b[None, :])
bbox_overlap: npt.NDArray[np.bool_] = (ix1 <= ix2) & (iy1 <= iy2)
```

在 5% 填充下，两个随机掩码重叠的概率约 4%。~96% 的 N² 配对会免费得到 IoU = 0——完全无像素操作。

**2. 子裁剪解码——只比较交集区域**

```python
ox_a, oy_a = int(x1a[i]), int(y1a[i])
sub_a = crops_a[i][ly1 - oy_a : ly2 - oy_a + 1, lx1 - ox_a : lx2 - ox_a + 1]

ox_b, oy_b = int(x1b[j]), int(y1b[j])
sub_b = crops_b[j][ly1 - oy_b : ly2 - oy_b + 1, lx1 - ox_b : lx2 - ox_b + 1]

inter = int(np.logical_and(sub_a, sub_b).sum())
```

两个重叠裁剪的交集子区域通常远小于完整画面。

**3. 裁剪缓存——每个掩码最多解码一次**

```python
if i not in crops_a:
    crops_a[i] = masks_true.crop(i)
```

面积由 `_rle_area`（奇数下标段求和）得到，完全不接触像素网格：

```python
areas_a: npt.NDArray[np.int64] = masks_true.area
```

在 FHD-200-50%-v600 上，稠密 IoU 用时 23 915 ms；Compact 用时 51.58 ms——**446 倍加速**。在 5% 填充/稀疏场景下，由于重叠的 bbox 配对更少，加速更大。

| 因素                                 | 减少量         |
| ------------------------------------ | -------------- |
| 稀疏填充下的 bbox 预过滤             | 25x            |
| 每对中子裁剪 vs 全帧                 | ~200x          |
| 面积由 RLE 而非 `sum(axis=(1,2))` 得到 | ~10x          |
| 无 5 GB 暂存分配                      | 时延           |
| **合计**                             | **~100 – 500x** |

在 20% 填充下，差距缩小——更多配对重叠、裁剪更大——加速会降至该范围的下限。

---

### NMS（`mask_non_max_suppression`）

稠密与 compact 路径现在都直接调用 `mask_iou_batch(masks, masks)`，在原始（未缩放）的掩码上计算精确的掩码 IoU。没有中间的缩放步骤。

```python
# detection/utils/iou_and_nms.py — NMS（两条路径）
ious = mask_iou_batch(masks, masks, overlap_metric)
```

`mask_iou_batch` 在内部分发：传入 `CompactMask` 时调用 `compact_mask_iou_batch`，应用全部三项 IoU 优化（bbox 预过滤、子裁剪解码、裁剪缓存）；传入稠密 ndarray 时则运行分块的像素 AND 路径。

全部三项 IoU 优化同时适用于 compact 路径：

| 因素                              | 减少量                      |
| --------------------------------- | --------------------------- |
| bbox 预过滤消除大多数配对         | 稀疏填充下 25x              |
| 对剩余配对进行子裁剪解码          | ~200x                       |
| 面积由 RLE 而非像素求和得到       | ~10x                        |
| **合计**                          | **与 IoU 相同：~100 – 500x** |

在 FHD-200-50%-v600 上，稠密 NMS 用时 5 231 ms；Compact 用时 48.15 ms——**481 倍加速**。在 4K-200 与 SAT-200 档位上，稠密 IoU/NMS 被跳过（> 1 GB）；Compact NMS 仍能正常运行。

---

### `merge`（`Detections.merge`）

稠密路径：`np.vstack` 分配新的 `(N1+N2, H, W)` 数组并复制两半：

```python
# detection/core.py — 稠密 merge 路径
return np.vstack([np.asarray(m) for m in masks])
# 在 FHD 上合并两个 100 掩码集合：2 × 100 × 2.1 MB = 414 MB 被复制
```

Compact 路径：`CompactMask.merge` 扩展一个 Python 列表并拼接两个小的 int32 数组：

```python
# detection/compact_mask.py — CompactMask.merge
new_rles: list[npt.NDArray[np.int32]] = []
for m in masks_list:
    new_rles.extend(m._rles)

new_crop_shapes: npt.NDArray[np.int32] = np.concatenate(
    [m._crop_shapes for m in masks_list], axis=0
)
new_offsets: npt.NDArray[np.int32] = np.concatenate(
    [m._offsets for m in masks_list], axis=0
)
```

`list.extend` 复制 N 个引用指针。`(N, 2)` int32 数组上的 `np.concatenate` 复制每个数组 N × 8 字节。

在 FHD-200-50%-v600 上，稠密 merge 用时 29.71 ms；Compact 用时 0.03 ms——**929 倍加速**。在 SAT-200-20%-v128 上加速可达 **89 046 倍**。

|              | 稠密                          | Compact                       |
| ------------ | ----------------------------- | ----------------------------- |
| 数据移动     | N × H × W（完整帧）           | N 个引用 + N × 8 字节         |
| 分配         | 新的 `(N, H, W)` 数组         | 新的 `CompactMask` 外壳       |
| **加速**     |                               | **实际上近乎免费**            |

**注意：** `Detections.merge` 会对每个输入调用 `is_empty()`。在加入 `len(xyxy) > 0` 的短路检查之前，`is_empty()` 会调用 `__eq__`，进而调用 `np.array_equal(self.to_dense(), ...)`——仅为检查空值就将整个 `(N, H, W)` 的 CompactMask 物化为稠密形式。修复方案：

```python
# detection/core.py — Detections.is_empty（已修复）
if len(self.xyxy) > 0:
    return False
```

这一 O(1) 检查避免了原先主导 compact merge 耗时的 O(N × H × W) 稠密物化。

---

### `offset` / `with_offset`（`InferenceSlicer` 瓦片拼接）

稠密的 `move_masks`：分配新的 `(N, new_H, new_W)` 数组并按偏移后的切片坐标复制每个掩码——O(N × H × W)：

```python
# detection/utils/masks.py — move_masks
mask_array = np.full((masks.shape[0], resolution_wh[1], resolution_wh[0]), False)
# ... 源/目标切片逻辑 ...
mask_array[:, dst_y1:dst_y2, dst_x1:dst_x2] = masks[:, src_y1:src_y2, src_x1:src_x2]
```

Compact 的 `with_offset(dx, dy)`：先做向量化的边界检查。全部新的边界框位置通过一次 numpy 操作计算完成。当没有任何一个越界时（`InferenceSlicer` 中的常见情况），RLE 数据完全不会被触及：

```python
# detection/compact_mask.py — CompactMask.with_offset（快速路径）
new_offsets = self._offsets + np.array([dx, dy], dtype=np.int32)  # O(N) numpy
needs_clip = (x1s < 0) | (y1s < 0) | (x2s >= new_w) | (y2s >= new_h)
if not needs_clip.any():
    return CompactMask(
        list(self._rles), self._crop_shapes.copy(), new_offsets, new_image_shape
    )
```

当某个裁剪确实越界（例如物体位于瓦片边缘）时，仅对该裁剪进行解码、切片并重新编码。完全位于边界外的掩码会得到一个 1×1 的全 False 占位，无需任何解码。

在 FHD-200-50%-v600 上，稠密 offset 用时 42.30 ms；Compact 用时 0.02 ms——**2 016 倍加速**。在 SAT-200-20%-v128 上加速可达 **290 779 倍**。

|                    | 稠密                                       | Compact（无裁剪快速路径）             |
| ------------------ | ------------------------------------------ | ------------------------------------- |
| 每个掩码的工作     | 分配 `(new_H, new_W)` 并复制 H × W         | 向偏移行加一个标量——O(1)              |
| N=200、FHD         | 200 × 2.1 MB = **414 MB** 分配并复制       | 对 `(N, 2)` int32 的两次 numpy 操作   |
| 输出分配           | 新的 `(N, new_H, new_W)`                    | 共享的 RLE 列表 + 新的 `(N, 2)` 数组  |
| **加速**           |                                            | **实际上近乎免费（> 1 000x）**        |

在 `InferenceSlicer` 流水线中，画布始终会按瓦片偏移进行扩展，因此不会出现裁剪越界——快速路径总是被命中。仅当物体真正跨越图像边界时才会触发裁剪。

---

### `centroids`（`calculate_masks_centroids`）

稠密路径：`np.tensordot` 读取每个掩码的每个像素来计算加权坐标和：

```python
# detection/utils/masks.py — 稠密质心路径
vertical_indices, horizontal_indices = np.indices((height, width)) + 0.5
# np.tensordot(masks, indices, axes=([1, 2], [0, 1]))
# 读取全部 N × H × W 的值
```

Compact 路径：在每个裁剪的循环中只解码边界框区域，并在该裁剪内计算质心：

```python
# detection/utils/masks.py — compact 质心路径
crop = masks.crop(i)
crop_h, crop_w = crop.shape
x1 = int(masks.offsets[i, 0])
y1 = int(masks.offsets[i, 1])
# ...
crop_rows, crop_cols = np.indices((crop_h, crop_w))
cx = float(np.sum((crop_cols + 0.5)[crop])) / total + x1
cy = float(np.sum((crop_rows + 0.5)[crop])) / total + y1
```

在 FHD-200-50%-v600 上，稠密 centroids 用时 1 133.68 ms；Compact 用时 60.39 ms——**13 倍加速**。在 SAT-200-20%-v128 上，由于稠密路径必须分配并扫描 13.4 GB 的数组，加速可达 **857 倍**。

| 因素                                      | 减少量               |
| ----------------------------------------- | -------------------- |
| 裁剪区域 vs 完整画面（每个掩码）          | 取决于填充度         |
| 无需全局 `np.indices((H, W))` 分配        | 节省大型 float64     |
| **合计（N=200）**                         | **~19 – 1 000x**     |

---

### 小结

在 **FHD-200-50%-v600** 工作点测得的加速（高填充、复杂多边形——一个贴近实际的硬场景）。稠密基线 = 1x。

| 操作            | 稠密开销   | Compact 开销 | 加速       |
| --------------- | ---------- | ------------ | ---------- |
| 内存            | 414 MB     | ~1.9 MB      | ~392x      |
| `.area`         | 84.66 ms   | 0.48 ms      | 71x        |
| `filter`        | 14.56 ms   | 0.03 ms      | 500x       |
| `annotate`      | 848.95 ms  | 32.67 ms     | 22x        |
| `mask_iou_batch` | 23 915 ms | 51.58 ms     | 446x       |
| NMS             | 5 231 ms   | 48.15 ms     | 481x       |
| `merge`         | 29.71 ms   | 0.03 ms      | 929x       |
| `with_offset`   | 42.30 ms   | 0.02 ms      | 2 016x     |
| `centroids`     | 1 133.68 ms | 60.39 ms    | 13x        |

在更稀疏的填充和更大的分辨率下，所有加速都会更大。在 SAT-200-20%-v128 上，`.area` 达到 1 204x，`merge` 达到 89 046x。在最稀疏的场景（5% 填充、8 顶点多边形）下，内存比值超过 60 000x。

---

## 即插即用兼容性

`CompactMask` 实现了与 `np.ndarray` 相同的鸭子类型接口：

```python
import supervision as sv
from supervision.detection.compact_mask import CompactMask

# 从现有的稠密 (N, H, W) 布尔数组构建：
compact = CompactMask.from_dense(masks_dense, xyxy, image_shape=(H, W))

# 完全像稠密掩码一样使用——其他代码无需任何修改：
detections = sv.Detections(xyxy=xyxy, mask=compact, class_id=class_ids)

# 过滤、合并、面积——全部透明工作：
filtered = detections[confidence > 0.5]
areas = detections.area  # RLE 求和，无需物化
merged = sv.Detections.merge([det_a, det_b])

# MaskAnnotator 无需任何修改即可工作：
annotated = sv.MaskAnnotator().annotate(frame, detections)

# 当你需要原始 numpy 时再物化为稠密：
dense_again = compact.to_dense()  # (N, H, W) bool
```

支持的索引模式：

| 表达式              | 返回值                          |
| ------------------- | ------------------------------- |
| `mask[i]`（int）    | 稠密的 `(H, W)` 布尔数组        |
| `mask[bool_array]`  | 新的 `CompactMask`（过滤后）    |
| `mask[slice]`       | 新的 `CompactMask`              |
| `np.asarray(mask)`  | 稠密的 `(N, H, W)` 布尔数组     |

---

## 基准测试

可在任何机器上运行——无需 GPU 或真实模型：

```bash
uv run python examples/compact_mask/benchmark.py
```

六档图像分辨率 × 三种填充比例（5 / 20 / 50%）× 三种顶点数（8 / 128 / 600）：

| 档位     | 分辨率    | 目标数 | 稠密数组 | 备注                              |
| -------- | --------- | ------ | -------- | --------------------------------- |
| FHD-100  | 1920x1080 | 100    | 0.21 GB  | 包含 IoU+NMS 的完整操作           |
| FHD-200  | 1920x1080 | 200    | 0.41 GB  | 包含 IoU+NMS 的完整操作           |
| FHD-400  | 1920x1080 | 400    | 0.83 GB  | 包含 IoU+NMS 的完整操作           |
| 4K-100   | 3840x2160 | 100    | 0.83 GB  | 包含 IoU+NMS 的完整操作           |
| 4K-200   | 3840x2160 | 200    | 1.66 GB  | 跳过稠密 IoU+NMS（数组 > 1 GB）   |
| SAT-200  | 8192x8192 | 200    | 13.4 GB  | 跳过稠密 IoU+NMS（数组 > 1 GB）   |

当稠密 IoU/NMS 数组将超过 1 GB 时，稠密计时会被自动跳过（`IOU_DENSE_SKIP_GB`），以避免换页抖动。所有稠密操作在超过 16 GB 时会被跳过（`DENSE_SKIP_GB`）；当前矩阵中没有场景达到该阈值。内存始终按理论 `N×H×W` 字节报告。

### 示例结果（macOS，Apple M4 Max，REPS=4）

| 场景                 | 稠密内存   | Compact 理论 | 内存 x  | 面积 x | 过滤 x | 标注 x | IoU x | NMS x | 合并 x  | 偏移 x   | 质心 x  |
| -------------------- | ---------- | ------------ | ------- | ------ | ------ | ------ | ----- | ----- | ------- | -------- | ------- |
| FHD-100-5%-v8        | 207 MB     | 28 KB        | 7 418x  | —      | —      | —      | —     | —     | —       | —        | —       |
| FHD-100-50%-v600     | 207 MB     | 913 KB       | 227x    | —      | —      | —      | —     | —     | —       | —        | —       |
| FHD-200-50%-v600     | 415 MB     | 933 KB       | 445x    | 71x    | 500x   | 22x    | 446x  | 481x  | 929x    | 2 016x   | 13x     |
| FHD-400-5%-v8        | 829 MB     | 60 KB        | 13 937x | —      | —      | —      | —     | —     | —       | —        | —       |
| 4K-100-5%-v8         | 829 MB     | 53 KB        | 15 554x | —      | —      | —      | —     | —     | —       | —        | —       |
| 4K-100-20%-v128      | 829 MB     | 586 KB       | 1 415x  | —      | —      | —      | —     | —     | —       | —        | —       |
| 4K-200-5%-v8         | 1 659 MB   | 76 KB        | 21 786x | —      | —      | —      | —     | —     | —       | —        | —       |
| SAT-200-5%-v8        | 13 422 MB  | 213 KB       | 62 968x | 6 942x | 30 255x | 204x  | †     | †     | 105 545x | 251 629x | 2 173x  |
| SAT-200-20%-v128     | 13 422 MB  | 2 596 KB     | 5 171x  | 1 204x | 14 757x | 89x   | †     | †     | 89 046x | 290 779x | 857x    |
| SAT-200-50%-v600     | 13 422 MB  | 14 222 KB    | 944x    | —      | —      | —      | —     | —     | —       | —        | —       |

- **Compact 理论** — 内部 numpy 缓冲区 `nbytes` 之和
- **内存 x** — 稠密 / Compact 理论内存比值
- **面积 x / 过滤 x / 标注 x / IoU x / NMS x / 合并 x / 偏移 x / 质心 x** — 各项操作上 Compact 相对稠密的加速比
- **†** — 稠密 IoU+NMS 跳过（稠密数组 > 1 GB）；Compact 仍运行并计时
- **—** — 不显示；完整的逐场景表格由基准测试脚本打印

所有未跳过的场景均通过：像素级完美的标注、精确的面积、无损的 `to_dense()` 往返。

---

## 使用场景

- **航空 / 卫星影像** — 大幅面瓦片上的数千个小目标；稠密掩码在推理完成前就会耗尽内存。
- **高密度人群 / 细胞分割** — 在 FHD 上 N > 500 时，每批次仅掩码存储就需要数 GB。
- **实时标注流水线** — 裁剪绘制将 4K 分辨率下的标注从秒级缩短到毫秒级。
- **长时间跟踪** — 跨多帧累积的 `Detections` 保持 KB 量级而非 GB 量级。
- **`InferenceSlicer`** — `with_offset()` 在拼接瓦片结果时直接调整裁剪原点；无需稠密物化。

---

## 局限性

- `CompactMask` **不是**完整的 `np.ndarray`。在传递给需要任意 ndarray 方法的代码之前，请先调用 `.to_dense()`（如 `astype`、`reshape`、`ravel`、`any`、`all` 等）。
- RLE 格式是**列优先（F-order）、基于裁剪范围**的——像素扫描顺序与 COCO / pycocotools 一致，但裁剪范围与全图范围不同。如需传递给 pycocotools，请先用 `.to_dense()` 物化全图稠密掩码，再将其编码为 COCO RLE。
- `from_dense()` 要求输入的 `(N, H, W)` 数组能够装入内存。对于真正 OOM 规模的数据，请直接从模型输出的每个目标的裁剪构建 `CompactMask`，而不是先分配稠密栈。

---

## 文件

| 文件           | 描述                                       |
| -------------- | ------------------------------------------ |
| `benchmark.py` | 跨 FHD / 4K / 卫星档位的完整基准测试       |
| `README.md`    | 本文件                                     |
