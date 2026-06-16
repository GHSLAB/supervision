<div align="center">
  <p>
    <a align="center" href="" target="https://supervision.roboflow.com">
      <img
        width="100%"
        src="https://media.roboflow.com/open-source/supervision/rf-supervision-banner.png?updatedAt=1678995927529"
      >
    </a>
  </p>

<br>

[notebooks](https://github.com/roboflow/notebooks) | [inference](https://github.com/roboflow/inference) | [autodistill](https://github.com/autodistill/autodistill) | [maestro](https://github.com/roboflow/multimodal-maestro)

<br>

[![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision) [![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision) [![license](https://img.shields.io/pypi/l/supervision)](LICENSE.md) [![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision) [![codecov](https://codecov.io/gh/roboflow/supervision/graph/badge.svg?token=HMNJ5FVZ36)](https://codecov.io/gh/roboflow/supervision)

[![snyk](https://snyk.io/advisor/python/supervision/badge.svg)](https://snyk.io/advisor/python/supervision) [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/supervision/blob/main/demo.ipynb) [![gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Roboflow/Annotators) [![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

<div align="center">
    <a href="https://trendshift.io/repositories/124"  target="_blank"><img src="https://trendshift.io/api/badge/repositories/124" alt="roboflow%2Fsupervision | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
  </div>

</div>

## 👋 你好

**我们是你不可或缺的计算机视觉工具箱。** 从数据加载到实时区域计数，我们提供完备的基础组件，让你能够专注于围绕模型构建应用。🤝

## 💻 安装

请在 [**Python>=3.9**](https://www.python.org/) 环境中通过 pip 安装 supervision 包。

```bash
pip install supervision
```

更多关于 conda、mamba 以及从源码安装的内容，请参阅我们的[指南](https://roboflow.github.io/supervision/)。

## 🔥 快速上手

### 模型

supervision 被设计为与具体模型无关。你可以接入任何分类、检测或分割模型。为了方便起见，我们为 Ultralytics、Transformers、MMDetection、Inference 等最流行的库提供了[连接器](https://supervision.roboflow.com/latest/detection/core/#detections)。其他集成（如 `rfdetr`）已经会直接返回 `sv.Detections`。

使用 `pip install pillow rfdetr` 安装本示例的可选依赖。

```python
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall

image = Image.open(...)
model = RFDETRSmall()
detections = model.predict(image, threshold=0.5)

len(detections)
# 5
```

<details>
<summary>👉 更多模型连接器</summary>

- inference

    使用 [Inference](https://github.com/roboflow/inference) 运行需要 [Roboflow API KEY](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)。

    ```python
    import supervision as sv
    from PIL import Image
    from inference import get_model

    image = Image.open(...)
    model = get_model(model_id="rfdetr-small", api_key="ROBOFLOW_API_KEY")
    result = model.infer(image)[0]
    detections = sv.Detections.from_inference(result)

    len(detections)
    # 5
    ```

</details>

### 标注器

supervision 提供了种类丰富、高度可定制的[标注器](https://supervision.roboflow.com/latest/detection/annotators/)，让你能针对自己的场景组合出完美的可视化效果。

```python
import cv2
import supervision as sv

image = cv2.imread(...)
detections = sv.Detections(...)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
```

https://github.com/roboflow/supervision/assets/26109316/691e219c-0565-4403-9218-ab5644f39bce

### 数据集

supervision 提供了一组[工具函数](https://supervision.roboflow.com/latest/datasets/core/)，可以让你以所支持的格式加载、拆分、合并和保存数据集。

```python
import supervision as sv
from roboflow import Roboflow

project = Roboflow().workspace("WORKSPACE_ID").project("PROJECT_ID")
dataset = project.version("PROJECT_VERSION").download("coco")

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/train",
    annotations_path=f"{dataset.location}/train/_annotations.coco.json",
)

path, image, annotation = ds[0]
# 按需加载图像

for path, image, annotation in ds:
    # 按需加载图像
    pass
```

<details close>
<summary>👉 更多数据集工具</summary>

- 加载

    ```python
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=...,
        annotations_directory_path=...,
        data_yaml_path=...,
    )

    dataset = sv.DetectionDataset.from_pascal_voc(
        images_directory_path=...,
        annotations_directory_path=...,
    )

    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=...,
        annotations_path=...,
    )
    ```

- 拆分

    ```python
    train_dataset, test_dataset = dataset.split(split_ratio=0.7)
    test_dataset, valid_dataset = test_dataset.split(split_ratio=0.5)

    len(train_dataset), len(test_dataset), len(valid_dataset)
    # (700, 150, 150)
    ```

- 合并

    ```python
    ds_1 = sv.DetectionDataset(...)
    len(ds_1)
    # 100
    ds_1.classes
    # ['dog', 'person']

    ds_2 = sv.DetectionDataset(...)
    len(ds_2)
    # 200
    ds_2.classes
    # ['cat']

    ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
    len(ds_merged)
    # 300
    ds_merged.classes
    # ['cat', 'dog', 'person']
    ```

- 保存

    ```python
    dataset.as_yolo(
        images_directory_path=...,
        annotations_directory_path=...,
        data_yaml_path=...,
    )

    dataset.as_pascal_voc(
        images_directory_path=...,
        annotations_directory_path=...,
    )

    dataset.as_coco(
        images_directory_path=...,
        annotations_path=...,
    )
    ```

- 转换

    ```python
    sv.DetectionDataset.from_yolo(
        images_directory_path=...,
        annotations_directory_path=...,
        data_yaml_path=...,
    ).as_pascal_voc(
        images_directory_path=...,
        annotations_directory_path=...,
    )
    ```

</details>

## 🎬 教程

想学习如何使用 Supervision？请查阅我们的[操作指南](https://supervision.roboflow.com/develop/how_to/detect_and_annotate/)、[端到端示例](./examples)、[速查表](https://roboflow.github.io/cheatsheet-supervision/)以及[实战手册](https://supervision.roboflow.com/develop/cookbooks/)！

<br/>

<p align="left">
<a href="https://youtu.be/hAWpsIuem10" title="使用计算机视觉进行停留时间分析 | 实时流处理"><img src="https://github.com/user-attachments/assets/014cffc7-72b3-4c0a-bb89-6de265b2c06b" alt="使用计算机视觉进行停留时间分析 | 实时流处理" width="300px" align="left" /></a>
<a href="https://youtu.be/hAWpsIuem10" title="使用计算机视觉进行停留时间分析 | 实时流处理"><strong>使用计算机视觉进行停留时间分析 | 实时流处理</strong></a>
<div><strong>创建时间：2024 年 4 月 5 日</strong></div>
<br/>学习如何利用计算机视觉分析等待时间并优化流程。本教程涵盖目标检测、跟踪以及计算目标在指定区域内停留的时间。这些技术可用于改善零售、交通管理等场景中的客户体验。</p>

<br/>

<p align="left">
<a href="https://youtu.be/uWP6UjDeZvY" title="车辆测速与跟踪 | 计算机视觉 | 开源"><img src="https://github.com/user-attachments/assets/b16b8e21-dc6c-4a73-a678-2f7d5d374793" alt="车辆测速与跟踪 | 计算机视觉 | 开源" width="300px" align="left" /></a>
<a href="https://youtu.be/uWP6UjDeZvY" title="车辆测速与跟踪 | 计算机视觉 | 开源"><strong>车辆测速与跟踪 | 计算机视觉 | 开源</strong></a>
<div><strong>创建时间：2024 年 1 月 11 日</strong></div>
<br/>学习如何结合 YOLO、ByteTrack 和 Roboflow Inference 来跟踪和估算车辆速度。本教程内容丰富，涵盖目标检测、多目标跟踪、检测过滤、透视变换、测速、可视化改进等主题。</p>

## 💜 使用 supervision 构建

你用 supervision 做出了什么很酷的东西吗？[告诉我们！](https://github.com/roboflow/supervision/discussions/categories/built-with-supervision)

https://user-images.githubusercontent.com/26109316/207858600-ee862b22-0353-440b-ad85-caa0c4777904.mp4

https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900

https://github.com/roboflow/supervision/assets/26109316/3ac6982f-4943-4108-9b7f-51787ef1a69f

## 📚 文档

请访问我们的[文档](https://roboflow.github.io/supervision)页面，了解 supervision 如何帮助你更快、更可靠地构建计算机视觉应用。

## 🏆 贡献

我们非常欢迎你的贡献！请查看我们的[贡献指南](.github/CONTRIBUTING.md)以开始参与。🙏 感谢所有贡献者！

<p align="center">
    <a href="https://github.com/roboflow/supervision/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=roboflow/supervision" />
    </a>
</p>

<br>

<div align="center">

<div align="center">
      <a href="https://youtube.com/roboflow">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://www.linkedin.com/company/roboflow-ai/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://docs.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://discuss.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584"
            width="3%"
          />
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://blog.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605"
            width="3%"
          />
      </a>
      </a>
  </div>
</div>
