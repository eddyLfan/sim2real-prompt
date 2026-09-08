# Sim2Real Multi-Reference Data Pipeline

本仓库是 Transfer 项目唯一的 paired Sim/Real LeRobot v2.1 数据预处理入口，统一完成：

1. 数据与元数据完整性检查；
2. Sim/Real 多视角时序抽帧；
3. Real 主视角首帧的语义区域定位；
4. 任务物体、机器人、工作区、环境与背景子图导出；
5. 单句自然语言视频描述生成与本地质检；
6. Multi-Reference manifest、Prompt 和审计报告写入。

```text
paired dataset validation
  -> Sim/Real temporal evidence + Real main-view frame 0
  -> one structured Annotation API call
  -> semantic bbox validation and tight crops
  -> one-sentence video caption
  -> atomic Multi-Reference/Prompt export
  -> completion audit
```

## Multi-Reference 约定

Annotation API 在归一化 `[0,1000]` 坐标中为每个可独立使用的视觉信息生成 bbox：

- 每个可见任务物体分别生成一个 `objects` crop；
- 可生成 `robot`、`workspace`、`environment`、`background` crop；
- crop 默认保留 12% 边缘上下文；
- 低置信度、尺寸小于 16 px 或无有效候选的样本不会回退到整帧；
- 每个 episode 最多保存 8 个候选，可在配置中调整到 1～12 个。

训练数据集从候选池随机 dropout，并保证每次实际输入 1～3 张；存在任务物体候选时至少保留
一个 `objects` crop。验证和推理确定性读取 manifest 中排序后的前 3 张。模型按照
Phantom-Wan 的方式，将每张图片分别通过 VAE 编码，
再沿 latent 时间维拼接。

数据集最终产物为：

```text
<dataset>/
  Reference/
    episode_000000/
      reference_00.jpg
      reference_01.jpg
  meta/
    reference_images.jsonl
    episodes_prompt.jsonl
```

`reference_images.jsonl` 每个 episode 包含完整候选池：

```json
{"schema_version":2,"episode_index":0,"reference_seed":42,"references":[{"reference_id":"sha256:...","reference_path":"Reference/episode_000000/reference_00.jpg","source_view":"camera_head","source_frame_index":0,"scope":"objects","label":"mug","description":"a green ceramic mug","bbox_xyxy":[240,310,430,690],"crop_xyxy":[120,160,240,350],"confidence":0.96,"sha256":"..."}]}
```

`episodes_prompt.jsonl` 使用 `reference_ids` 与图片表建立严格关联：

```json
{"episode_index":0,"prompt":"The Agilex robot uses its left arm to place a green ceramic mug onto a black coaster in a scene with a white tabletop, gray partitions, and diffuse overhead lighting.","reference_ids":["sha256:..."]}
```

## Prompt

最终 Prompt 是一句自然的视频内容描述，包含机器人本体、任务动作、物体与目标关系、环境和
光照。它不再包含 `Match ... to the reference image`、`Render the scene ...` 等训练机制说明。
任务语义严格来自权威任务元数据；视觉字段只提供外观描述和 Reference crop，不得增加动作、
目标或完成约束。

结构化 annotation 仍完整保留，用于审计和重新渲染；最终文案由本地 renderer 确定性生成，
不会由 API 自由输出。

## 安装与配置

```bash
python3 -m pip install -e .
cp config.example.yaml config.yaml
export DASHSCOPE_API_KEY='your-key'
export DASHSCOPE_BASE_URL='your-openai-compatible-endpoint'
```

API key 不应写入代码、YAML 或日志。

关键配置：

```yaml
media:
  reference_view: camera_head
  reference_seed: 42
  reference_crop_padding: 0.12
  reference_min_confidence: 0.55
  reference_pool_max_images: 8
```

## 使用

完整检查并处理一个数据集：

```bash
sim2real-prompt process --dataset /path/to/dataset --config config.yaml
```

只读检查：

```bash
sim2real-prompt process --dataset /path/to/dataset --check-only
```

批量运行、断点续跑与审计：

```bash
sim2real-prompt run --config config.yaml --dataset-glob 'paired_task_*'
sim2real-prompt audit --config config.yaml --dataset-glob 'paired_task_*'
```

`run` 会同时生成 Reference crops 和 Prompt，不再要求预先执行 Reference 导出。已有 canonical
annotation 默认断点续跑；`--force` 会重新调用 API。`references` 命令仅用于根据已有 canonical
annotation 重建 crop 文件。

中间产物默认位于 `sim2real-prompt/outputs/<dataset_name>/`，包括 annotations、references、
validations、prompts、logs 和 `data_processing_report.json`；源数据集内只写训练直接需要的
Reference 图片与两个 JSONL 表。

## Python 接口

```python
from sim2real_prompt_annotation import DatasetProcessingPipeline

pipeline = DatasetProcessingPipeline("/path/to/dataset", config="config.yaml")
report = pipeline.run()
```
