# Sim2Real Prompt Annotation

面向 paired Sim/Real LeRobot 数据集的结构化 Prompt 自动标注工具。

Qwen VLM 负责从 Sim、Real、Reference 和 metadata 中提取结构化信息；Pydantic schema、
critic 和本地规则负责质量控制；最终训练 Prompt 由程序确定性渲染，不由 VLM 直接自由
生成。

```text
Sim + Real + Reference + Metadata
        → Structured Annotation
        → Critic / Validation
        → Canonical Annotation
        → full / reference / semantic / minimal
```

信息优先级为：

```text
Metadata > Direct Visual Observation > VLM Inference
```

## 数据格式

项目只支持 paired LeRobot episode。Sim 视频 key 以 `_sim` 结尾，对应 Real key 为去掉
`_sim` 后的同名 key：

```text
data/
  paired_task/
    meta/
      info.json
      episodes.jsonl
    labels/
      labels.json                       # optional
    videos/chunk-000/
      observation.images.camera_head/
        episode_000000.mp4              # Real
      observation.images.camera_head_sim/
        episode_000000.mp4              # Sim
```

视频路径由 `meta/info.json` 中的 `video_path` 和 `chunks_size` 解析。Reference 默认使用
Real `camera_head` 的第 0 帧。

## 安装

需要 Python 3.10+：

```bash
git clone <repository-url>
cd sim2real-prompt-annotation
python3 -m pip install -e .
cp config.example.yaml config.yaml
```

设置 Qwen API：

```bash
export DASHSCOPE_API_KEY='your-key'
export DASHSCOPE_BASE_URL='your-openai-compatible-endpoint'
```

API key 不应写入代码、YAML 或日志。默认模型为 `qwen3.7-plus`。

## 运行

编辑 `config.yaml` 中的数据根目录和参数，然后执行：

```bash
sim2real-prompt run \
  --config config.yaml \
  --dataset-glob 'paired_task_*'
```

可用 `--episodes 0,2,5-9` 或 `--limit N` 选择子集。默认支持断点续跑：完整样本会跳过，
失败样本会单独重试。仅在需要重新调用 API 时使用 `--force`。

检查当前输出是否完整：

```bash
sim2real-prompt audit \
  --config config.yaml \
  --dataset-glob 'paired_task_*'
```

## Python 接口

项目只公开 `PromptAnnotationPipeline`：

```python
from sim2real_prompt_annotation import PromptAnnotationPipeline

pipeline = PromptAnnotationPipeline("config.yaml")
result = pipeline.run(dataset_glob="paired_task_*", episodes="0-2")
audit = pipeline.audit(dataset_glob="paired_task_*")
```

主要方法：

- `run(...)`：运行可恢复批处理；
- `audit(...)`：离线检查输出完整性；
- `inspect(...)`：查看匹配的 LeRobot episode；
- `render(...)`：从 canonical annotation 重渲染 Prompt；
- `schemas()`：返回 annotation 和 critic JSON Schema。

## 输出

```text
outputs/
  annotations/
  annotations_raw/
  critiques/
  prompts/
    full/
    reference/
    semantic/
    minimal/
  logs/
    requests.jsonl
    completion_report.json
    incomplete_samples.jsonl
  failures.jsonl
```

最终训练 Prompt 同时聚合到每个源数据集：

```text
<dataset>/meta/episodes_prompt.jsonl
```

每行格式：

```json
{"episode_index":0,"prompts":{"full":"...","reference":"...","semantic":"...","minimal":"..."}}
```

部分 episode 运行只更新对应行，已有其他行会保留。`completion_report.json` 和
`incomplete_samples.jsonl` 表示当前完整性；`failures.jsonl` 是历史失败记录。

所有模型、抽帧、critic、重试、并发、renderer 和导出参数均集中在
`config.example.yaml`。YAML 相对路径以配置文件所在目录为基准。
