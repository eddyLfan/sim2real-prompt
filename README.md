# Sim2Real Prompt Annotation

本工具是 Wan2.2 Multi-View Sim-to-Real Transfer 项目专用的 Prompt 标注 pipeline。
每个 paired LeRobot episode 只生成两项核心结果：

1. 一份详细、可审计的结构化标注；
2. 一个用于训练的精简 Prompt。

```text
Sim + Real + same-episode Reference + Metadata
        → Detailed Structured Annotation
        → Critic / Local Validation
        → Canonical Annotation
        → One Compact Prompt
```

## 条件职责

- Sim 和 RobotState 控制机器人与物体运动、接触、状态变化、空间关系、相机视角、构图和时序；
- Prompt 描述高层任务语义及可编辑的粗粒度目标外观；
- Reference 补充随机帧中实际可见的精细机器人、物品、工作台、背景和光照细节；
- Real Video 是目标域标注证据和训练监督。

外观冲突时遵循：

```text
Prompt 显式属性 > Reference 外观 > Sim 外观
```

结构化标注保留 Sim invariants、任务物体语义角色、几何/affordance、目标 Real
外观、Reference 可见范围、证据与置信度。最终 Prompt 不包含轨迹、动作阶段、逐帧状态、
相机参数或冗余质量口号。

典型 Prompt：

```text
Real-world video of a dual-arm robot placing a blue ceramic mug on a white tray. Setting: a gray workbench; a robotics laboratory; soft overhead lighting. Use the reference for fine robot, object, workspace, and background details; explicit text attributes take priority.
```

默认最多 55 个英文词。超限会触发验证失败和重试，不会静默截断。

## 数据格式

只支持 paired LeRobot episode。Sim 视频 key 以 `_sim` 结尾，对应 Real key 为去掉
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

Reference 从同 episode 的 `reference_view` Real 视频中按 `reference_seed` 可复现地
随机选择一帧。不进行最佳帧搜索。实际视角和帧号写入 canonical annotation 与
`episodes_prompt.jsonl`，供训练侧使用同一张图。

## 安装与配置

需要 Python 3.10+：

```bash
python3 -m pip install -e .
cp config.example.yaml config.yaml
```

设置 Qwen OpenAI-compatible API：

```bash
export DASHSCOPE_API_KEY='your-key'
export DASHSCOPE_BASE_URL='your-openai-compatible-endpoint'
```

API key 不应写入代码、YAML 或日志。YAML 相对路径以配置文件所在目录为基准。

## 运行

```bash
sim2real-prompt run \
  --config config.yaml \
  --dataset-glob 'paired_task_*'
```

可用 `--episodes 0,2,5-9` 或 `--limit N` 选择子集。默认断点续跑；仅在需要重新调用
API 时使用 `--force`。

检查输入及随机 Reference 帧：

```bash
sim2real-prompt run \
  --config config.yaml \
  --dataset-glob 'paired_task_*' \
  --dry-run --prepare-media
```

审计当前输出：

```bash
sim2real-prompt audit \
  --config config.yaml \
  --dataset-glob 'paired_task_*'
```

从 canonical annotation 重新渲染 Prompt：

```bash
sim2real-prompt render \
  --config config.yaml \
  --annotation outputs/annotations/paired_task__episode_000000.json
```

## Python 接口

```python
from sim2real_prompt_annotation import PromptAnnotationPipeline

pipeline = PromptAnnotationPipeline("config.yaml")
result = pipeline.run(dataset_glob="paired_task_*", episodes="0-2")
audit = pipeline.audit(dataset_glob="paired_task_*")
```

## 输出

```text
outputs/
  annotations/           # canonical detailed annotation
  annotations_raw/
  critiques/
  prompts/               # one .txt per episode
  logs/
    requests.jsonl
    completion_report.json
    incomplete_samples.jsonl
  failures.jsonl
```

训练 Prompt 聚合到每个源数据集：

```text
<dataset>/meta/episodes_prompt.jsonl
```

每行格式：

```json
{"episode_index":0,"prompt":"Real-world video of ...","reference_view":"camera_head","reference_frame_index":42}
```

部分 episode 运行只替换对应行，其他已有行会保留。旧版
`{"prompts":{"full":...}}` 多版本格式不再兼容。
