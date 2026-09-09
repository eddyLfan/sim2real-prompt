# sim2real-prompt

`sim2real-prompt` 是 Transfer 项目唯一的配对数据预处理仓库。数据读取保持
paired Sim/Real 数据集的身份和 split 约束，但预处理本身只消费配置的 **Real**
主视角；不读取 Sim 视频，也不依赖仓库外的 `data_processing/`。

每个未命中缓存的 episode 只有两条工作分支：

```text
Real camera_head video
  ├─ 均匀 8 帧 + 原始任务/机器人元数据
  │    └─ 单次 Qwen OpenAI-compatible VLM 请求
  │         ├─ 一句英文视频内容 prompt
  │         └─ 任务实体 reference_queries
  └─ 原分辨率首帧 + reference_queries
       └─ YOLOE-11s-seg → mask 质检/去重 → 候选池
            └─ sample_id + seed 确定性随机选择 1～3 张 crop
```

VLM 同一次响应同时给出 `prompt` 和 `reference_queries`，因此无需额外的本地任务
分词器或第二次 API 请求。任务物体与目标只能来自权威任务描述；为了让
Multi-Reference 同时覆盖物体与场景，VLM 也可从第一张图提取少量清晰可分割的机器人、
工作区、环境或背景区域。YOLOE 是唯一定位后端：无有效 segmentation mask、漏检
required 实体或没有 primary 任务物体时，该 episode 明确失败，不使用整帧或其他模型
兜底。

完整的模块边界、缓存键和逐文件职责见
[docs/architecture.md](docs/architecture.md)。

## 安装

Python 3.10+ 环境中安装主包以及 YOLOE 可选依赖：

```bash
cd /media/vlm/vlm-model/asset_llm_ckpt/vla-representation/project/yifan/Transfer/sim2real-prompt
python3 -m pip install -e '.[reference]'
cp config.example.yaml config.yaml
```

将官方 `yoloe-11s-seg.pt` 权重放到 `config.yaml` 中
`reference.model_path` 指向的位置。仓库不附带模型权重。

配置 Qwen 的 OpenAI-compatible endpoint；密钥不要写入 YAML、代码或日志：

```bash
export DASHSCOPE_API_KEY='your-key'
export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
```

如使用其他地域或内部兼容网关，以实际 endpoint 为准。也可以在 YAML 中设置
`prompt.provider.base_url`；API key 始终只从 `prompt.provider.api_key_env` 指定的
环境变量读取。

## 数据约束

每个被发现的数据集都必须满足以下条件：

- `meta/info.json` 显式包含 trim 后非空的 `source_id` 和 `domain`；目录名不会作为
  回退身份，且同一次发现中的 `source_id` 必须全局唯一。
- canonical `sample_id` 固定为
  `f"{len(source_id)}:{source_id}:{episode_index}"`，不做有损字符替换。
- `meta/episodes.jsonl` 中 `episode_index` 非负且唯一，`length` 默认至少 81，
  `tasks[0]` 是权威任务描述。
- 配置的 Real 视角默认是 `camera_head`，必须唯一存在；不会自动改用腕部视角。
- 每个 episode 必须能从 episode 自带 `split`、`meta/info.json:splits` 或
  `dataset.split_manifest` 的完整 domain assignment 中唯一解析为 `train` 或
  `validation`；多个来源同时存在时必须一致，任何冲突都会失败。
- 同一 domain 在一个或多个数据集中必须始终属于同一 split；
  `metadata_manifest` 不允许覆盖 split。

`split_manifest` 的 JSONL 格式为：

```json
{"domain":"lab_a","split":"train"}
{"domain":"lab_b","split":"validation"}
```

源数据内容或任务语义发生正式变更时，应分配新的 `source_id`，并重新生成缓存；
不要让同一个身份静默代表两版数据。

## 在指定小测试集上正式运行

以下命令直接使用原测试数据集，不创建软链接、镜像或临时 split。该数据集当前含
60 个 episode（12 个任务 × 5 个 episode），并已提供稳定的 `source_id`、`domain`
和 `train` split。

```bash
TEST_DATASET=/media/datasets/EWM_SIM_REAL_PAIRS/model_test/test_0905_agilex_cobotmagic2_12task_5episode

# 只读元数据预检：不解码视频、不调用 API、不加载 YOLOE。
sim2real-prompt inspect \
  --config config.yaml \
  --dataset "$TEST_DATASET" \
  --show 5

# 先用 episode 0 跑完整正式链路，并直接审计发布结果。
sim2real-prompt run \
  --config config.yaml \
  --dataset "$TEST_DATASET" \
  --episodes 0

sim2real-prompt audit \
  --config config.yaml \
  --dataset "$TEST_DATASET" \
  --episodes 0

# 单条确认后处理完整 60 条；已完成的两个分支会分别命中缓存。
sim2real-prompt run --config config.yaml --dataset "$TEST_DATASET"
sim2real-prompt audit --config config.yaml --dataset "$TEST_DATASET"
```

`run` 会直接在该测试数据集内发布训练产物，所以运行账号需要对数据集的
`Reference/` 和 `meta/` 有写权限。`--force` 会忽略有效缓存并重新调用 API/YOLOE；
普通续跑不要加它。

## CLI

CLI 只保留三个命令：

```bash
sim2real-prompt inspect --config config.yaml [--dataset PATH] [--episodes 0,2,5-9]
sim2real-prompt run     --config config.yaml [--dataset PATH] [--episodes 0,2,5-9]
sim2real-prompt audit   --config config.yaml [--dataset PATH] [--episodes 0,2,5-9]
```

- `inspect`：只读发现、身份/split/task/Real 视角预检。
- `run`：执行两个分支、保存独立 checkpoint、发布成功 episode，并默认做最终审计。
- `audit`：不需要 API key 或 YOLOE，校验 manifest 关联、1～3 张图片、路径、JPEG
  哈希、首帧来源和 primary Reference。

根目录不是单个 LeRobot 数据集时，可在配置或命令行设置 `dataset_glob`。`--limit`
适合快速抽取发现顺序中的前 N 条；`--episodes` 按 episode index 精确选择。

## 输出契约

Transfer 训练契约只包含以下三个产物（发布时还会在 `meta/` 使用隐藏锁文件与临时
transaction marker 协调并发写入）：

```text
<dataset>/
  Reference/
    episode_000000/
      reference_00.jpg
      reference_01.jpg
  meta/
    episodes_prompt.jsonl
    reference_images.jsonl
```

`meta/episodes_prompt.jsonl` 每个 episode 恰好包含：

```json
{"episode_index":0,"prompt":"The Agilex robot aligns the handles of the preassembled banana bunch in the same direction on a worktable under even indoor lighting.","reference_ids":["sha256:..."]}
```

`meta/reference_images.jsonl` 使用 schema v2，并只发布最终选中的 1～3 张图；候选池
只留在续跑 checkpoint 中：

```json
{"schema_version":2,"episode_index":0,"references":[{"reference_id":"sha256:...","reference_path":"Reference/episode_000000/reference_00.jpg","source_view":"camera_head","source_frame_index":0,"query":"banana","label":"banana","role":"primary","scope":"objects","confidence":0.83,"bbox_xyxy":[677.0,368.0,820.0,498.0],"crop_xyxy":[660,352,838,514],"sha256":"...","provenance":{"backend":"yoloe","model":"yoloe-11s-seg.pt","selection_seed":42}}]}
```

`episodes_prompt.jsonl.reference_ids` 的内容与顺序必须和同 episode 的
`reference_images.jsonl.references[*].reference_id` 完全相同。Reference ID 是最终
JPEG 字节的 SHA-256，图片固定来自 Real 首帧。

可恢复的中间状态写到 `output.root`，默认是仓库内的 `outputs/`：

```text
outputs/
  prompt/<hash-prefix>/<sample-hash>.json
  reference/<hash-prefix>/<sample-hash>.json
  reference_images/<hash-prefix>/<sample-hash>/<cache-key>/reference_XX.jpg
  run_report.json
```

这三类是 checkpoint/报告，不是训练数据契约。

## 性能、失败与续跑

- Prompt 未缓存时，每个视频只打开一次，并直接 seek 到包含首尾的 8 个均匀位置，
  不解码中间无用帧；原始 BGR 首帧直接复用给 Reference，不经过有损 JPEG 往返。
  解码由 `runtime.decode_workers` 并发执行。
- API 请求由 `runtime.api_concurrency` 控制并发。结构化响应错误、质检错误、超时、
  限流和服务端错误按 `runtime.api_retry_count` 指数退避重试。
- YOLOE 按完整 query signature 分组做 GPU batch，常驻复用 MobileCLIP 文本编码器，并用
  有界 LRU 缓存重复词表的 embedding。首次文本提示运行会按 Ultralytics 机制准备额外的
  MobileCLIP 权重和 tokenizer，因此应在批处理节点联网预热一次，再进入离线大批处理。
- Prompt 和 Reference 使用不同 fingerprint/checkpoint。若 Prompt 已缓存而 Reference
  缺失，只解码首帧；查询、权重、视频或相关配置改变时只失效受影响的分支。
- `runtime.fail_fast: false` 时，单条失败会记录在 `run_report.json`，其他 episode 继续；
  命令以非零状态结束并报告 `partial`。失败条目不生成伪 Reference，也不覆盖成功条目。
- 发布采用原子写、数据集级文件锁和 transaction marker；子集运行始终 merge，不会
  删除 manifest 中未选中的旧 episode。`status` 表示本次 selection 是否成功，
  `annotations_ready` 只有在全量 annotation manifest 与所有源 episode 精确一致且审计
  通过时才为 true；它不替代 Transfer 对视频、Parquet、关节映射和训练配置的完整预检。
  中断后重跑必须覆盖 marker 记录的全部 episode，Transfer reader 也会拒绝中断代次。

## Python API

高层接口只有 `Sim2RealPreprocessingPipeline`，示例见
[examples/python_api.py](examples/python_api.py)：

```python
from sim2real_prompt_annotation import Sim2RealPreprocessingPipeline

pipeline = Sim2RealPreprocessingPipeline(
    "config.yaml",
    dataset_root="/path/to/lerobot-dataset",
)
print(pipeline.inspect(show=3))
report = pipeline.run(episodes="0,2,5-9")
print(report["status"])
print(pipeline.audit(episodes="0,2,5-9"))
```

## 许可证

本仓库自身代码为 Apache-2.0。可选 Reference 依赖 Ultralytics/YOLOE 涉及
AGPL-3.0，商业使用可另行取得商业许可；源码和模型权重均未 vendored。使用者负责
获取权重并遵守适用条款，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
