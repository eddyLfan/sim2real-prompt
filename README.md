# sim2real-prompt

`sim2real-prompt` 是 Transfer 项目唯一的配对数据预处理仓库。它保留 paired
Sim/Real LeRobot 数据集的身份和 split 约束，但两条标注分支都只读取配置的 Real
主视角（默认 `camera_head`）；不读取 Sim 视频，也不依赖仓库外的
`data_processing/`。

每个 episode 的固定数据流是：

```text
Real camera_head video
  ├─ 8 个均匀帧 + 原始任务/机器人元数据
  │    └─ 一次 OpenAI-compatible API VLM 请求 ──> 一句英文视频 prompt
  └─ 原分辨率 frame 0
       └─ RobotSeg whole-robot mask
            └─ closing + dilation
                 └─ Big-LaMa inpainting
                      └─ YOLOE residual-robot QA
                           └─ 一张完整 robot_removed_scene Reference
```

Prompt 和 Reference 有独立 fingerprint、成功 checkpoint 和失败状态。Reference
不依赖 VLM 输出或任务分词；Prompt 命中缓存不会阻塞 Reference，反之亦然。只有两条
分支都成功的 episode 才会 join 并原子发布给 Transfer。

Reference 固定保留原图宽高与环境、桌面、任务物体、背景和光照，只去除机器人本体。
它不是物体 crop。RobotSeg 无有效 whole-robot mask、mask 面积异常、Big-LaMa 变化不足、
残留机器人超阈值或任一模型不可用时都 fail closed：不发布原首帧、局部 crop、OpenCV
填充结果或其他分割模型兜底。

完整模块边界、缓存键和逐文件职责见
[docs/architecture.md](docs/architecture.md)。

## 安装与生产模型

基础包和 YOLOE/OmegaConf 适配依赖：

```bash
cd /media/vlm/vlm-model/asset_llm_ckpt/vla-representation/project/yifan/Transfer/sim2real-prompt
python3 -m pip install -e '.[reference]'
cp config.example.yaml config.yaml
```

生产 Reference 还需要在同一运行环境中安装官方
[RobotSeg](https://github.com/showlab/RobotSeg) 与
[LaMa](https://github.com/advimman/lama) runtime。RobotSeg 的 CUDA/PyTorch 组合应按其
官方安装说明构建；LaMa 源码目录必须能提供 `saicinpainting` import。仓库不会联网安装
它们，也不会自动下载权重。

在 `config.yaml` 中设置以下四个本地模型资产：

| 配置 | 需要的本地产物 | 用途 |
| --- | --- | --- |
| `reference.model_path` | RobotSeg 的 `robotseg.pt` | frame 0 whole-robot mask |
| `reference.inpainting_model_path` | `config.yaml` + `models/best.ckpt` 的 Big-LaMa 目录，或兼容 TorchScript 文件 | 填补扩张后的 robot mask |
| `reference.yoloe_model_path` | `yoloe-11s-seg.pt` | 仅做 inpaint 后残留机器人 QA |
| `reference.yoloe_text_model_path` | 与 Ultralytics 版本匹配的 `mobileclip_blt.ts` | YOLOE 本地文本编码器 |

正式批处理必须填写四个对应的 SHA-256 配置。Big-LaMa 目录的摘要对应
`models/best.ckpt`。所有资产都应在批处理节点提前准备并完成离线预热；pipeline 不允许
Ultralytics 隐式下载 MobileCLIP。源码和模型权重均不纳入 Git。

配置 API VLM 的 OpenAI-compatible endpoint；密钥不要写入 YAML、代码或日志：

```bash
export DASHSCOPE_API_KEY='your-key'
export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
```

如使用其他地域或内部兼容网关，以实际 endpoint 为准。也可以在 YAML 中设置
`prompt.provider.base_url`；API key 始终只从 `prompt.provider.api_key_env` 指定的
环境变量读取。

## 数据约束

每个被发现的数据集都必须满足：

- `meta/info.json` 显式包含 trim 后非空的 `source_id` 和 `domain`；同次发现中的
  `source_id` 全局唯一。
- canonical `sample_id` 固定为
  `f"{len(source_id)}:{source_id}:{episode_index}"`，不做有损字符替换。
- `meta/episodes.jsonl` 中 `episode_index` 非负且唯一，`length` 默认至少 81，
  `tasks[0]` 是权威任务描述。
- 配置的 Real 视角必须唯一存在；不会自动切换到腕部视角。
- episode split 必须由行内 `split`、`meta/info.json:splits` 或
  `dataset.split_manifest` 唯一确定；多来源同时存在时必须一致。
- 同一 domain 在一个或多个 source 中不能跨 train/validation；
  `metadata_manifest` 不允许覆盖 split。

`split_manifest` 是每行一个 domain assignment 的 JSONL：

```json
{"domain":"lab_a","split":"train"}
{"domain":"lab_b","split":"validation"}
```

源数据内容或任务语义发生正式变更时，应分配新的 `source_id` 并重新生成缓存，不能让
同一身份静默代表两版数据。

## 在小测试集上运行正式配置

当前预实验集位于
`/media/datasets/EWM_SIM_REAL_PAIRS/model_train/test`，包含 12 个 source、48 个
episode。先确认四个模型资产、runtime、GPU 和 API 凭据都已经就绪，再执行：

```bash
cd /media/vlm/vlm-model/asset_llm_ckpt/vla-representation/project/yifan/Transfer
TEST_DATASET=/media/datasets/EWM_SIM_REAL_PAIRS/model_train/test

# 只读元数据预检：不解码视频、不调用 API、不加载模型。
PYTHONPATH=sim2real-prompt/src .venv/bin/python -m sim2real_prompt_annotation inspect \
  --config sim2real-prompt/config.test.yaml \
  --dataset "${TEST_DATASET}" \
  --show 5

# 先跑一条完整正式链路并审计。
bash scripts/process.sh \
  --config sim2real-prompt/config.test.yaml \
  --dataset "${TEST_DATASET}/train_00_hang_scissors" \
  --episodes 0

# 人工检查 frame0、whole-robot mask 和 clean scene 后再跑 48 条。
bash scripts/process.sh \
  --config sim2real-prompt/config.test.yaml \
  --dataset "${TEST_DATASET}" \
  --gpus 0,1,2,3,4,5,6,7 \
  --api-concurrency 16 \
  --decode-workers 16 \
  --micro-batch-size 2
```

`run` 会在源数据集内发布 `Reference/` 和 `meta/`，运行账号需要相应写权限。
`--force` 会忽略两个分支的有效缓存并重新调用 API/模型，普通续跑不要加。父仓库并行
入口先做全局 metadata 预检，再按 source 稳定分片到常驻 GPU worker；总 API/decode
并发会在 worker 间分配，最终仍进行一次全数据集 audit。

## CLI

```bash
sim2real-prompt inspect --config config.yaml [--dataset PATH] [--episodes 0,2,5-9]
sim2real-prompt run     --config config.yaml [--dataset PATH] [--episodes 0,2,5-9]
sim2real-prompt audit   --config config.yaml [--dataset PATH] [--episodes 0,2,5-9]
```

- `inspect`：只读发现并预检身份、split、task 和 Real 视角。
- `run`：执行两个独立分支、保存 checkpoint、join 成功 episode，并默认审计。
- `audit`：无需 API key 或模型，校验 schema v3、exact-one 图片、路径、尺寸、哈希、
  frame 0、场景类型和 Prompt 关联。

根目录不是单个 LeRobot 数据集时，可在配置或命令行设置 `dataset_glob`。`--limit`
适合快速选择发现顺序中的前 N 条；`--episodes` 按 episode index 精确选择。

## 输出契约

Transfer 只消费以下三个产物：

```text
<dataset>/
  Reference/
    episode_000000/
      reference_00.jpg
  meta/
    episodes_prompt.jsonl
    reference_images.jsonl
```

`episodes_prompt.jsonl` 每个 episode 的 `reference_ids` 固定为单元素列表：

```json
{"episode_index":0,"prompt":"The Agilex robot hangs scissors on a rack in a well-lit workshop.","reference_ids":["sha256:..."]}
```

`reference_images.jsonl` 固定使用 schema v3，并包含恰好一个 full-frame 环境
Reference：

```json
{"schema_version":3,"episode_index":0,"references":[{"reference_id":"sha256:...","reference_path":"Reference/episode_000000/reference_00.jpg","source_view":"camera_head","source_frame_index":0,"scope":"environment","reference_kind":"robot_removed_scene","width":1280,"height":720,"source_frame_sha256":"...","mask_sha256":"...","mask_area_fraction":0.182,"sha256":"...","provenance":{"operation":"robot_removal_inpainting","segmenter":{"backend":"robotseg"},"final_mask":{"sha256":"...","area_fraction":0.182},"inpainter":{"backend":"big_lama"},"residual_qa":{"enabled":true,"detector":{"backend":"yoloe"},"queries":["robot","robot arm","robot gripper"],"area_fraction":0.0,"threshold":0.002,"pass":true},"quality_control":{"outside_mask_unchanged":true,"residual_check_enabled":true,"residual_mask_area_fraction":0.0,"max_residual_area_fraction":0.002}}}]}
```

`reference_id` 是最终 JPEG 字节 SHA-256 的 `sha256:` 形式，且必须和同一行的
`sha256`、Prompt row 的单元素 `reference_ids` 一致。`source_frame_index` 恒为 0；
图片宽高必须与源 frame 0 一致。`mask_sha256` 标识完整分辨率二值 removal-mask 的像素，
mask PNG 只保存在可恢复 checkpoint 中，不进入 Transfer 训练 manifest。

中间状态位于 `output.root`：

```text
outputs/
  prompt/<hash-prefix>/<sample-hash>.json
  reference/<hash-prefix>/<sample-hash>.json
  reference_failures/<hash-prefix>/<sample-hash>.json
  reference_images/<hash-prefix>/<sample-hash>/<cache-key>/
    reference_00.jpg
    removal_mask.png
  run_report.json
```

这些是 checkpoint/诊断产物，不是训练数据契约。

## 性能、失败与续跑

- Prompt miss 时每个视频一次定点解码 8 个均匀位置；该 bundle 已含 frame 0，可供
  Reference 复用。只有 Reference miss 时只解码 frame 0。不会扫描整段视频。
- 每个未缓存 episode 只有一次 VLM 逻辑生成任务，以及一次 RobotSeg、一次 Big-LaMa 和
  一次 YOLOE residual QA；API 瞬时错误或响应截断时，逻辑生成任务可能按配置重试。
  本地模型在 worker 生命周期内懒加载并复用。
- API 由 `runtime.api_concurrency` 控制。结构化响应截断时可逐次扩大 completion 预算，
  其他可重试错误按 runtime 配置指数退避。
- 两个分支的 cache key 各自只绑定相关输入。Prompt 模型或 system prompt 变化不会使
  Reference 失效；RobotSeg/Big-LaMa/YOLOE 权重或 mask 参数变化不会使 Prompt 失效。
- 确定性 Reference 失败写入 `reference_failures/`，相同 key 续跑不重复昂贵模型调用；
  runtime 异常不伪装为成功 Reference。
- `runtime.fail_fast: false` 时，单条失败写入 `run_report.json` 并继续其他 episode；命令
  最终返回非零且状态为 `partial`。失败条目不会发布 raw frame fallback。
- 发布采用原子写、dataset lock 和 transaction marker。子集运行 merge 既有 row；中断
  代次必须显式恢复。迁移到 schema v3 时，成功发布会清理该 episode 旧的额外
  `reference_01.jpg`、`reference_02.jpg` 等文件。

## Python API

高层接口只有 `Sim2RealPreprocessingPipeline`：

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

本仓库自身代码为 Apache-2.0。RobotSeg、LaMa、Ultralytics/YOLOE 的源码和模型权重
均由使用者独立取得；其中 Ultralytics 涉及 AGPL-3.0 或商业许可。详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
