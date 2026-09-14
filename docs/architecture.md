# 架构与文件职责

## 1. 固定边界

该 pipeline 把已具备稳定身份和 domain split 的 paired LeRobot episode 转换为
Transfer 训练所需的自然语言 Prompt 与一张干净环境 Reference。两条分支都只消费配置的
Real 主视角，默认 `camera_head`：

```text
                         ┌─ uniform 8 frames + task metadata
Real main-view video ────┤        └─ API VLM ──> prompt-only
                         │
                         └─ full-resolution frame 0
                                  └─ RobotSeg whole-robot mask
                                       └─ morphology + Big-LaMa
                                            └─ YOLOE residual QA
                                                 └─ one full scene Reference
```

Prompt 用一句自然英文描述机器人本体、任务、场景和可见光照。Reference 保留 frame 0
的完整环境、任务物体和背景，只移除机器人。Reference 不读取 Prompt、VLM query 或本地任务
分词，不做物体 crop，也不选择其他帧。

## 2. 每个 episode 的执行顺序

1. `dataset.py` 只读发现并验证 `source_id`、domain/split、任务和唯一 Real 视频路径。
2. `pipeline.py` 分别计算 Prompt 与 Reference cache key；两者互不包含对方的结果。
3. Prompt miss 时，`video.py` 一次打开视频并 seek 到 8 个均匀位置。第一个位置固定为
   frame 0，原始 BGR frame 0 可直接供 Reference 分支复用。
4. 只有 Reference miss 而 Prompt hit 时，`video.py` 只解码原分辨率 frame 0。
5. `prompt_branch.py` 把任务/机器人补充元数据和 8 张压缩 JPEG 发给 `qwen.py`，每条
   episode 只请求一个 prompt JSON 字段。
6. `reference_branch.py` 合并 RobotSeg 返回的 whole-robot mask，执行 closing、dilation 和
   面积门禁，再让 `inpainting.py` 填补 mask 区域。
7. 可选但生产默认开启的 YOLOE residual QA 只检查 inpaint 结果是否还含机器人；它不是
   RobotSeg 的 mask fallback。
8. `validation.py` 分别校验两个分支。`export.py` 保存独立 checkpoint；只有两个成功结果
   都存在时才构造单元素关联并原子发布。
9. `audit.py` 不加载模型或源视频，重新验证最终 manifest、Reference JPEG 与跨表关联。

当两个分支同时 miss 时，它们可在独立 executor 中交错执行；API 等待不会串行阻塞 GPU
Reference 路径。外层依然按 `reference.batch_size` 分块，以限制内存中的原分辨率 frame 0
数量。

## 3. Prompt 分支

输入固定为：

- `meta/episodes.jsonl:tasks[0]`；
- `meta/info.json:robot_type`、可选 `labels/labels.json:subtasks` 和可选
  `dataset.metadata_manifest` 补充字段；
- Real 主视角中包含首尾的 8 个唯一、递增、均匀帧。

图片只缩小到 `prompt.resize_long_edge`，不放大。OpenAI-compatible endpoint 返回精确
`PromptPayload`：

```json
{
  "prompt": "The Agilex robot hangs scissors on a rack in a well-lit workshop."
}
```

输出必须是一句自然英文视频描述，最多 56 词、560 字符，包含任务主体、动作和可见环境
信息，不暴露抽帧、标注或 Sim/Real 数据机制。VLM 不再生成任何 Reference 查询；旧 Prompt
checkpoint 中遗留的多余字段只在核心输入 fingerprint 兼容时被丢弃迁移，不会流入正式产物。

Prompt cache identity 覆盖 sample ID、任务和补充元数据、episode 长度/fps、Real 视频
路径/size/mtime、8 帧采样/编码设置、system prompt 字节、API 模型与 endpoint、生成参数和
分支 schema。命中时不解码 8 帧且不调用 API。

## 4. Reference 分支

### 4.1 Whole-robot mask

`RobotSegSegmenter` 懒加载官方 `showlab/RobotSeg` predictor，并对 frame 0 使用自动
whole-robot 类别；不要求人工 point/box。返回 mask 必须映射到原图尺寸。多个有效预测先做
union，然后经过：

1. 二值化；
2. 奇数核 closing，闭合机械臂/夹爪内部小裂缝；
3. `mask_dilation_pixels` 扩张，去除机器人轮廓边缘；
4. `min_mask_area_fraction <= area <= max_mask_area_fraction` 门禁。

最终 mask 的 SHA-256 按 full-resolution `uint8` 二值像素字节计算；同一个 mask 还以 PNG
保存在 Reference checkpoint，供人工检查和续跑完整性验证。

### 4.2 Big-LaMa inpainting

`BigLamaInpainter` 接收原始 BGR frame 0 与 final mask，内部转成 RGB `[0,1]` tensor，并按
`inpainting_modulo` 只向右/下 pad。生产支持：

- 官方 Big-LaMa 目录：`config.yaml` 与 `models/best.ckpt`；
- 已由部署方验证、接口兼容的 TorchScript 文件。

输出裁回原始宽高，再编码为一张 full-frame JPEG。mask 内发生变化的像素比例必须达到
`min_inpaint_change_fraction`；否则视为没有真实执行去除并失败。

### 4.3 Residual-robot QA

生产默认 `residual_check: true`。YOLOE 用固定小词表 `robot/robot arm/robot gripper`
检查 inpaint 后整图，合并所有残留 mask 后计算面积比例。超过
`max_residual_area_fraction` 则拒绝发布。YOLOE 不影响 removal mask，也不能在 RobotSeg
漏检时替代分割。

Reference cache identity 覆盖 sample ID、frame-0 Real 视频路径/size/mtime/ctime/device/inode、
声明的图像尺寸、视角、RobotSeg
runtime/config/checkpoint identity、morphology 和质量阈值、Big-LaMa runtime/checkpoint、
YOLOE runtime/checkpoint/固定词表以及 JPEG 设置。Prompt 模型、任务描述和 Prompt cache 均不
参与。命中成功 checkpoint 时不解码 frame 0、不调用三个模型；命中确定性失败 checkpoint 时
同样不重复昂贵推理。

## 5. Fail-closed 规则

以下情况都使当前 Reference 分支失败：

- RobotSeg runtime/checkpoint 不可用或没有非空 whole-robot mask；
- mask 几何、面积或哈希不满足配置；
- Big-LaMa runtime/checkpoint 不可用、输出 shape/数值错误或 mask 内变化不足；
- residual QA runtime 不可用，或残留机器人面积超过门限；
- JPEG、mask PNG、checkpoint 或发布文件无法完成一致性验证。

失败时不允许发布源 frame 0、裁图、纯色/OpenCV 填充、YOLOE mask 或任何其他备用产物。
确定性失败写入 `reference_failures/` 并绑定相同 cache key；模型初始化、CUDA/OOM 等运行时
错误仍明确进入 `run_report.json`，不会伪装为确定性视觉结论。

`runtime.fail_fast=false` 时其他 episode 继续；命令最终以 `partial` 和非零退出码报告失败。

## 6. Checkpoint、并发与发布

两套独立 checkpoint 位于：

```text
output.root/
  prompt/<prefix>/<sha256(sample_id)>.json
  reference/<prefix>/<sha256(sample_id)>.json
  reference_failures/<prefix>/<sha256(sample_id)>.json
  reference_images/<prefix>/<sha256(sample_id)>/<reference-cache-key>/
    reference_00.jpg
    removal_mask.png
```

`runtime.resume=true` 且没有 `--force` 时，只运行 miss 的分支。`--force` 同时忽略两套
checkpoint。cache JSON 损坏、schema/key 不匹配、staging JPEG 或 mask PNG 哈希不一致均按
miss/failure 处理，不盲信文件存在。

父仓并行入口先全局 discovery，再按 source 稳定分片到固定 GPU worker。每个 worker 生命周期
内复用 RobotSeg、Big-LaMa 和 YOLOE；`runtime.decode_workers` 与
`runtime.api_concurrency` 分别约束视频解码和 API 并发。全局身份/domain/split 不变量在分片
前验证，per-worker report 最后聚合。

最终发布使用临时文件替换、dataset-level `flock` 与 transaction marker。子集运行只 merge
本次成功 episode，保留其他行。一次 episode 的 Prompt 和 Reference 必须同时有效才更新两个
manifest。迁移到 exact-one 时发布器固定写 `reference_00.jpg`，并清理同 episode 目录中过时的
其他 ordinal JPEG，避免旧数据被误消费。

## 7. 最终 schema v3

数据集内只有三个训练产物：

- `Reference/episode_XXXXXX/reference_00.jpg`；
- `meta/episodes_prompt.jsonl`；
- `meta/reference_images.jsonl`。

Prompt row 是 `{episode_index, prompt, reference_ids}`，其中 `reference_ids` 恰有一个
SHA-prefixed ID。Reference row 是 `{schema_version: 3, episode_index, references}`，其中
`references` 恰有一项，字段包括：

```text
reference_id             = "sha256:" + final JPEG SHA-256
reference_path           = Reference/episode_XXXXXX/reference_00.jpg
source_view              = configured Real main view
source_frame_index       = 0
scope                    = environment
reference_kind           = robot_removed_scene
width, height            = source frame-0 dimensions
source_frame_sha256      = decoded source frame identity
mask_sha256              = final full-resolution binary mask pixel identity
mask_area_fraction       = final removal mask fraction
sha256                   = final JPEG SHA-256
provenance               = segmenter/inpainter/residual-QA identities and gates
```

两个 row 按 episode join，Prompt 的单元素 `reference_ids` 必须精确等于 Reference ID。
旧版 schema、零项/多项 Reference、非 frame-0、非 environment、非 robot-removed scene、尺寸或
内容哈希不一致都会被拒绝。

`audit` 只读取发现阶段 metadata 与最终产物，不加载 API client、RobotSeg、Big-LaMa、YOLOE、
源视频或 Parquet。全量 audit 还要求 manifest episode 集合与源 episode 集合精确相等；局部
selection audit 只验证所选行，`annotations_ready` 保持未定。视频、Parquet、关节映射和 Transfer
训练配置由父仓 validator 检查。

## 8. 身份与 split

`source_id` 是版本化源身份，不是目录显示名。canonical sample ID 使用长度前缀：

```text
<len(source_id)>:<source_id>:<episode_index>
```

checkpoint 文件名对完整 ID 求 SHA-256，避免字符替换碰撞和路径问题。正式修改任务、视频或
其他决定样本语义的源内容后，必须发布新的 `source_id`。

`domain` 是 split 隔离单位。一个 domain 不能在当前 source 或同次发现的多个 source 间跨越
train/validation。episode row、`info.splits` 和外部 assignment 冲突时立即失败，不做随机重划。

## 9. 文件职责

| 文件 | 唯一职责 |
| --- | --- |
| `README.md` | 安装、模型准备、数据要求、正式命令、CLI 与输出入口。 |
| `config.example.yaml` | 严格配置的完整示例；相对路径以 YAML 所在目录为基准。 |
| `pyproject.toml` | 包元数据、基础/Reference/dev 依赖、CLI、pytest 与 Ruff 设置。 |
| `THIRD_PARTY_NOTICES.md` | RobotSeg、LaMa、Ultralytics/YOLOE 的许可边界。 |
| `examples/python_api.py` | 高层 Python facade 的最小示例。 |
| `docs/architecture.md` | 本文：数据流、缓存、失败、schema 与逐文件边界。 |
| `src/sim2real_prompt_annotation/__init__.py` | 导出高层 facade 与配置接口。 |
| `src/sim2real_prompt_annotation/__main__.py` | `python -m` 到 CLI 的入口。 |
| `src/sim2real_prompt_annotation/api.py` | `inspect/run/audit` facade、配置 override 与 selector。 |
| `src/sim2real_prompt_annotation/cli.py` | 三个子命令、JSON 输出和退出码。 |
| `src/sim2real_prompt_annotation/config.py` | Pydantic 配置、交叉约束与路径解析。 |
| `src/sim2real_prompt_annotation/models.py` | Prompt、mask、scene artifact、checkpoint 与 manifest DTO。 |
| `src/sim2real_prompt_annotation/dataset.py` | 只读 discovery、canonical ID、Real view 和 split 校验。 |
| `src/sim2real_prompt_annotation/video.py` | 均匀 8 帧和独立 frame-0 定点解码、Prompt JPEG。 |
| `src/sim2real_prompt_annotation/qwen.py` | OpenAI-compatible transport 与 prompt-only JSON 解析。 |
| `src/sim2real_prompt_annotation/prompt_branch.py` | VLM 请求组装、Prompt 校验与分支 fingerprint。 |
| `src/sim2real_prompt_annotation/robot_mask.py` | Segmenter protocol、BGR/mask geometry 公共工具。 |
| `src/sim2real_prompt_annotation/robotseg.py` | 官方 RobotSeg 的懒加载、checkpoint 校验和 frame-0 adapter。 |
| `src/sim2real_prompt_annotation/inpainting.py` | Big-LaMa 懒加载、tensor/padding 适配和 batch inpainting。 |
| `src/sim2real_prompt_annotation/yoloe.py` | YOLOE 懒加载与固定词表 residual-mask QA。 |
| `src/sim2real_prompt_annotation/reference_branch.py` | mask union/morphology、inpaint 门禁、residual QA 和 JPEG 构造。 |
| `src/sim2real_prompt_annotation/validation.py` | 无 I/O 的 Prompt/scene Reference 不变量验证。 |
| `src/sim2real_prompt_annotation/io_utils.py` | containment、哈希、JSON(L) 和原子写基础函数。 |
| `src/sim2real_prompt_annotation/export.py` | 独立 checkpoint、staging mask/JPEG、manifest merge 与事务发布。 |
| `src/sim2real_prompt_annotation/audit.py` | 无模型的 schema v3/JPEG/关联审计。 |
| `src/sim2real_prompt_annotation/pipeline.py` | 分块、并发、retry/cache、success join、发布和报告。 |
| `src/sim2real_prompt_annotation/prompts/prompt_system.txt` | 一句视频 caption 的 VLM 规则。 |
| `tests/unit/` | DTO、配置、解码、VLM、RobotSeg/Big-LaMa/YOLOE 和分支单元测试。 |
| `tests/test_pipeline_integration.py` | CPU fake 下的双分支、schema v3、失败和续跑集成测试。 |

CPU 单测通过 dependency injection，不代表真实权重质量已经验收。生产放行必须在目标 GPU 上
检查源 frame 0、最终 removal mask、inpaint 后场景、YOLOE 残留率、schema v3 audit 和第二次
运行的双分支 cache hit。
