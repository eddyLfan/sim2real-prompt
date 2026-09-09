# 架构与文件职责

## 设计边界

这条 pipeline 的目标是把一个已具备稳定身份和 split 的 paired LeRobot episode
转换为 Transfer 训练所需的自然语言 prompt 与 1～3 张 Multi-Reference 图片。运行时
只读取配置的 Real 主视角，默认 `camera_head`。任务语义来自数据集元数据，视频帧只
补充可见执行方式、场景和光照。

每个 episode 的主路径固定为：

1. `dataset.py` 只读发现并验证身份、split、任务和 Real 视频路径，不扫描 Parquet。
2. `pipeline.py` 分别计算 Prompt 和 Reference cache key。
3. Prompt 未缓存时，`video.py` 只打开一次 Real 视频，直接 seek 并取得恰好 8 个均匀
   帧，同时把原始 BGR 首帧供 Reference 共用。
4. `prompt_branch.py` 把原始任务、机器人/子任务元数据和 8 帧交给 `qwen.py`。每个
   episode 只发一次结构化 VLM 请求，响应同时包含一句 prompt 和 1～8 个
   `reference_queries`。
5. Reference 未缓存时，`yoloe.py` 只在 Real 首帧执行 YOLOE-11s-seg；相同 query
   signature 的 episode 合并成 GPU batch，并复用文本 embedding。
6. `reference_branch.py` 依据 segmentation mask 做面积/置信度/重叠质检，生成带上下文
   padding 的候选 crop，再用 `selection_seed + sample_id` 选择 1～3 张。
7. `validation.py` 检查 prompt、required query 覆盖、primary Reference 与数量；
   `export.py` 保存独立 checkpoint，并原子发布图片和两个 manifest。
8. `audit.py` 重新读取最终训练产物，验证关联、顺序、路径和图片哈希。

不包含备用定位链路。YOLOE 没有返回可用 mask，required 实体不在候选池，或最终选择
中没有 primary 任务物体时，当前 episode 失败并进入报告。

## Prompt 分支契约

输入固定为：

- `meta/episodes.jsonl:tasks[0]`；
- `meta/info.json:robot_type`、可选 `labels/labels.json:subtasks` 与可选
  `dataset.metadata_manifest` 补充字段；
- 配置的 Real 视角中包含首帧和尾帧的 8 个均匀帧。

8 帧会缩小到 `prompt.resize_long_edge`，不会放大。Qwen OpenAI-compatible endpoint
返回符合 `PromptPayload` 的单个 JSON object，例如：

```json
{
  "prompt": "The Agilex robot aligns the handles of a preassembled banana bunch in the same direction on a worktable under even indoor lighting.",
  "reference_queries": [
    {"query": "banana", "role": "primary", "required": true},
    {"query": "Agilex robot", "role": "robot", "required": false},
    {"query": "worktable", "role": "workspace", "required": false}
  ]
}
```

角色集合是
`primary | destination | secondary | robot | workspace | environment | background`。
每个 primary 必须 `required=true`；依赖目标实体才能完成任务时，destination 也应
required。任务实体必须来自权威任务描述，但 query 应去除装配状态、关系和动作修饰，
保留 YOLOE 容易匹配的最短可见名词（例如把 `preassembled banana bunch` 化为
`banana`）。可选的机器人、工作区、环境和背景 query 必须在第 1 张输入图（Real frame 0）中清晰、可分割，不能把后续帧才出现的偶然物体
提升为任务实体。光照只写入 prompt，不作为检测 query。

最终 prompt 必须是一句自然英文视频描述，最多 56 词、560 字符，并且不能暴露数据
预处理机制。

## Reference 分支契约

YOLOE-11s-seg 输出的 mask 会映射回首帧原始尺寸。mask 的紧致边界是
`bbox_xyxy`，按 `reference.crop_padding` 扩展并裁剪到图片边界后得到 `crop_xyxy`。
候选依次经过：

- 配置置信度阈值；
- mask 必须存在且非空；
- 默认分割面积占比 `[0.0005, 0.85]`；
- bbox IoU 去重；
- JPEG 字节去重；
- `candidate_pool_size` 上限。

候选池可以大于 3，但只存在 Reference checkpoint 中。最终目标数量在
`[min_images, min(max_images, candidate_count)]` 内 seeded 随机产生；primary 和 required
候选优先占用槽位，其余候选随机 dropout。相同 `selection_seed`、`sample_id` 和输入会
得到完全相同的结果，便于续跑与复现；修改 seed 会使 Reference cache 失效并重新选择。
这是预处理时的静态采样，不是训练过程中逐 epoch 变化的 dropout。

所有 selected crop 固定来自 Real frame 0，路径固定为：

```text
Reference/episode_<episode_index:06d>/reference_<ordinal:02d>.jpg
```

图片内容 SHA-256 同时构成 `reference_id`；最终顺序先按 primary/required/role 优先级
稳定排列。每个最终 episode 至少有一个 primary Reference。

## 批处理和缓存

外层按 `reference.batch_size` 分块，避免一次将整个数据集的视频帧留在内存中。
Real 视频解码使用 `runtime.decode_workers` 线程池，VLM 使用
`runtime.api_concurrency` 独立线程池。YOLOE 请求按完整 query signature 分组；每组只
激活一次词表并做一次 batch prediction。

`output.root/prompt/<prefix>/<sha256(sample_id)>.json` 和
`output.root/reference/<prefix>/<sha256(sample_id)>.json` 是两套分片 checkpoint；最终
crop 先按 sample/cache key 写入 `output.root/reference_images/` staging，再进入发布事务：

| 分支 | cache key 的核心输入 | 命中后的行为 |
| --- | --- | --- |
| Prompt | sample ID、任务/机器人/补充元数据、episode length/fps、Real 视频路径/size/mtime、8 帧配置、system prompt、模型与 endpoint、生成参数 | 不解码 8 帧、不调用 VLM |
| Reference | Prompt queries、Real 视频路径/size/mtime、YOLOE 参数、权重 SHA-256、Ultralytics 版本、选择 seed | 不解码首帧、不调用 YOLOE |

当 Prompt 命中而 Reference 未命中，pipeline 只打开并解码 Real 首帧。Prompt 未命中时，
同一次 8 帧定点解码已经保留首帧，不会为了 Reference 再开视频。cache JSON 损坏、key
不匹配或已发布图片哈希不匹配都按 miss 处理。

`runtime.resume=true` 且没有 `--force` 时启用上述行为。`--force` 同时忽略两分支已有
checkpoint。VLM 的可重试错误按 runtime 的次数和指数退避处理；YOLOE batch 只有发生
显存/内存分配错误时才递归二分降载，确定性配置或解析错误会让该组一次失败，避免指数级
重复推理；后处理错误按 episode 隔离。
`fail_fast=false` 会继续
其他 episode，并把 `sample_id`、阶段、异常类型和信息写进 `run_report.json`。

发布使用临时文件替换、dataset-level `flock` 和 transaction marker，避免并发进程写出
半行 manifest，并使中断发布能被 audit 和 Transfer reader 明确识别。中断后的下一次运行
必须覆盖 marker 中的全部 sample 才能恢复，不能用无关子集掩盖半完成代次。发布始终只更新
本次成功 episode、保留其他 row；一次 `run` 只把 Prompt 和 Reference 都通过校验的 episode
加入发布集合。

## 最终训练契约

数据集内的训练产物只有：

- `Reference/episode_XXXXXX/reference_YY.jpg`；
- `meta/episodes_prompt.jsonl`；
- `meta/reference_images.jsonl`。

Prompt row 的 schema 是精确的
`{episode_index, prompt, reference_ids}`。Reference row 是
`{schema_version: 2, episode_index, references}`；每个 reference 包含图片身份/路径、
首帧与视角、query/label/role/scope、YOLOE confidence、mask bbox、实际 crop 边界以及
图片 SHA-256。两个 row 按 episode join，Prompt 的 `reference_ids` 必须与 Reference
数组中的 ID 数量、内容和顺序完全相同。

`audit` 只使用发现阶段元数据和上述最终产物，不加载 API client、YOLOE、源视频内容或
Parquet。它验证：每条 episode 都有两个 row，manifest key 与源 episode 集合精确一致，
1～3 张图均存在且可解码，图片尺寸与 crop 边界一致，路径严格符合 ordinal，哈希与 ID
一致，视角/帧号正确，至少一个 primary，且跨表顺序一致。显式传入 episode/limit 时只保留
并审计选中 row，不让未选 episode 的损坏产物影响局部重跑，并将 `annotations_ready` 留空；
默认全量 audit 才能确认 annotation 产物完整。视频、Parquet、关节映射及训练配置仍由
Transfer 自身的 validator 检查。

## 身份与 split 不变量

`source_id` 是源版本身份，不是易变的显示名。canonical sample ID 使用长度前缀编码：

```text
<len(source_id)>:<source_id>:<episode_index>
```

这避免空格、下划线、路径分隔符等字符被替换后发生碰撞；checkpoint 文件名再对完整 ID
做 SHA-256，因此不会产生路径穿越或文件名过长。正式修改任务、视频或其他决定样本语义
的源内容后，应给数据分配新的 `source_id`。

`domain` 是 split 隔离单位。一个 domain 不允许在当前数据集内、或同次发现的多个数据集
间跨越 train/validation。episode row、`info.splits` 和外部 domain assignment 之间的任何
冲突都立即失败，不做 episode 级随机拆分。

## 文件职责

以下是重构后仍属于运行路径的文件；各模块只承担一层职责。

| 文件 | 唯一职责 |
| --- | --- |
| `README.md` | 安装、数据要求、测试数据正式命令、CLI 和输出契约入口。 |
| `config.example.yaml` | 严格配置 schema 的可运行示例；相对路径以 YAML 所在目录为基准。 |
| `.env.example` | API 环境变量名称示例，不含真实密钥。 |
| `pyproject.toml` | Python 包元数据、基础/YOLOE/dev 依赖、CLI entry point、测试与 lint 设置。 |
| `LICENSE` | 本仓库自有代码的 Apache-2.0 许可。 |
| `THIRD_PARTY_NOTICES.md` | Ultralytics/YOLOE 的 AGPL-3.0 与权重许可边界。 |
| `.gitignore` | 排除密钥、本地输出、模型权重和构建缓存。 |
| `.github/workflows/ci.yml` | 运行 pytest、Ruff、构建 wheel，并验证安装后的 CLI。 |
| `examples/python_api.py` | 唯一高层 Python facade 的最小示例。 |
| `docs/architecture.md` | 本文：分支、并发、缓存、失败和逐文件边界。 |
| `src/sim2real_prompt_annotation/__init__.py` | 导出高层 facade 与配置类型。 |
| `src/sim2real_prompt_annotation/__main__.py` | 支持 `python -m sim2real_prompt_annotation`，转交 CLI。 |
| `src/sim2real_prompt_annotation/api.py` | `inspect/run/audit` 的 Python facade、配置 override 与 episode selector 解析。 |
| `src/sim2real_prompt_annotation/cli.py` | 仅定义 `inspect`、`run`、`audit` 三个子命令及 JSON/退出码行为。 |
| `src/sim2real_prompt_annotation/config.py` | Pydantic 严格配置、默认值、交叉字段约束和相对路径解析。 |
| `src/sim2real_prompt_annotation/models.py` | 分支 DTO、VLM schema、检测/图片对象及最终 manifest row 类型。 |
| `src/sim2real_prompt_annotation/dataset.py` | 严格只读 metadata discovery、canonical ID、任务/视角路径和 domain split 校验。 |
| `src/sim2real_prompt_annotation/video.py` | 单次打开并定点读取 Real 8 帧、原始首帧共享和 Prompt JPEG 编码。 |
| `src/sim2real_prompt_annotation/qwen.py` | OpenAI-compatible Qwen transport、8 张图片打包、JSON schema 请求和响应解析。 |
| `src/sim2real_prompt_annotation/prompt_branch.py` | 组装一次 VLM 请求并产出 prompt、queries 与输入 fingerprint。 |
| `src/sim2real_prompt_annotation/yoloe.py` | YOLOE 懒加载、query normalization/embedding cache、signature 分组和 mask 几何解析。 |
| `src/sim2real_prompt_annotation/reference_branch.py` | 首帧检测后处理、候选池、seeded 1～3 选择和 JPEG crop 构造。 |
| `src/sim2real_prompt_annotation/validation.py` | 无 I/O 的 Prompt/Reference 产品不变量检查。 |
| `src/sim2real_prompt_annotation/io_utils.py` | 路径 containment、哈希、JSON(L) 与原子写基础函数。 |
| `src/sim2real_prompt_annotation/export.py` | 两分支 checkpoint、Reference 图片保存、manifest merge、锁和发布。 |
| `src/sim2real_prompt_annotation/audit.py` | 不调用模型的最终 manifest/图片结构与哈希审计。 |
| `src/sim2real_prompt_annotation/pipeline.py` | 分块、并发、retry/cache 调度、成功 join、发布和 run report。 |
| `src/sim2real_prompt_annotation/prompts/prompt_system.txt` | VLM 的单句 caption 与任务实体 query 规则。 |
| `src/sim2real_prompt_annotation/py.typed` | 声明该发行包提供类型信息。 |
| `tests/unit/test_config.py` | 配置常量、严格校验和路径解析测试。 |
| `tests/unit/test_dataset.py` | 身份、split、视角和只读 discovery 测试。 |
| `tests/unit/test_video.py` | 均匀索引、单次打开与定点解码测试。 |
| `tests/unit/test_prompt_branch.py` | 单请求、8 帧顺序和结构化 Prompt 输出测试。 |
| `tests/unit/test_yoloe.py` | YOLOE 懒加载、有界 embedding cache、batch 和原图 mask 几何测试。 |
| `tests/unit/test_reference_branch.py` | mask 过滤、去重、seeded 选择与无 fallback 测试。 |
| `tests/test_pipeline_integration.py` | CPU fake 下验证双分支、发布、审计、失败和零解码续跑。 |

测试通过依赖注入 fake VLM/YOLOE 保持 CPU-only；真实运行仍需要 endpoint、YOLOE 可选
依赖、分割权重和配置指定的设备。
