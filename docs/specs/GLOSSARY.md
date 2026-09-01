# iLLM Learning Glossary

本表给第一次手搓推理引擎的学习者使用。它解释本项目中术语的**操作性含义**：你应能在代码、事件、指标或测试中指出它，而不是只会背定义。首次阅读某个 Version SPEC 时，先检查本表；具体版本契约仍以该 SPEC 为准。

## 1. 模型与生成

| 术语 | 本项目中的含义 | 应能观察/验证 |
|---|---|---|
| decoder-only Transformer | 用已有 token 自回归预测下一个 token 的模型族；V1 只支持一个精确定义的 dense architecture/revision | module tree、每层 shape、逐 op golden |
| tokenizer | 文本与 token ID 之间的版本化映射；revision、special token 与输入模板属于运行 identity | token IDs、revision/digest、空输入/BOS 行为 |
| logits | 模型对词表中每个下一个 token 给出的未归一化分数 | shape/dtype、有限性、与 oracle 的 tolerance |
| prefill | 一次处理 prompt token、建立其 KV 并产生末位置 next-token logits | prefill token 数、KV 写入、TTFT 组成 |
| decode | 在已有 KV 上每步处理新 token，并继续生成下一个 token | 每步 query/context length、KV append、ITL |
| KV cache | attention 为历史 token 保存的 key/value；避免每步重算全部历史 | valid token 数、bytes、page/slot、生命周期 |
| RMSNorm | 按均方根缩放 hidden state 的归一化算子 | epsilon、dtype、逐 op 输出 |
| RoPE | 把 position 注入 query/key 的旋转位置编码 | position、base/scaling、边界 golden |
| MHA/GQA | Multi-Head/Grouped-Query Attention；GQA 让多个 query heads 共享较少 KV heads | Q/KV head 数、head mapping、shape |
| MLP | Transformer block 的前馈子层 | activation、gate/up/down shape 与输出 |
| sampling | 按 `SamplingSpec` 从 logits 选择 token；它是模型语义，不是随意 backend 细节 | processor 顺序、FP32 中间值、RNG test vector |
| greedy | 选择最大 logit，tie 选最小 token ID；不消耗 RNG | token 完全一致、tie fixture |
| structured output | 用确定语言状态约束允许 token，使最终输出属于声明的 regex/JSON 子集 | allowed-token set、独立 validator、dead-end/EOS |
| speculative decoding | draft 先提议 token，target 验证后只提交分布等价的结果 | proposal/verify/accept/reject、事务 rollback |

## 2. 引擎、调度与内存

| 术语 | 本项目中的含义 | 应能观察/验证 |
|---|---|---|
| continuous batching | 每个 iteration 重新组合仍可执行的请求，而不是等待固定 batch 全部结束 | request×step 矩阵、batch membership、reason |
| admission | 在 accepted 前依据队列、连接、token 与内存硬预算决定接收或拒绝 | 429/503、watermark、`MemoryPlan` |
| backpressure | 下游容量不足时有界地减速/拒绝上游，禁止无限排队 | queue/credit 上限、恢复水位、reason code |
| `SchedulePlan` | pure scheduler 根据不可变 snapshot 产生的执行 intent；executor 不自行改策略 | input digest、selected/deferred、plan validation |
| `MemoryPlan` | engine ready 前冻结的权重、KV、workspace、观测与 headroom 容量事实源 | manifest digest、reserve、峰值、越界失败 |
| contiguous KV | 每个请求使用连续 KV 存储的早期 reference layout | capacity、append、cache-off differential |
| paged KV | 逻辑 token 通过 page table 映射到固定物理页 | page/slot、fragmentation、allocator invariant |
| chunked prefill | 把长 prompt 拆成有界 chunk，与 decode 共享 step 预算 | chunk progress、decode reservation、TTFT/ITL |
| prefix cache | 跨请求复用完全相同、identity 匹配的已验证 prompt KV | matched/reused/forced tokens、lease/refcount |
| preemption/recompute | 内存压力时释放受害请求 KV，之后用已提交 token 历史重算恢复 | victim reason、released/recomputed token、liveness |
| goodput | 同时满足正确性与预注册延迟 SLO 的有效吞吐；不是原始 tokens/s | SLO 通过请求、offered load、失败率 |
| open-loop load | 到达速率独立于服务完成速度的负载模型，可显露排队和过载 | offered vs achieved load、queue growth |

## 3. 所有权、并发与失败

| 术语 | 本项目中的含义 | 应能观察/验证 |
|---|---|---|
| owner | 唯一有权改变某对象生命周期/状态的组件 | owner ID、合法 transition、重复命令结果 |
| lease | 在 lease 释放前资源不可被驱逐或复用的显式引用 | acquire/release、refcount、leak/double-free |
| fence | 异步设备工作完成的证明；逻辑取消不等于 fence 已完成 | launch/resolve、pending-free、地址复用时刻 |
| ticket | 把一次异步执行或传输的资源、状态、fence 和 terminal 绑在一起 | 状态机、commit/abort、orphan cleanup |
| linearization point | 并发操作对外“唯一生效”的原子时刻，例如 terminal CAS 或 drain 关闭 admission | race test 中唯一 winner、前后不变量 |
| terminal outcome | accepted request 最终唯一的 `FINISHED/CANCELLED/FAILED/DEADLINE_EXCEEDED` | terminal accounting、无 lost/duplicate |
| fail closed | 无法证明输入、identity、capability 或 artifact 兼容时拒绝，而不是猜测继续 | 结构化错误、无 silent fallback |
| idempotency | 重复相同 operation ID 不产生第二份副作用 | retry/fault test、返回既有状态 |
| generation fencing | 重启/替换后使用新 generation，拒绝旧进程迟到的 ack、ticket、output | stale-message counter、无重新加入 |
| fail-stop | 可信性未知时把整个故障域摘除，不让部分成功被提交 | group unhealthy、统一失败、无半写 |

## 4. 可观测、Capture 与 Replay

| 术语 | 本项目中的含义 | 应能观察/验证 |
|---|---|---|
| canonical event | engine 在决策/状态变化当时产生的版本化事实；UI 不重新推测原因 | event ID/schema、producer sequence、reason |
| capture | 一次运行的 manifest、events、checkpoint、指标与完整性声明 | checksum、drop/gap、privacy capability |
| checkpoint | replay reducer 在安全边界的版本化完整控制状态快照 | state digest、schema、后续 delta |
| state digest | 对 canonical control state 的确定编码摘要，用于定位首次 divergence | online/replay 每 step equality |
| R0 Event Playback | 原样播放已记录事件和时间关系，不重建完整状态 | event order/link、缺口 warning |
| R1 Control-plane Replay | 由 checkpoint + canonical delta 重建 scheduler/KV/control state，不运行模型数值 | 每 step digest、首次 diff |
| R2 Numerical Re-execution | 在严格 identity/capability 下重执行数值或采样；默认 capture 通常不具备 | model/token/RNG identity、tolerance |
| R3 What-if | 用替代策略模拟调度/容量；时间预测必须标 estimated 并在线验证 | policy input、estimated badge、real-run check |
| measured | 工具在该 run 直接测得 | source clock/counter 与误差范围 |
| derived | 从 recorded facts 确定计算，例如 duration 或聚合 | 输入事件与公式可追溯 |
| estimated | 模型/假设预测，不能冒充实测 | 假设、校准范围、验证计划 |
| unknown | capture 不足以回答；正确 UI 应明确不知道 | missing capability/reason |

## 5. 多 GPU 与分离式运行时

| 术语 | 本项目中的含义 | 应能观察/验证 |
|---|---|---|
| rank | 一个分布式 worker 的逻辑编号及其 generation/capability identity | `RankManifest`、GPU UUID、group membership |
| TP | Tensor Parallel；一个模型副本的 tensor/layer 运算跨 GPU 分片 | shard coverage、collective sequence、TP1 differential |
| DP | Data Parallel serving；多个独立模型副本由 router 分配不同请求 | replica ownership、routing、独立 KV namespace |
| collective | 多 rank 必须按相同顺序参与的通信操作，如 all-reduce/all-gather | `CollectivePlan`、participants、bytes、timeout |
| NCCL | NVIDIA GPU collective backend；错误/timeout 是 group failure boundary | version/capability、async error、rank trace |
| P/D disaggregation | Prefill 与 Decode 在不同 service/GPU 执行，通过受控 KV transfer 交接 | ticket、credit、source/destination ownership |
| topology | GPU、link、rank、TP group、DP replica 与服务角色的版本化映射 | `TopologyManifestV1`、digest、support matrix |

## 6. 验证、性能与发布

| 术语 | 本项目中的含义 | 应能观察/验证 |
|---|---|---|
| oracle/reference path | 最简单、可信、永久保留的正确性比较对象 | fixed fixture、可单独运行、差异定位 |
| golden test | 固定输入和预期输出的测试 | revision/digest、tolerance、首个 mismatch |
| differential test | 同一输入比较 reference 与 candidate | equal contract、全量 raw diff |
| property test | 生成多条状态/操作序列验证不变量 | seed、shrunk counterexample、invariant |
| gate | 进入下一切片/版本前必须机械通过的证据门 | command、threshold、artifact |
| DoD | Definition of Done；版本可发布所需的 Must/Conditional 清单 | 每项能映射到 evidence |
| ADR | Architecture Decision Record；记录一个重要选择、替代方案与后果 | decision、status、supersedes link |
| evidence bundle | clean reproduction、测试、capture、benchmark、截图、限制和 release notes 的集合 | manifest/checksum、命令、raw data |
| SLO | Service Level Objective；实验前冻结的延迟/失败目标 | TTFT/ITL/E2E threshold、达标 goodput |
| TTFT | Time To First Token；accepted/request start 到首个 committed token 可见 | start/end 事件与分位数 |
| ITL | Inter-Token Latency；相邻 committed output token 的间隔 | 每请求 token 间隔；说明首 token 是否排除 |
| TPOT | Time Per Output Token；稳定定义下的输出阶段平均时间，不等同所有 ITL 分位数 | 明确公式与 token weighting |
| E2E | 端到端请求延迟，从规范化起点到 terminal | queue/prefill/decode/stream 分解 |
| held-out suite | 调优完成、配置冻结后才打开的验证负载 | immutable digest、无事后改阈值 |
| Pareto improvement | 在一个指标改善时，其他 guardrail 不恶化的可复现优势 | baseline/candidate 曲线与 CI |
| SupportMatrix | 明确哪些模型、硬件、软件、拓扑和协议组合经过 qualification | exact versions、capabilities、unsupported behavior |

## 7. 发布状态、资格与责任帽子

发布状态不是一个布尔值，也不能由“命令跑完了”推断。本项目把文档、artifact、资格、性能和长期维护分开记录：

| 术语 | 本项目中的含义 | 应能观察/验证 |
|---|---|---|
| `spec_status` | Version SPEC 的文档生命周期：draft、accepted、implemented、released、superseded | review/acceptance 记录、amendment、successor 链接 |
| `release_stage` | artifact 阶段：dev snapshot、release candidate、released、withdrawn | immutable artifact digest、RC/tag、撤回原因 |
| RC / release candidate | 命令和候选 evidence 已冻结、等待剩余 qualification/promotion 的 artifact；不是稳定发布的同义词 | `release_stage=release_candidate`、未完成 profile 清单 |
| `qualification_tier` | evidence 已证明的最高层：unqualified、learning complete、functionally qualified、stable released | required profile 结果、support matrix、rollback 与独立复核 |
| `learning_complete` | 预热、toy/simulator、Learning Demo、测试与 Teach-back 完成 | CPU/simulator transcript；不能外推真实 GPU/拓扑 |
| `functionally_qualified` | 声明硬件上的 correctness、failure 和 compatibility required profiles 全部通过 | 固定环境 manifest、逐 profile pass 与原始证据 |
| `stable_released` | clean-room、全部 required profiles、support matrix、rollback、tag/evidence 和独立审计完整 | release verifier 重算结果；矩阵外组合仍 unsupported |
| qualification profile | 绑定 fixture、硬件/拓扑、软件、命令和门槛的一组资格测试 | stable profile ID、environment/evidence digest、result/reason |
| `not_run_resource_unavailable` | 需要的固定资源不存在，因此该 profile 没运行；它不是 pass 或 fail | 资源能力探测、缺失原因、可恢复条件 |
| `performance_outcome` | 与资格正交的性能季结论：pending、NoCandidate、Adopt、No-go 或不适用 | frozen ExperimentCard、raw runs、机械判定 |
| NoCandidate | Profile Gate 没有任何候选满足进入条件，因此不实现 candidate path | `selected_branch=null`、candidate code/dispatch 为 0、revisit trigger |
| Adopt | 唯一候选满足预注册 success 与全部 guardrail，可在 capability 内 guarded enable | held-out raw data、CI、fallback/rollback 门 |
| No-go | 候选已评估但未满足 success 或触发 guardrail；可信负结果仍可发布 | 默认 reference、完整归因、预算消耗与 revisit trigger |
| `lts_state` | 长期维护状态：not applicable、candidate、promoted、EOL | promotion attestation、SupportPolicy、起止/EOL/backport 规则 |
| LTS candidate | 普通 release 已形成，但 72h 等额外长期资格或独立复核尚未完成 | `lts_state=candidate`；不得使用 LTS promoted 徽标 |
| LTS promotion | 不改写原 release manifest，通过新签名 attestation/patch tag 把候选提升为 LTS | candidate digest、72h evidence、support/rollback audit、promotion digest |
| EOL | LTS 维护窗口结束，不再承诺常规 backport | 提前公告、日期、迁移目标和最后支持版本 |

角色采用“帽子”而非人数模型，owner ID 一律使用 kebab-case：

| 帽子 | 核心责任 | 常见领域别名 |
|---|---|---|
| `project-owner` / `spec-owner` | Scope、契约、DoD、跨域取舍和 accepted review | `architecture-owner` |
| `implementer` | 实现、测试、迁移和修复 | `model-owner`、`grammar-owner`、`sampling-owner`、`memory-owner`、`executor-owner` |
| `experiment-owner` | 预注册 workload/baseline/阈值并保存所有 run | `performance-owner` |
| `release-owner` | clean-room、qualification、rollback、tag、promotion artifact | `reliability-owner`、`security-owner` 可共同签署领域门 |
| `independent-verifier` | 只凭 finalized artifact/runbook 复核，不复用作者临时状态 | 真人、干净 CI 或无作者上下文的新会话 |

同一个人可以依次承担多个帽子，但不能把实现时的隐式环境或未保存判断带进独立复核。领域 owner 最终仍映射到上表责任边界，不能因“单人项目”而省略 owner、resolution artifact 或 verifier transcript。

## 8. 如何使用本表

每进入一个版本，至少完成三步：

1. 从该 SPEC 的“必须掌握”圈出陌生词，并在本表找到可观察证据；
2. 运行“进入门槛”预热实验，能用自己的话解释输入、状态变化和失败边界；
3. 在 teach-back 中引用真实 capture/test，而不是只复述本表。

若术语在本表和 accepted Version SPEC 中含义冲突，以 Version SPEC 为准，并通过 amendment 同步本表。
