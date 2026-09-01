# iLLM Cross-Version Contracts

本文是 17 个 Version SPEC 之间的契约账本。它不重复某一版本的全部需求，而是记录核心对象何时引入、何时冻结、何时允许 breaking change，以及所有后续版本必须继承的 identity、ownership、terminal、failure 和 replay 语义。

## 1. 统一成熟度词汇

| 词汇 | 精确定义 | 对外含义 |
|---|---|---|
| `experimental` | 仍允许在后续 V0 release 改字段或语义；artifact 必须带 version/digest 并 fail closed | 不能声称兼容或稳定 |
| `format_frozen_for_v0` | 本 V0 列车内不再复用 enum/code、删除 required field 或改变已有字段含义 | 仍不是 public stable；V1.0 前可通过 migration 做最后一次收敛 |
| `candidate` | 语义、conformance corpus、migration 与 evidence 已齐，等待稳定发布门 | 只能称 release candidate |
| `stable` | 已通过声明 support matrix 的 clean-room、compatibility、failure 与 release gates | 同 major 内按兼容策略维护 |

“在 V0 中稳定”“继承稳定”等模糊说法一律替换为上表四值之一。V0.x 契约最早只能是 `candidate`；V1.0 才建立第一组 public `stable` 单卡契约。

## 2. Contract maturity ledger

| Contract | 首次引入 | V0/候选边界 | Public stable | Breaking change 规则 |
|---|---|---|---|---|
| `ToolchainManifest` | V0.0 experimental | V0.8 candidate | V1.0 | project-root/bootstrap/lock/interpreter/entrypoint/artifact digest 或环境逃逸校验语义变化需新 schema；qualification/rollback 必须绑定 exact manifest |
| `RunManifest` | V0.0 experimental | V0.8 candidate | V1.0 | V1.x 只加 optional 字段；语义变化需新 schema/migration |
| `ReleaseManifest` six-axis qualification schema | V0.0 experimental | V0.8 candidate | V1.0 stable | 六轴及逐 profile 结果正交保存；claim 只能由 evidence 推导，字段含义/层级变化需新 schema |
| `DomainEvent` envelope | V0.0 experimental | V0.8 candidate | V1.0 | enum number 不复用；required/meaning 变化需 major |
| `ErrorRegistry` | V0.0 experimental | V0.3 `format_frozen_for_v0`，V0.8 candidate | V1.0 | code 不改义；新增 code 可 minor |
| `SupportedModelSpec` | V0.1 `format_frozen_for_v0` | V0.8 candidate | V1.0 | spec identity 改变使 cache/graph/replay artifact 失效 |
| `WeightLoadManifest` | V0.1 `format_frozen_for_v0` | V0.8 candidate | V1.0 | mapping、alias 或 digest 语义改变需新 version |
| `SamplingSpec v1` / RNG | V0.1 `format_frozen_for_v0` | V0.8 candidate | V1.0 | processor 顺序、Philox 映射、uniform/CDF 规则变化需新 contract version |
| `Engine.generate` | V0.1 experimental | V0.8 candidate | V1.0 | V1.x backward compatible |
| `Engine.stream` | V0.2 experimental | V0.8 candidate | V1.0 | terminal、delta 或 cancel 语义不可 minor breaking |
| Request state machine | V0.2 experimental | V0.8 candidate | V1.0 | 可加 non-terminal state；terminal 含义不可改变 |
| `MetricRegistryV1` | V0.2 experimental | V0.8 candidate | V1.0 stable | V1+ ExperimentCard 只能引用 fully-qualified、带版本 ID；含义/单位/边界/分母/weighting 改变必须新 ID |
| `MemoryPlan` | V0.2 experimental | V0.4 candidate | V1.0 | engine ready 前冻结；运行期不得隐式重算容量 |
| `SchedulePlan` | V0.3 experimental | V0.8 candidate | V1.0 | 可加 optional intent，不改已有 commit/rollback 含义 |
| `PageHandleV1` / `KVLayoutSpec` | V0.4 `format_frozen_for_v0` | aggregate V0.8 candidate | V1.0 | layout/instance/epoch 变化需 capability check 与 artifact invalidation |
| `PreemptionPolicyV1` / `PreemptionIntentV1` | V0.5 candidate / `format_frozen_for_v0` | aggregate V0.8 candidate | V1.0 | victim、progress、lease/fence 语义变化需新 contract |
| `ResumeCursorV1` | V0.5 `format_frozen_for_v0` | aggregate V0.8 candidate | V1.0 | logical history、KV valid-through、next output position、pending input、base-next-logits 或 request generation 的含义变化需新 cursor version |
| `AttentionPlanV1` / homogeneous paged ABI | V0.4 experimental | V0.8 candidate | V1.0 | query/position/block-table/write-slot 语义变化需 backend conformance version |
| `AttentionPlanV2` / chunk contract | V0.6 `format_frozen_for_v0` | aggregate V0.8 candidate | V1.0 | mixed segment/indptr/chunk 语义变化需新 plan version |
| `CachePoolIdentityV1` / `PrefixKeyV1` / `PrefixIndex` | V0.7 `format_frozen_for_v0` / candidate | aggregate V0.8 candidate | V1.0 | identity 字段减少或 isolation 放宽是 breaking change |
| Capture R0/R1 full-engine | V0.8 | V0.8 candidate | V1.0 | V1.x 至少读取前一个 minor，或返回结构化不兼容诊断 |
| `SelectionManifest` / `CandidateStudyManifestV1` / `ExternalBaselineManifestV1` / `ExternalGapStudyManifestV1` | V1.1 | V1.1 candidate | V1.1 stable release | branch 基数、冻结 baseline、预算/metric/suite identity 或 outcome 语义变化需新 schema/study；接受后不可换外部版本 |
| `GraphKey` | V1.1 M7b 条件式 | Adopt 时为 candidate；其他 outcome 不创建该公开能力 | Adopt 的 V1.1 stable release | compiler/runtime/shape identity 改变即失效 |
| `GrammarSpec` / DFA / `SamplingSpec v2` | V1.2 | V1.2 candidate | V1.2 stable release | byte alphabet、processor 顺序或 EOS/dead-end 语义变化需新 version |
| `DraftModelManifestV1` / `SpeculativeSpec` / `SpecTxnV1` | V1.3 | V1.3 candidate | V1.3 stable release | p/q、RNG domain、KV valid-through 或 commit 语义变化需新 version |
| `TopologyManifestV1` / `RankManifest` / `TensorDistributionManifest` / `CollectivePlan` | V2.0 | V2.0 candidate | V2.0 stable release | mixed rank 必须 handshake；所有 rank 校验同一 plan digest |
| `GroupReservationV1` / `SamplingCommitRecord` | V2.0 | V2.0 candidate | V2.0 stable release | local group accepted/all-rank commit 语义不可兼容性改变 |
| `RoutingDecisionRecord` / `AdmissionLease` / `LeaseOutcomeCertificate` | V2.1 | V2.1 candidate | V2.1 stable release | accepted 线性化、unknown outcome 与 retry fencing 不可兼容性改变 |
| `KVTransferManifest` / `DecodeCursorV1` / `DeadlineBudgetV1` | V3.0 | V3.0 candidate | V3.0 stable release | cursor/layout/deadline authority 不匹配 fail closed |
| `FutureDecodeCredit` / `AdmissionBundle` | V3.0 | V3.0 candidate | V3.0 stable release | 双资源接单必须原子 |
| `SupportMatrix` / `UpgradePlan` / `OperationJournal` | V3.1 candidate | V3.1 candidate | V3.1 stable release；LTS 是另一个资格轴 | support window 内不得移除组合；plan 固定 source/target/rollback identity；LTS promotion 只新增签名 attestation，不改原 manifest |
| `CoordinatorEpoch` / `MutationLease` | V3.1 candidate | V3.1 candidate | V3.1 stable release；LTS 是另一个资格轴 | stale epoch/lease 永远无权提交 mutation |
| `LtsPromotionAttestationV1` | V3.1 conditional candidate | V3.1 candidate | V3.1 stable release；实例仍只在 promotion gate 后签发 | detached、append-only；必须引用原 candidate manifest/evidence digest、签名与 support policy，字段含义变化需新 schema |

### 2.1 Release qualification axes

`ReleaseManifest` 必须分别保存 `spec_status`、`release_stage`、`qualification_tier`、`qualification_profiles[].result`、`performance_outcome` 与 `lts_state`；任何两个轴都不得互相推导或覆盖。ExperimentCard 的 `success/no_go` 在 manifest 中规范化为 `Adopt/No-go`，无候选才是 `NoCandidate`，运行前为 `pending`。`release_stage=released` 不证明 `qualification_tier=stable_released`，NoCandidate/No-go 不表示功能失败，Adopt 不覆盖 correctness gate，LTS candidate 也不等于 promoted。`not_run_resource_unavailable` 只能属于一个具体 profile，不能记为 pass。发布工具从 evidence 重算最高 qualification tier，claim 超前时 fail closed。

## 3. Identity chain

identity 是**按 artifact 类型定义的投影**，不是把所有字段无差别塞进一个 key。每种可复用 artifact 必须声明 required identity schema；缺字段、digest 变化或 pool/protocol identity 不同就失效或 fail closed，绝不凭模型名、进程地址或“看起来相同”猜测。

### 3.1 Execution identity 与 artifact projection

```text
ToolchainManifest digest + installed software artifact/build digest
+ SupportedModelSpec.spec_id + weights_digest + WeightLoadManifest.digest
+ tokenizer revision + chat/input template digest
+ SamplingSpec.version + RNG algorithm/mapping version
+ KVLayoutSpec.layout_id + dtype + backend/capability identity
```

这是一次完整 execution/replay 的 identity。具体 artifact 使用下列规范投影：

| Artifact | Required identity | 明确不属于该 key |
|---|---|---|
| Weight/loader result | build 的 model numerical contract、model spec、weight/load manifest | scheduler、SamplingSpec |
| `CachePoolIdentityV1` | software/model numerical contract、model spec、weights、runtime dtype、KV layout、RoPE/position、backend numerical contract、adapter | request sampling/RNG |
| `PrefixKeyV1` | CachePoolIdentity digest、tokenizer/input-template digest、tenant/cache namespace/salt、exact token IDs | SamplingSpec/RNG；相同 token 的 KV 与其采样来源无关 |
| Graph/compile artifact | build/compiler/runtime、model/weights、layout/backend、shape bucket、captured sampling mode | tenant/request ID |
| R2 numerical replay | 完整 execution identity + input tokens + RNG draw identity | wall-clock arrival order |

若 cache pool 在单个 process/run 内创建，build/model/backend 等常量仍必须进入 `CachePoolIdentityV1` 与 manifest；不能因为“当前恰好相同”从 schema 删除。Prefix lookup 可以只携带 pool digest 与 request-varying字段。

RNG draw identity 至少包含 `actual_seed` 与逻辑 tuple：

```text
(absolute_output_pos:uint64, draw_kind:uint32, draw_index:uint32)
```

batch index、scheduler step、rank arrival order 或 proposal chunking 不得进入随机流 identity。每个新 draw purpose 都分配稳定 `draw_kind`；已发布值不复用。

### 3.2 Optional generation identity

V1.2+ 约束解码再加入：

```text
GrammarSpec.version + grammar_bytes_digest + byte_alphabet_version
+ DFA compiler/conformance version + processor_order_version
```

V1.3+ speculative 路径再加入：

```text
target model/spec/weights identity
+ draft model/spec/weights identity
+ SpeculativeSpec.version + q representation/recompute mode
+ proposal/rejection/bonus RNG draw domains
```

### 3.3 Distributed identity

V2+ 再加入：

```text
TopologyManifestV1 + RankManifest set + CollectivePlan digest
+ tensor distribution mode + protocol/capability version
```

V2.0 合入 V1.3 时，manifest 必须声明 `v1.3-merged` profile，并携带 3.2 全部 identity；否则不得读取/复用 speculative artifact。V3+ 再加入 source/destination worker generation、transfer protocol、AdmissionBundle 与 destination KV reservation identity。

任一字段变化时，受影响 artifact 必须显式迁移、失效重建或 fail closed，不能静默按新语义读取。

### 3.4 Metric identity

`MetricRegistryV1` 中每个 ID 同时冻结 measurement boundary、单位、aggregation、weighting、分母、censor/failure 规则与 schema version。V1.0 及以后所有 ExperimentCard、SLO、图表和 release gate 必须引用形如 `illm.server.ttft.request_weighted.p99.v1` 的 fully-qualified ID；`p99_ttft`、`request_failure_rate` 等短名只能作为 V0 学习期显示别名，不能进入 V1+ evidence。若数学定义变化，创建新版本 ID；不得在原 ID 下替换口径。

V1 latency ID 的规范单位是 SI 秒（TPOT 为 s/token）。TTFT 在 accepted request 实际产生首 token 时形成 request-weighted 样本；ITL 对 accepted request 已形成的全部相邻 output-token intervals 做 token weighting；TPOT 在 accepted 且 output≥2 时形成 request-weighted 样本。上述样本不因请求随后失败、取消或超时而删除，未形成的样本进入对应 censor counter。server/client E2E 覆盖全部 accepted+terminal 请求并按 terminal outcome 分层；pre-accept rejection 只进入 offered/admission 指标。UI 可以显示换算后的 ms，但 metric、threshold 与 evidence 仍必须保存规范秒值。

## 4. Ownership and linearization chain

### 4.1 V0.0～V0.3

```text
Engine owns RequestState
Scheduler owns no mutable engine state; it returns immutable SchedulePlan
Capture owns serialized facts, never live engine objects
```

V0.3 的 accepted point 是资源 reservation 与 `request.accepted` mutation 的同一原子提交。accepted 前失败是 rejection；accepted 后必须进入唯一 terminal 仲裁。

### 4.2 V0.4～V1.1

```text
KVAllocator owns physical pages and allocation_epoch
Request/PageTable owns logical mapping through generation-checked PageHandle
PreparedBatch owns input/output/workspace leases
ExecutionTicket owns CompletionFence + KVWriteLease until resolve
PrefixIndex owns lookup metadata; cache_owner is separate from request/read/write counts
GraphInstance owns graph-pinned buffers until destruction
```

allocation state、cache metadata、device validity、request lease 与 in-flight reader/writer 必须正交。V0.7 eviction 的唯一顺序为：锁定并验证 entry/version、`request_lease_count=0`、`inflight_reader_writer_count=0` 与 fences complete → 从 index 删除（逻辑 eviction 线性化点）→ 清除 cache owner/identity → allocation epoch 前进并 FREE。删除后新 acquire 因 entry/version stale 失败。cache owner 自身不应阻止驱逐；active request/in-flight/fence 必须在删 index 前阻止 eviction。

cancel 只能逻辑抑制未提交输出；最后一个 reader/writer fence 完成前，不得回收地址、发布 prefix entry 或让下一 allocation 复用旧 handle。

`ResumeCursorV1` 是单机 recompute/resume 的规范 cursor，固定 logical history、KV valid-through、`next_output_position`、pending-input、base-next-logits 与 request generation。V3.0 的 `DecodeCursorV1` 在跨 Prefill/Decode handoff 中继承这些字段和“一步 KV/output lag”含义，再增加 source/destination generation、transfer/payload 与 distributed identity；不得把同名 count 重新解释为“已 stream 数”或假设 KV valid count 等于 committed output count。

### 4.3 V1.2～V1.3

```text
Request owns committed grammar/DFA state
SpecTxnV1 owns proposal tokens, p/q evidence, shadow grammar state and RNG draws
Target/draft KV each expose explicit valid_through and transaction generation
```

proposal/verify 期间只写 shadow state。一次 transaction 在同一 boundary 原子提交 token、各 KV 的 **valid-through/ownership metadata**、grammar state、RNG commit cursor 和 output stream；这不表示刚产生的 residual/bonus 已物化为 KV，它们仍可处于 pending-input。reject/cancel/error 只保留已提交前缀并 rollback 其余状态。

### 4.4 V2.0～V2.1

```text
RankGroup owns collective epoch and sharding manifests
TP leader owns full-logits sampling decision provisionally
GroupReservationV1 owns one TP group-local capacity claim and accepted mutation
All-rank acknowledgement commits the sampled token and each rank KV valid-through metadata
Router owns routing intent, not worker KV
AdmissionLease owns one replica capacity claim
LeaseOutcomeCertificate fences unknown old attempts before cross-replica retry
```

TP leader 不得在任一 rank 失败/超时前对外提交 token；all-rank commit 仍服从 KV/output 一步 lag，新 sampled token 可以是 pending-input。V2.1 的 durable AdmissionLease acquire/CAS 必须同时写入 lease、底层 GroupReservation、request accepted mutation 与 chosen replica；这是唯一 accepted 线性化点。timeout/unknown 只能查询原 authority；换 replica 前必须取得旧 authority 的 durable `REJECTED/FENCED_NO_ACCEPT` certificate。

### 4.5 V3.0～V3.1

```text
AdmissionBundle atomically owns AdmissionLease + FutureDecodeCredit + accepted mutation
Prefill worker owns source KV until transfer handoff resolves
Decode worker owns destination reservation after validated conversion
DecodeCursorV1 owns prompt-KV valid-through, base-next-logits and pending-input position
TransferTicket owns in-flight KV bytes and completion/abort
OperationJournal owns durable operation phase and recovery decision
CoordinatorEpoch + MutationLease fence every lifecycle mutation
```

AdmissionBundle 的 durable atomic commit 是 V3 唯一 accepted point；任一 lease/credit 部分失败都全部 rollback。transfer bytes、base logits、DecodeCursor 完整校验、layout conversion 和 destination page commit 完成前，destination KV 不得标 VALID。所有 hop 使用同一 DeadlineBudget authority deadline，transit/retry 不得延长预算。

V3.1 drain acknowledgement 的线性化点是：新 admission attempt 已被 epoch-fenced、路由已移除、所有先前 attempt 已原子决议为 `ACCEPTED_BUNDLE` 或 `REJECTED/FENCED_NO_ACCEPT`，且状态持久写入 journal；P/D accepted 决议必须包含 FutureDecodeCredit。coordinator restart 只能依据 durable phase/epoch/lease 恢复；无法证明唯一 owner 时进入 safe-pause/fail-stop。

## 5. Terminal outcome contract

从 V0.2 起，任一 `request.accepted` 必须且只能产生一个公开 terminal outcome：

```text
FINISHED | CANCELLED | FAILED | DEADLINE_EXCEEDED
```

`REJECTED` 只发生在 accepted 前，不是 terminal outcome。`COMMITTED`、`ABORTED`、`VALID` 等可以是 plan/ticket/page 的内部状态，但不得作为 request terminal 名称。transport 断开、client cancel、deadline、rank failure、transfer failure 与 executor completion 的 race 由一个原子 terminal transition 仲裁。

版本暂未启用 deadline 时仍须保留 `DEADLINE_EXCEEDED` 数值/字符串，不得将相同 code 复用于其他状态。

## 6. Failure semantics evolution

| Failure | 首次要求 | 后续增强 | 永久禁止 |
|---|---|---|---|
| invalid/unsupported input | V0.1 | API 映射与 capability detail | 静默忽略参数 |
| client cancel | V0.2 | V0.5 fence；V3 跨服务传播 | in-flight buffer 提前复用 |
| overload | V0.3 | V2.1 replica lease；V3 双资源 credit | 无限队列或 accepted 后转 rejection |
| OOM/capacity miss | V0.2 | V0.4 MemoryPlan；V2 per-rank；V3 per-role | 以 CUDA OOM 作为正常流控 |
| kernel async error | V0.4 | V1.0 worker health；V3 replacement | 未归属 step/request 的 silent corruption |
| preemption liveness | V0.5 | V0.6 decode reservation；V0.7 cache pressure | 无上限重复抢占 |
| rank failure | V2.0 | V3.1 replacement/rollback | 其余 rank 永久 hang 或 leader 提前 commit |
| duplicate admission | V2.1 | V3 AdmissionBundle；V3.1 durable fencing | 同一请求多处 accepted |
| KV transfer failure | V3.0 | V3.1 recovery/runbook | 半传输 KV 标 VALID |
| coordinator restart | V3.1 | journal replay/safe-pause | 仅凭内存/heartbeat 猜 owner |
| exporter/UI failure | V0.0 | 全版本 bounded degradation | 使推理失败 |
| incompatible version | V0.0 | V2/V3 handshake | 猜测/最佳努力继续 |

V3 的 deadline 使用入口的单调绝对 deadline，并在每个 queue/transfer/retry 阶段扣减 `remaining_budget_ns`；不得在重试时重置预算。默认最多一次有界 transfer retry，除非新 SPEC 给出更严格证明。

## 7. Reference paths that must not disappear

| Reference | 从何时保留 | 验证对象 |
|---|---|---|
| Transformers/per-op oracle | V0.1 | model/loader/logits |
| full-context cache-off runner | V0.2 | KV/prefill/decode |
| max-active=1 runner | V0.3 | batching/scheduler |
| contiguous KV + gather attention | V0.4 | paged allocator/backend |
| no-preemption mode | V0.5 | victim/recompute/fence |
| full-prefill mode | V0.6 | chunk/decode reservation |
| no-prefix mode | V0.7 | radix/lease/eviction |
| canonical reducer + CLI | V0.8 | UI/R1 replay |
| eager/sync execution | V1.1 | selected fast path |
| unconstrained sampler | V1.2 | grammar mask/DFA |
| target-only decode | V1.3 | proposal/verify/rollback |
| single-GPU runner | V2.0 | TP/collective |
| one TP group direct route | V2.1 | DP routing/admission |
| unified engine | V3.0 | P/D disaggregation |
| previous stable binary | V3.1 | upgrade/rollback |

移除 reference path 需要新的 major SPEC 和替代 oracle；“维护麻烦”不是充分理由。

## 8. Replay capability evolution

能力必须由 Capture Bundle manifest 显式声明；不支持的 mode fail closed。表中“保留 reader”表示可读旧 artifact，不表示新状态已可 R1。

| Version | R0 Event Playback | R1 Control-State Replay | R2 Numerical Re-execution | R3 Policy Simulation |
|---|---|---|---|---|
| V0.0 | toy | toy | 不支持 | 不支持 |
| V0.1 | model/sampling events | limited request/model/sampling | 不支持 | 不支持 |
| V0.2～V0.7 | 新状态支持 R0 | 只保留旧 limited reader；新 KV/scheduler/cache 不声明 R1 | 不支持 | 不支持 |
| V0.8 | full-engine candidate | full-engine candidate + 三视图 | 不支持 | 不支持 |
| V1.0～V1.1 | stable | stable | 不支持 | 不支持 |
| V1.2 | grammar events/state | grammar control state | 不支持 | 不支持 |
| V1.3 | speculation events/state | transaction control state | 仅 encrypted-token opt-in qualification profile | 不支持 |
| V2.0～V2.1 | rank/router events | rank/router control state | 仅 `v1.3-merged` + encrypted profile 继承；TP/DP 本身不新增 R2 | 不支持 |
| V3.0～V3.1 | cross-service/lifecycle events | admission/transfer/operation control state | 不新增；仅可显式继承 `v1.3-merged + encrypted_tokens` qualification profile | 不支持 |

本列车没有正式 R3。Scheduler Lab、capacity lab 或 topology UI 给出的离线结果只能标 `estimated`/`what_if_preview`，且必须由真实 workload 验证；不能因为界面能调参数就宣称 R3。

## 9. Compatibility guarantees

### 9.1 V0.x

- contract 可以演进，但必须携带 version/digest；
- `format_frozen_for_v0` 对象不得在后续 V0 复用 code 或暗改字段含义；
- reader/loader 不支持时返回结构化缺失能力与建议版本；
- GPU/cache artifact 不承诺自动迁移，identity 改变时失效重建。

### 9.2 V1.x

- public API/CLI、terminal semantics、model/sampling/event identity 在同 major 内兼容；
- capture 至少读取前一个 minor 的 R0/R1；
- config 新字段默认 optional，未知 required semantics fail closed；
- V1.2/V1.3 的可选契约只有被 manifest 声明时才参与 compatibility，未声明不能猜测启用。

### 9.3 V2/V3

- collective、routing、transfer、lifecycle protocol 使用 capability handshake；
- mixed-version upgrade 只在 `SupportMatrix` 声明窗口允许；
- rollback 包含 binary、config、schema/journal、protocol、traffic drain 与 cache invalidation；
- 性能 no-go 可发布功能/研究版，但 correctness、ownership、fault、required-resource 或 replay gate 失败必须阻断资格发布。

## 10. Contract change checklist

任何 Version SPEC 从 draft 进入 accepted 前逐项检查：

- [ ] 是否改变 identity、RNG draw domain 或 artifact invalidation？
- [ ] 是否改变 owner、lease、epoch、fence 或线性化点？
- [ ] 是否改变 public API/CLI/error/terminal？
- [ ] 是否增加 required event/capture 字段？
- [ ] 是否影响旧 capture reader 或 replay capability 声明？
- [ ] 是否改变 MemoryPlan、admission、deadline 或 backpressure？
- [ ] 是否需要 Feature Design、ADR、migration fixture 或 rollback rehearsal？
- [ ] 是否保留可独立验证的新旧 reference path？
- [ ] optional branch 是否写入 manifest identity 与独立 qualification profile？

任一答案为“是”而 Version SPEC 未给出 compatibility、rollback、evidence 与 owner 时，SPEC 不能进入 accepted。
