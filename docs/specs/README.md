# iLLM Version SPEC Index

本目录把[项目总方案](../../AI_INFERENCE_ENGINE_PROJECT_PLAN.md)展开为 17 个可独立学习、实现、演示和发布的版本。总方案负责长期边界与架构原则；Version SPEC 负责一个 release 的唯一增量、契约、切片、测试、迁移、回滚和可执行 DoD。

第一次接触术语时先读[统一学习术语表](GLOSSARY.md)，跨版本对象的成熟度、identity、ownership、terminal 与 replay 能力以[跨版本契约账本](CROSS_VERSION_CONTRACTS.md)为准。

## 1. 为什么采用细粒度版本列车

推理引擎的学习依赖前一层正确性。若把分页、抢占、分块、前缀缓存、可视化和 CUDA 优化塞进同一 release，一旦结果错误，很难确定故障层。版本列车遵循：

```text
事实源 → 数值 oracle → 增量状态 → 多请求调度
→ 分页 → 抢占 → 分块 → 前缀复用 → 可视化回放
→ 单卡稳定 → 单假设性能季 → 可选生成能力
→ TP → DP 路由 → P/D 分离 → 可靠运行
```

每个版本都必须能安装、能运行、能观察、能验证、能回退。尚未达到本版 DoD 的代码只能叫 dev snapshot，不能叫 release。

## 2. 完整版本列车

下表按每周 10～15 小时估算。增量包含该版本的学习、实现、测试和 evidence，不包含长时间等待稀缺 GPU；第一次处理驱动、CUDA、NCCL 或分布式故障时，日历时间可按 1.5～2 倍规划。

| 版本 | 名称 | 阶段 | 本版唯一主要交付 | 直接依赖 | 预计增量 |
|---|---|---|---|---|---:|
| [V0.0](versions/v0.0-bootstrap-lab.md) | Bootstrap Lab | M0 | 空仓库变为可复现实验室 | 无 | 1～2 周 |
| [V0.1](versions/v0.1-reference-runner.md) | Reference Runner | M1 | full-context 数值与采样 oracle | V0.0 | 2～3 周 |
| [V0.2](versions/v0.2-stateful-streaming.md) | Stateful Streaming | M2 | prefill/decode、contiguous KV、stream/cancel | V0.1 | 2～3 周 |
| [V0.3](versions/v0.3-batched-engine.md) | Batched Engine | M3 | continuous batching 与 hard admission | V0.2 | 3～4 周 |
| [V0.4](versions/v0.4-paged-memory.md) | Paged Memory | M4a | paged allocator 与 Attention ABI | V0.3 | 3～4 周 |
| [V0.5](versions/v0.5-preemptible-memory.md) | Preemptible Memory | M4b | recompute preemption 与安全延迟回收 | V0.4 | 1～2 周 |
| [V0.6](versions/v0.6-chunked-prefill.md) | Chunked Prefill | M5a | mixed AttentionPlan 与 decode reservation | V0.5 | 2～3 周 |
| [V0.7](versions/v0.7-prefix-aware-core.md) | Prefix-Aware Core | M5b | exact radix prefix reuse | V0.6 | 3～4 周 |
| [V0.8](versions/v0.8-visual-replay.md) | Visual Replay | M6 | full-engine R0/R1 与三个核心视图 | V0.7 | 4～6 周 |
| [V1.0](versions/v1.0-stable-single-gpu.md) | Stable Single-GPU | M7 Gate + M8 | 有限支持矩阵内的公开稳定发布 | V0.8 | 3～4 周 |
| [V1.1](versions/v1.1-performance-season.md) | Performance Season | 选零或一个 M7a～M7d | profile-gated 单假设实验与外部差距报告 | V1.0 | 2～5 周 |
| [V1.2](versions/v1.2-structured-outputs.md) | Structured Outputs | M9a，可选 | byte-level regex/有限 JSON 约束解码 | V1.1 | 2～3 周 |
| [V1.3](versions/v1.3-speculative-decoding.md) | Speculative Decoding | M9b，可选 | lossless proposal/verify/rollback transaction | V1.2 | 3～5 周 |
| [V2.0](versions/v2.0-single-node-tensor-parallel.md) | Single-Node Tensor Parallel | M10a | 单机 TP、collective 与 rank 一致性 | V1.1 | 4～6 周 |
| [V2.1](versions/v2.1-data-parallel-routing.md) | Data Parallel Routing | M10b | 多 replica 路由与原子 admission lease | V2.0 | 2～4 周 |
| [V3.0](versions/v3.0-disaggregated-runtime.md) | Disaggregated Runtime | M11 | P/D 分离、KV transfer 与双资源接单 | V2.1 | 8～12+ 周 |
| [V3.1](versions/v3.1-reliable-operations.md) | Reliable Operations | M12 | drain、升级/回滚、chaos/soak 与 support matrix | V3.0 | 8～14 周 |

V1.0 累计约 24～35 周。主线继续走完 V1.1、V2.0、V2.1、V3.0 和 V3.1，累计约 48～76+ 周；再选做 V1.2/V1.3，增加约 5～8 周。V0.8 虽然只做薄只读 UI，但 reducer、bundle、隐私/损坏处理和三视图仍需独立验证；V3.1 的 S1～S8 是可靠性 release checkpoints，因此都按新手可留出故障排查和长测等待的现实周期估算。V1.0 已是完整学习成果，分布式不应成为“项目才算成功”的前提。

## 3. 依赖图与可选分支

```mermaid
flowchart LR
    V00[V0.0 Bootstrap] --> V01[V0.1 Reference]
    V01 --> V02[V0.2 Stateful]
    V02 --> V03[V0.3 Batched]
    V03 --> V04[V0.4 Paged]
    V04 --> V05[V0.5 Preemptible]
    V05 --> V06[V0.6 Chunked Prefill]
    V06 --> V07[V0.7 Prefix-Aware]
    V07 --> V08[V0.8 Visual Replay]
    V08 --> V10[V1.0 Stable Single-GPU]
    V10 --> V11[V1.1 Performance Season]
    V11 --> V12[V1.2 Structured optional]
    V12 --> V13[V1.3 Speculative optional]
    V11 --> V20[V2.0 Tensor Parallel]
    V13 -. optional v1.3-merged profile .-> V20
    V20 --> V21[V2.1 DP Routing]
    V21 --> V30[V3.0 Disaggregated]
    V30 --> V31[V3.1 Reliable Operations]
```

V1.1 无论结果为 `NoCandidate`、`Adopt` 或 `No-go` 都发布 v1.1.0：`selected_branch` 的基数是 0..1，不允许为了版本号堆叠多个 fast path。V1.2/V1.3 是可选生成能力线，不是 V2.0 的硬依赖。若 V2.0 合入 V1.3，必须启用 SPEC 中单独定义的 `v1.3-merged` qualification profile，并把 grammar、draft model、RNG 与 transaction identity 写入 manifest；未合入时保持 V1.1 语义。

## 4. 各版本唯一新增复杂度

| 版本 | 唯一主要新复杂度 | 永久保留的 oracle/fallback |
|---|---|---|
| V0.0 | 事实事件与可回放 toy 状态 | pure reducer + toy scheduler |
| V0.1 | Transformer 与采样数值语义 | Transformers/per-op golden |
| V0.2 | 增量请求状态和 KV | cache-off full-context runner |
| V0.3 | 多请求 iteration scheduling | max-active=1 runner |
| V0.4 | 非连续物理 KV | contiguous KV + gather attention |
| V0.5 | preemption 与异步安全释放 | no-preemption mode |
| V0.6 | chunked prefill | full-prefill mode |
| V0.7 | 跨请求 exact prefix reuse | no-prefix mode |
| V0.8 | full-engine 控制面回放与多视图 | canonical reducer + CLI |
| V1.0 | 公开稳定契约与发布工程 | local facade + eager reference |
| V1.1 | 至多一个 profile 选择的 fast path | 对应 reference mode |
| V1.2 | grammar-constrained sampling | unconstrained sampler |
| V1.3 | speculative transaction | target-only decode |
| V2.0 | tensor parallel rank/collective | single-GPU runner |
| V2.1 | replica routing/admission | one TP group direct route |
| V3.0 | 跨服务 KV 与部分失败 | unified engine |
| V3.1 | 变更管理与持久故障恢复 | pinned previous stable release |

若一个 release 同时引入两个以上主要复杂度，应拆版，或用 ADR 证明它们无法安全解耦。

## 5. 统一启动方式

V0.0 必须提供两个跨平台入口：bootstrap 只负责建立并证明环境，runner 负责在该环境内执行命令。不能依赖 shell activation，也不能预设全局 `uv` 已安装。

```text
python scripts/bootstrap.py --profile dev --locked --write-manifest out/toolchain-manifest.json
python scripts/in_env.py --manifest out/toolchain-manifest.json -- python -m pytest tests/cpu
python scripts/in_env.py --manifest out/toolchain-manifest.json -- python -m illm demo toy-scheduler
```

`bootstrap.py` 负责检查受支持 Python、创建隔离环境、按锁文件安装工具，并原子写入 finalized `ToolchainManifest`。manifest 至少记录 schema、environment root、环境内解释器/entrypoint、OS/arch、lock/bootstrap/project-root digest 和 environment ID。V1+ 正式资格必须在同一次 bootstrap 中传 `--artifact <exact-wheel-or-container> --artifact-manifest <signed-manifest>`；bootstrap 先核验 checksum/signature，再安装候选，并把 artifact digest/entrypoint 写入 finalized manifest。禁止在 manifest finalized 后另跑 `pip install` 改变环境再沿用旧 digest；CPU/source learning profile 必须显式标为 source profile，不能据此证明 release artifact。

`in_env.py` 由系统 Python 启动只为读取 manifest；它必须把命令中的 `python` 改写为 manifest 指定解释器，把 console entrypoint 解析到 environment root 内，并先用子进程握手验证 `sys.executable + environment_id + lock_digest`。manifest 未完成、digest 漂移、解释器/entrypoint 逃出 environment root、环境握手不一致或只在全局 PATH 找到命令时，runner 必须在项目代码运行前返回稳定非零退出码。runner 不使用 shell 字符串拼接，参数在 `--` 后逐项传递，并原样返回子进程退出码。

后续 SPEC 中出现的 `uv run`、`pytest`、`illm` 或 GPU 命令，写入 clean-room evidence 时一律转换为 `scripts/in_env.py --manifest ... -- ...` 形式。CI、release demo、rollback 和独立复核都必须从 bootstrap 开始；直接激活环境只可作为开发便利，不能成为资格证据。

## 6. 学习门、硬件资格与状态分层

硬件不足不应阻断学习，但不能把 simulator/CPU 结果冒充真实硬件资格。发布事实使用六个正交轴，禁止用一个模糊的 `status` 同时表达它们：

| 轴 | 允许值 | 只回答的问题 |
|---|---|---|
| `spec_status` | `draft/accepted/implemented/released/superseded` | Version SPEC 走到哪个文档生命周期阶段 |
| `release_stage` | `dev_snapshot/release_candidate/released/withdrawn` | 当前 artifact 是开发快照、RC、正式发布还是已撤回 |
| `qualification_tier` | `unqualified/learning_complete/functionally_qualified/stable_released` | 当前 evidence 证明到哪一层资格 |
| `qualification_profiles[].result` | `not_run/pass/fail/invalid/not_run_resource_unavailable/not_applicable` | 每个固定 CPU/GPU/拓扑 profile 实际跑出了什么结果 |
| `performance_outcome` | `not_applicable/pending/NoCandidate/Adopt/No-go` | 预注册性能季的结论是什么 |
| `lts_state` | `not_applicable/candidate/promoted/eol` | 是否进入、完成或结束 LTS 维护承诺 |

`qualification_tier` 的层级含义如下；它与性能和 LTS 轴互不替代：

| 资格层级 | 进入条件 | 不能声称什么 |
|---|---|---|
| `learning_complete` | 预热、toy/simulator、Learning Demo、Teach-back 与对应测试完成 | 未证明正式支持矩阵 |
| `functionally_qualified` | correctness/failure/compatibility required profiles 在声明硬件上全部 pass | 未证明性能优于 baseline，也未自动获得稳定发布 |
| `stable_released` | clean-room、全部 required profiles、support matrix、rollback、tag/evidence 和独立复核完整 | 未列入矩阵的组合仍不受支持 |

NoCandidate/No-go 可以与 `functionally_qualified` 或 `stable_released` 同时成立；Adopt 也不能覆盖失败或未运行的 correctness profile。`release_stage=released` 不等于 `qualification_tier=stable_released`，`lts_state=candidate` 也不等于已 promotion。

ExperimentCard 内的规则键统一写 `success` / `no_go`，但签发到 `performance_outcome` 时分别规范化为 `Adopt` / `No-go`；只有 Profile Gate 未选中任何 candidate 时使用 `NoCandidate`。draft/运行中为 `pending`，完全没有性能决策的版本才是 `not_applicable`。No-go 是否允许功能版降级发布仍由对应 Version SPEC 明确，不能从枚举本身猜测。

每个 `ReleaseManifest` 必须原样保存六轴及逐 profile 证据，不允许只保存聚合 pass：

```yaml
spec_status: implemented
release_stage: release_candidate
qualification_tier: learning_complete
qualification_profiles:
  - profile_id: cpu-learning-v1
    result: pass
    environment_manifest_digest: sha256:...
    evidence_digest: sha256:...
    reason: null
  - profile_id: single-gpu-qualification-v1
    result: not_run_resource_unavailable
    environment_manifest_digest: null
    evidence_digest: sha256:...
    reason: fixed_runner_unavailable
performance_outcome: not_applicable
lts_state: not_applicable
```

`illm release verify` 必须从 profile/evidence 重算可声明的最高 tier；claim 高于证据、required profile 为 fail/invalid/not-run、或缺少 reason/digest 时 fail closed。现有元数据中的 `status` 只是校验器兼容镜像，必须等于 `spec_status`，不得进入 ReleaseManifest 成为第七个含混轴。

资源缺失统一记为 `not_run_resource_unavailable`，绝不能记 pass。推荐资源矩阵：

| 版本段 | 最低学习路径 | 正式 qualification | 无正式资源时允许交付 |
|---|---|---|---|
| V0.0 | CPU tiny fixture | CPU clean-room CI | `learning_complete` |
| V0.1～V0.3 | CPU tiny + 可选缩小 real fixture | 固定单 GPU real fixture；CPU tiny 必过 | CPU oracle、状态机、capture 与缺失资源记录 |
| V0.4～V0.8 | CPU allocator/scheduler/reducer simulator | 固定单 GPU；V0.8 UI 可离线读正式 capture | simulator 教学证据，不授予 GPU qualification |
| V1.0 | CPU 全套回归 | 一张固定 NVIDIA GPU 上跑有限 `qualification-matrix.yaml` | release candidate，不能标 `stable_released` |
| V1.1 | 已保存的 trace 可做离线 profile 练习 | 与 frozen vLLM/SGLang baseline 相同的固定 GPU、预算和功耗条件 | gap plan/离线诊断；不得声称 Adopt |
| V1.2 | CPU grammar corpus/DFA property test | 单 GPU 集成与超时/配额矩阵 | grammar library 的学习完成 |
| V1.3 | CPU transaction simulator | 能同时容纳固定 target/draft 的单 GPU，或 SPEC 声明的等价固定拓扑 | proposal/rollback 模拟，不声称端到端 lossless qualification |
| V2.0 | CPU collective simulator | 至少两张同构 GPU，固定互联拓扑 | design/simulator artifact，不是 TP release |
| V2.1 | CPU router simulator | 至少两个可独立 admission 的 replica/TP group | 路由性质证明，不是 DP release |
| V3.0 | mock transfer + fault simulator | 至少一个 prefill worker 和一个 decode worker 的声明 GPU 拓扑 | wire/failure 学习包，不是分离式 release |
| V3.1 | journal/recovery simulator | support matrix 每种拓扑上至少一条升级和一条回滚路径，并完成 soak/chaos | runbook 演练；不能授予稳定/LTS 状态 |

V3.1 首次发布只能是 LTS candidate。只有 support matrix 全门通过、72 小时 soak/chaos 窗口完成、升级/回滚 evidence 经独立复核后，才按该 SPEC 的 promotion policy 生成新的签名 promotion attestation（或新 patch tag）并更新支持索引；原 candidate manifest/checksum 永不原地改写。缩短拓扑的学习演练不能替代这一步。

## 7. 单人项目的角色分工

SPEC 中出现多个角色不代表必须组队。单人开发时同一人可以依次戴不同“帽子”，但 evidence 要留下独立检查边界：

| 规范 ID | 责任 | 单人执行建议 |
|---|---|---|
| `project-owner` / `spec-owner` | 冻结 scope、contract、DoD 与跨域取舍 | 开工前完成 accepted review；二者默认由同一帽子承担 |
| `implementer` | 写代码、测试与迁移 | 不在实验后改预注册阈值 |
| `experiment-owner` / `performance-owner` | 固定 workload/baseline 并运行 | 保存 raw data、无效 run 和失败 run |
| `release-owner` | clean-room、rollback、tag 与 promotion artifact | 使用新环境，从文档命令重放 |
| `independent-verifier` | 核对 evidence 与结论 | 可由另一人、干净 CI、或不共享临时状态的新会话承担 |

`model-owner`、`grammar-owner`、`sampling-owner`、`memory-owner`、`executor-owner`、`architecture-owner`、`reliability-owner`、`security-owner` 是领域帽子，不暗示必须新增成员；它们在当版分别向 `spec-owner` 或 `implementer` 负责。owner ID 一律用 kebab-case，空格写法在 SPEC accepted 前规范化。单人不能省略 independent boundary：作者可切换帽子，但 verifier 必须只拿 finalized artifact、命令与 runbook，不复用作者的临时环境状态。

## 8. SPEC 生命周期

```text
draft → accepted → implemented → released → superseded
```

这是 `spec_status`，不是 artifact 的 release stage、qualification、performance 或 LTS 状态。

| `spec_status` | 进入条件 | 允许变更 |
|---|---|---|
| `draft` | 开始规划 | 任意；保留有答案选项的 Open Questions |
| `accepted` | Scope、Contracts、DoD、Non-goals 已审查 | 只做澄清；语义变化写 amendment/ADR |
| `implemented` | 实现与自动化 gates 通过 | 补 evidence、限制和已实现偏差 |
| `released` | clean install、qualification、evidence bundle、tag 完成 | 需求冻结；勘误单独记录 |
| `superseded` | successor 已发布 | 只补 successor 链接 |

推荐只把“当前版本”写到 implementation-ready，把“下一版本”写到 accepted-ready，更后版本保持全局契约与边界可审查但不冻结代码细节。公开 API、所有权、并发、持久 artifact、GPU layout 或跨进程协议发生变化时，必须先写 Feature Design/ADR，再修改 accepted SPEC。

## 9. 文档职责

| 文档 | 回答的问题 | 发布是否必需 |
|---|---|:---:|
| Project Plan | 为什么做、长期边界和总体路线是什么 | ✓ |
| Version SPEC | 本版新增什么、如何验收和回退 | ✓ |
| Feature Design | 复杂机制具体如何实现 | 条件式 |
| ADR | 为什么选择这个方案而非替代方案 | 条件式 |
| Experiment Card | 可证伪假设、指标、预算和退出门是什么 | 性能版必需 |
| Evidence Bundle | 第三方如何复现本版结论 | ✓ |
| Release Notes | 实际发布了什么、兼容与限制是什么 | ✓ |

只修 bug 且不改变契约的 patch release 不需要新 Version SPEC；若所谓 patch 改变公开语义、schema 或 compatibility，它应成为 minor release 或 accepted amendment。

## 10. Release Gate 顺序

```text
Spec accepted
→ canonical bootstrap / clean build
→ correctness / golden / property
→ failure / stress
→ compatibility / migration / rollback rehearsal
→ performance / overhead（适用时）
→ tutorial / visual QA
→ independent evidence audit
→ recompute six-axis release state
→ tag + release notes
```

correctness 永远早于 performance。V1.1、V2.0、V2.1 与 V3.0 的探索性性能门允许 no-go：可信的负结果可以发布，功能正确性版本不应因“没有赢”被无限延期；但缺少该版本 required hardware 的正确性门不能被 no-go 替代。V3.1 的 reliability-overhead 是稳定发布 guardrail，是否阻断以其 Version SPEC 为准，不能从这一通用规则外推。

## 11. 统一 Evidence Bundle

每个 release 至少产生：

```text
release/版本号/
├─ manifest.json
├─ commands.txt
├─ test-report.json
├─ qualification/
│  ├─ release-state.json
│  └─ profiles/
├─ rollback/
│  ├─ manifest.json
│  ├─ commands.txt
│  └─ rollback-report.md
├─ capture/
├─ benchmark/
├─ screenshots/
├─ teach-back.md
├─ known-limitations.md
└─ release-notes.md
```

`rollback/manifest.json` 绑定 source/target binary、config、lock、model/weights、schema/support matrix digest 和 previous release evidence；report 保存 canary、terminal/resource 终态与结论。V2+ 增加 topology、rank manifests 和 communication trace；V3+ 增加 admission/transfer、failure/chaos、upgrade/rollback、operation journal 与 capacity evidence。所有结果写清 `measured`、`estimated`、`not_run_resource_unavailable`，不允许静默降级 replay mode 或 qualification level。

## 12. “稳定可靠”的最终定义

V3.1 的稳定可靠是有限 support matrix 内可验证的 runtime 属性：正确性和容量可复现；接单、取消、OOM、rank/service failure 只有一种语义；drain/替换/升级/回滚不制造重复 terminal 或失效 KV；协议不兼容 fail closed；soak/chaos 无泄漏、死锁、无界队列或 silent corruption；观测、告警、runbook、capacity plan 和 release evidence 完整。

它不包括计费、商业 SLA、租户/IAM 平台、Kubernetes 平台建设或全云自动运维。

## 13. 从哪里开始

先阅读并审查 V0.0 与 V0.1，但只实现 [V0.0](versions/v0.0-bootstrap-lab.md)。V0.0 的 release evidence 完成后，再根据真实环境修订并接受 V0.1；不要同时启动分页、CUDA 或 UI。

新版本从 [SPEC_TEMPLATE.md](SPEC_TEMPLATE.md) 复制，保留 13 个主章节、Appendix A/B、requirement traceability、Open Questions、rollback 和 evidence。提交前在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validate_specs.ps1 -RepositoryRoot .
```

校验器检查完整 17-version release train、元数据与依赖、15 个必需章节、requirement ID、占位符、代码围栏、编码、尾随空白和本地 Markdown 链接。它证明结构闭环，但不能替代技术语义与证据的人工审查。
