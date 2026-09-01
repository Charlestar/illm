# iLLM <VERSION> <RELEASE_NAME> SPEC

```yaml
status: draft                 # deprecated validator mirror; MUST equal spec_status
spec_status: draft            # draft | accepted | implemented | released | superseded
release_stage: dev_snapshot   # dev_snapshot | release_candidate | released | withdrawn
qualification_tier: unqualified # unqualified | learning_complete | functionally_qualified | stable_released
qualification_profiles:
  - profile_id: <PROFILE_ID>
    result: not_run           # not_run | pass | fail | invalid | not_run_resource_unavailable | not_applicable
    reason: spec_not_implemented
performance_outcome: not_applicable # not_applicable | pending | NoCandidate | Adopt | No-go
lts_state: not_applicable     # not_applicable | candidate | promoted | eol
spec_version: 1
target_release: <VERSION>
release_type: learning        # foundation | learning | stable | performance | optional | scale | reliability
depends_on: [<PREVIOUS_VERSION>]
project_plan: ../../../AI_INFERENCE_ENGINE_PROJECT_PLAN.md
previous_spec: ./<PREVIOUS_SPEC_FILE_OR_NONE>
owners: [project-owner]
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
```

> 本模板约束一个可独立安装、运行、演示、验证和复盘的版本。它只描述相对上个 release 的增量；跨版本长期原则以项目总方案和已接受 ADR 为准。
>
> 本模板按复制到 `docs/specs/versions/` 后的路径书写；不要从模板当前位置解释 `project_plan`、`previous_spec` 或下文术语表路径。`status` 只是现有校验器的兼容镜像，发布事实只读取六个正交轴。

## 1. 本版本学习目标

完成后，开发者应能解释、实现并验证哪些 AI Infra 概念。目标应使用“能够解释/实现/测量/诊断”，而不是“了解”。

### 1.1 必须掌握

- <LEARNING_OUTCOME>

### 1.2 建议先修

- <PREREQUISITE>

### 1.3 进入门槛与预热实验

| 预热实验 | 命令/材料 | 通过标准 | 预计时间 |
|---|---|---|---:|
| <WARMUP_LAB> | <COMMAND_OR_LINK> | 能解释输入、状态变化、输出和一个失败边界 | <HOURS> |

陌生术语先查仓库中的 `docs/specs/GLOSSARY.md`（复制后相对路径为 `../GLOSSARY.md`）。预热实验不属于 release 功能，但未通过时不应开始本版实现。

### 1.4 Teach-back 产物

- <ONE_PAGE_NOTE_OR_DEMO>

## 2. 用户可观察的成果

### 2.1 Release demo

```text
python scripts/bootstrap.py --profile <PROFILE> --locked --write-manifest out/<VERSION>/toolchain.json
python scripts/in_env.py --manifest out/<VERSION>/toolchain.json -- <COMMAND>
```

所有 clean-room 命令都必须经过 `scripts/in_env.py`。runner 只执行 `ToolchainManifest` 指向且经握手验证的环境内解释器/入口；manifest 缺失、未 finalized、lock/bootstrap/project-root digest 不符、解释器逃出 environment root，或命令解析到全局 PATH 时必须在执行项目前 fail closed。

V1+ 的正式 release qualification 不得隐式运行工作区源码或旧安装包；应把第一行替换为下列 artifact-bound 形式，并在 evidence bundle 中保存完全相同的 artifact manifest、digest 与签名：

```text
python scripts/bootstrap.py --profile release-verify --locked --artifact release/<VERSION>/dist/<EXACT_ARTIFACT> --artifact-manifest release/<VERSION>/dist/artifact-manifest.json --write-manifest out/<VERSION>/toolchain.json
```

### 2.2 用户故事

| ID | 用户故事 | 可观察结果 |
|---|---|---|
| US-01 | 当我…… | 我能看到…… |

### 2.3 发布资产

- 安装包或可复现环境；
- 示例配置与固定 fixture；
- capture/evidence bundle；
- 教程、已知限制和 release notes。

## 3. Scope

### 3.1 本版本新增

- <FEATURE>

### 3.2 本版本保持不变

- <CONTRACT_INHERITED_FROM_PREVIOUS_VERSION>

### 3.3 Feature flags 与 fallback

| 能力 | 默认值 | Reference/fallback | 失败时行为 |
|---|---|---|---|
| <FEATURE> | off/on | <REFERENCE_PATH> | fail closed / fallback with reason |

## 4. Non-goals

- <EXPLICITLY_EXCLUDED_WORK>

任何新增需求若改变 Non-goals、公开契约或 DoD，必须通过 spec amendment 或 ADR，而不是直接扩大 Issue。

## 5. Architecture Delta

### 5.1 相对上一版本的变化

```text
<BEFORE>
  ↓
<AFTER>
```

### 5.2 新增或修改的模块

| 模块 | 变化 | 责任边界 | 禁止事项 |
|---|---|---|---|
| <MODULE> | add/change | <RESPONSIBILITY> | <MUST_NOT_DO> |

### 5.3 数据流与状态迁移

说明新数据、状态、事务、异步边界和所有权变化；必要时使用 Mermaid。

## 6. Contracts

### 6.1 API/CLI contract

- 输入、输出、错误、取消和 streaming 行为；
- 未支持参数必须 fail closed；
- 每个 accepted request 只产生一个 terminal outcome。

### 6.2 Config/manifest contract

| 字段/对象 | 稳定性 | 兼容规则 |
|---|---|---|
| <FIELD> | `experimental` / `format_frozen_for_v0` / `candidate` / `stable` | <RULE> |

`ReleaseManifest` 必须保存六个互不替代的发布轴：

```yaml
spec_status: <SPEC_LIFECYCLE_STATE>
release_stage: <ARTIFACT_STAGE>
qualification_tier: <HIGHEST_PROVEN_TIER>
qualification_profiles:
  - profile_id: <STABLE_PROFILE_ID>
    result: <RESULT>
    environment_manifest_digest: <SHA256_OR_NA>
    evidence_digest: <SHA256_OR_NA>
    reason: <REQUIRED_WHEN_NOT_PASS>
performance_outcome: <OUTCOME_OR_NOT_APPLICABLE>
lts_state: <LTS_STATE>
```

- `spec_status` 只描述文档生命周期；`release_stage` 只描述 artifact 是 dev snapshot、RC、已发布或撤回。
- `qualification_tier` 只能由 required profile 的结果和 evidence 推导，不能因 tag、demo 或性能结果手填升级。
- `qualification_profiles[].result=not_run_resource_unavailable` 不是 pass；它允许保留学习证据，但不能满足相应 qualification gate。
- `performance_outcome` 与资格层级正交；ExperimentCard 的 `success/no_go` 签发为 `Adopt/No-go`，没有候选才用 `NoCandidate`，尚未决用 `pending`；NoCandidate/No-go 不等于功能失败，Adopt 也不自动授予稳定资格。
- `lts_state` 与普通发布正交；只有独立 promotion artifact 才能从 candidate 进入 promoted，禁止原地改写已签发 manifest。

### 6.3 Event/capture/replay contract

- 新增事件与 state digest；
- 支持的 replay 等级 R0/R1/R2/R3；
- privacy mode 与 replay capability 必须匹配；
- schema number 不复用，breaking change 有迁移或显式不兼容声明。

### 6.4 资源、并发与失败 contract

- CPU/GPU buffer、KV page、lease/fence 的 owner 与释放点；
- admission、backpressure、OOM、timeout、cancel race 和 worker failure 行为；
- 容量必须来自版本化 MemoryPlan 或等价单一事实源。

## 7. Implementation Slices

切片按可独立演示和验证的纵向能力排列，不按目录或团队横向拆分。

| Slice | 垂直成果 | depends_on | 主要文件/模块 | 完成证据 |
|---|---|---|---|---|
| S1 | <RUNNABLE_RESULT> | - | <MODULES> | command + test + capture |

每个 slice 结束必须保留 reference path；任何优化 slice 在 correctness gate 之前不得成为默认路径。

## 8. Correctness and Invariants

### 8.1 Golden/differential

- <REFERENCE_COMPARISON>

### 8.2 Property/state invariants

- <INVARIANT>

### 8.3 Failure/stress matrix

| 场景 | 注入方式 | 必须结果 | 禁止结果 |
|---|---|---|---|
| <FAILURE> | <INJECTION> | <EXPECTED> | leak/double terminal/corruption |

所有数字阈值绑定固定 fixture、seed、硬件与命令；修改阈值需新 evidence 或 ADR。

## 9. Visualization Learning Story

### 9.1 本版本必须回答的问题

1. <QUESTION>

### 9.2 视图与证据映射

| 视图 | 事实源 | 交互 | measured/estimated 边界 |
|---|---|---|---|
| <VIEW> | canonical event/metric | <ACTION> | <BOUNDARY> |

UI 不得通过重新运行调度逻辑“推测”当时原因；决策解释来自当时生成的 DecisionRecord。

## 10. Performance Hypothesis

### 10.1 本版本性能立场

选择以下之一：仅建立基线；验证一个假设；profile-gated 优化；不设性能目标。

### 10.2 Experiment card

```yaml
question: <QUESTION>
hypothesis: <FALSIFIABLE_HYPOTHESIS>
primary_metric: illm.benchmark.request_goodput.slo.v1
guardrails: [illm.correctness.request_result.v1, illm.server.ttft.request_weighted.p99.v1, illm.server.itl.token_weighted.p99.v1, illm.client.e2e.request_weighted.p99.v1, illm.gpu.memory.peak_bytes.v1, illm.request.terminal.failure_rate.accepted_request_weighted.v1]
tuning_suite: <DIGEST_OR_NA>
held_out_suite: <DIGEST_OR_NA>
budget: <TIME_OR_GPU_HOURS>
success: <PRE_REGISTERED_RULE>
no_go: <PRE_REGISTERED_RULE>
```

负结果是有效结果；不得为了发布修改 held-out、SLO 或 baseline。

## 11. Compatibility and Migration

### 11.1 Compatibility matrix

| Artifact | 读取上一版本 | 写入格式 | Breaking 条件 | 迁移/降级 |
|---|:---:|---|---|---|
| config |  |  |  |  |
| public API/CLI |  |  |  |  |
| event/capture |  |  |  |  |
| model/weight manifest |  |  |  |  |
| cache/graph artifact |  |  |  |  |

### 11.2 Rollback

描述如何关闭新 feature flag、回退旧 binary/config，以及哪些缓存/capture 不可复用。至少执行一次 clean-room rollback rehearsal；`rollback/manifest.json` 绑定 source/target binary、config、lock、model/weights、schema/support matrix 与 previous evidence digest，另保存命令、canary、terminal/resource 终态和结论。只写操作说明不能满足发布门。

## 12. Definition of Done

### 12.1 Must

- [ ] <REQUIRED_OUTCOME>

### 12.2 Conditional

- [ ] <ONLY_IF_TRIGGERED>

### 12.3 Evidence bundle

```text
release/<VERSION>/
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

## 13. Known Limitations

- <LIMITATION_AND_IMPACT>

### 13.1 下一版本输入

- 哪些 limitation 成为下一版 Scope；
- 哪些实验结果触发下一版设计；
- 哪些契约在下一版仍必须保持。

---

## Appendix A. Requirement Traceability

| Requirement | Design/contract | Test/evidence | DoD item |
|---|---|---|---|
| <REQ-ID> | §<N> | <TEST_OR_ARTIFACT> | §12 |

## Appendix B. Open Questions

| Question | Owner | Resolve before | Resolution artifact |
|---|---|---|---|
| <QUESTION> | <OWNER> | accepted/implementation/release | ADR/spec amendment |
