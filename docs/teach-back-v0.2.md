# V0.2：从一次 forward 到有状态流式推理

> 当前状态：`0.2.0.dev0` 开发快照，**unqualified**。本文是学习材料，
> 不是 release evidence，也不表示 CPU、GPU、clean-room 或正式 tag 已通过。
> 冻结边界以 [V0.2 SPEC](specs/versions/v0.2-stateful-streaming.md) 和
> [ADR-006](adr/ADR-006-stateful-streaming.md) 及已获批的
> [ADR-007](adr/ADR-007-numerical-execution-policies.md) 为准；当前授权仅规范与CPU，GPU暂缓。

V0.1 回答“给定完整上下文，下一个 token 是什么”。V0.2 把这条数学 oracle
装进一个最小引擎：请求有生命周期，模型有增量状态，结果能逐 token 交付，取消
必须与正在执行的工作仲裁，资源只有在 completion fence 已知完成后才能回收。

## 1. 一次请求经过什么

```text
GenerationRequest
  → validate/tokenize/capacity check
  → accepted + MetadataChunk
  → prefill(prompt)
  → TokenChunk(output[0])
  → decode(last committed output) × N
  → logical terminal decision
  → wait/resolve fence + release leases/KV
  → TerminalChunk + final_text_delta
```

当前引擎一次只接受一个 non-terminal request。第二个请求不会排队，而是在
acceptance 之前以 `OVERLOADED{reason=SINGLE_ACTIVE_LIMIT}` 拒绝。accepted 请求的
正常 stream 恰好包含一个 metadata、零个或多个 token，以及一个 terminal。

| chunk | 何时可见 | 关键事实 |
|---|---|---|
| `MetadataChunk` | validate、tokenize、reserve 成功后 | request/model/sampling/MemoryPlan identity、input 长度、accepted 时间 |
| `TokenChunk` | logits、sampling、KV write 与输出位置全部 commit 后 | 连续 `output_index`、token、增量文本、server-ready 时间、KV/output 两种长度 |
| `TerminalChunk` | terminal winner 已决定，且已证明 fence 完成与安全回收后 | outcome、reason/error、`final_text_delta`、完整 `GenerationResult` |

“算出了 logits”不等于“token 已发布”。如果取消在 prepare 与 commit 之间胜出，
proposal 必须被丢弃，output index 和 RNG accounting 都不能前进。

## 2. KV cache 为什么总比输出慢一步

tiny fixture 对 `hello` 默认加 BOS，所以 prompt 长度是 6。固定生成 4 个输出时，
contiguous KV 路径的实际 query token 数是 `[6, 1, 1, 1]`：第一次 prefill 处理
完整 prompt，之后每次 decode 只处理上一个已提交输出。全量重算则处理
`[6, 7, 8, 9]`。同一显式 candidate policy 的 B/C 完整 token 必须一致；
不能把这个 tiny 例子推广为历史 C0 的真实低精度兼容已通过。

| step | 本步 query | fence 后有效 KV | 本步发布后 committed outputs |
|---:|---:|---:|---:|
| prefill / output 0 | 6 | 6 | 1 |
| decode / output 1 | 1 | 7 | 2 |
| decode / output 2 | 1 | 8 | 3 |
| decode / output 3 | 1 | 9 | 4 |

最后一个已发布 token 没有必要为了结束请求再做一次 forward，因此终态是
`kv_valid_token_count=9`、`committed_output_count=4`，不能把二者压成同一个
`length`。这个“一 token lag”也是以后做 resume、preemption 和 speculative
decoding 时必须显式保存的状态。

KV mode 只描述是否缓存，不能独自确定数值身份：`legacy+none=A` 保留旧reference；
显式 candidate 下 `none=B` 是 full-recompute 对照、`contiguous=C` 是增量路径。
B 与 C 共享算术实现，因此 B 不是独立 oracle，仍需普通 Transformers、独立
FP64 和原 golden。未显式policy的旧 C0/schema2 仅保留作开发诊断。
V0.2 没有 paged KV、block table、prefix cache 或 eviction。

“每步 query 只剩 1 个 token”不等于每步成本是常数：attention 仍然要读取
历史 K/V，序列越长，读取和注意力计算仍会增长。当前 eager 路径还会拼接/展开
K/V，并不是 vLLM 或 SGLang 的优化 kernel。这里先证明减少重复的 projection、
MLP 和上下文 forward；实际快多少要看固定 workload 的实测，不能用 query 数比例
代替吞吐提升。

## 3. 文本流不是逐 token 独立 decode

一个 token 可能只包含 UTF-8 或 byte-BPE 序列的一部分，所以 `TokenChunk.text_delta`
允许暂时为空。每个请求持有一个增量 detokenizer；terminal 时，
`final_text_delta` flush 尚未成形的尾部字节，但不新增 token，也不新增 output index。

必须满足：

```text
concat(all TokenChunk.text_delta, TerminalChunk.final_text_delta)
  == canonical_decode(all committed output token IDs)
```

因此 UI 应分别展示 token commit 与文本增量，不能用“这一帧没有字符”推断“没有
生成 token”。

## 4. 取消有两个时刻

取消首先争夺**逻辑 terminal**，然后等待**物理 terminal-ready**：

1. `request.terminal_decided` 记录唯一 winner，例如 `CANCELLED`。
2. 已 launch 的工作仍持有 input/output/workspace lease 和 KV write lease。
3. completion fence 已知完成后，未提交 KV 才能 abort，lease/KV 才能释放。
4. 资源回到基线后才发布正常 terminal-ready event/chunk。

所以 logical cancel 可以早于客户端看到 terminal。相反，如果 fence 永久 unknown，
引擎必须 quarantine 仍被引用的资源并 fail-stop；capture 保持 `INCOMPLETE`，stream
以结构化错误关闭。此时不能伪造正常 terminal、延迟样本或“资源已归零”。

CLI 第一次收到 `SIGINT` 时只登记意图；独立控制线程在请求已 accepted 后调用
幂等 cancel，主执行流继续等待安全 cleanup。30 秒只会发出“仍在等待”的告警。第二次中断
可以走 emergency abandon 并以 130 退出，但这代表 quarantine/未证明释放，
不是一次成功 cancel。

还有一个边界：服务器已经完成 fence、回收、terminal 和指标记录，但客户端尚未
收到最后一个 chunk 时也可能中断。此时 CLI 仍返回130，表示交付中断；服务器
已经完成的 capture 可以保留 COMPLETE，不能把真实的已完成事实改写为未完成，
也不能虚构已经归零的 GPU 资源仍处于 quarantine。

Python API 的 `cancel()` 也只仲裁逻辑状态，不会偷偷启动后台推理或清理线程。
如果 stream 正暂停在 `yield`，调用方需要继续消费到 terminal，或显式
`stream.close()` 触发安全清理；不要取消后丢弃生成器并期待 GC 替你回收 GPU 资源。

## 5. MemoryPlan 是 admission 事实，不是运行时猜测

引擎启动时根据 model/spec、dtype、device、最大 context、KV layout 与预算生成不可变
`MemoryPlan`。请求必须在 acceptance 前通过 context 和 capacity 检查；预算不足返回
确定性的 `RESOURCE_EXHAUSTED`，而不是先运行再把任意 OOM 解释成 admission。

注意“冻结”不等于“全部实测”：weights bytes 来自已校验的 storage manifest，
KV bytes 来自固定 layout 的张量大小；activation、workspace、IO、telemetry 和
headroom 是本 eager 实现的保守静态估算。它们不是运行时 allocator peak，也不
保证当前空闲显存充足。基线报告要把这份预算与实际 peak allocated/reserved 分列。

tiny profile 的最大 context 保持 256。新的 V0.2 real-model profile 是 512，用来容纳
预声明的 256-token prompt + 64-token output 基线；它不改变旧 V0.1 profile。

## 6. Capture 能回答什么

schema2 capture 使用独立 catalog `illm-stateful-v02-v1`，能力严格为 `[R0]`。
这属于未显式policy的历史 C0 路径；新policy的capture3 writer 尚未实现，不能
把 schema2 capture 冒充新数值策略证据。
默认 metadata-only：保存 phase、index、长度、fence/lease、时间关系和可选本地 HMAC，
不保存 prompt、原始文本、token IDs、logits 或 HMAC key。live CLI 输出 token/result
不代表落盘 capture 也保存这些内容。

R0 Request Story 可以回答“先发生了什么、KV 在哪一步增长、terminal winner 是谁、
何时完成回收”。它不能重建完整控制状态，不能重跑模型，也不能进行 what-if。
对 schema2 请求 `state` replay 必须 fail closed；R1、R2、R3 都不是 V0.2 能力。

## 7. 五个指标如何不说谎

`MetricRegistryV1` 在 V0.2 仍是 experimental。时间戳先在同一 monotonic clock
domain 内以整数 ns 相减，再转换为 SI 秒；server 与 client domain 永远不能交叉相减。

| metric ID | 样本与 weighting |
|---|---|
| `illm.server.ttft.request_weighted.p99.v1` | accepted 到首个 committed token；每个有输出的 request 一个样本 |
| `illm.server.itl.token_weighted.p99.v1` | 相邻 committed token 间隔；每个 interval 等权 |
| `illm.server.tpot.request_weighted.p99.v1` | 首末 token 时间差除以 `output_count-1`；每个 output≥2 的 request 一个样本 |
| `illm.server.e2e.request_weighted.p99.v1` | accepted 到 terminal-ready；每个正常形成 terminal 的 accepted request 一个样本，按 outcome 分层 |
| `illm.client.e2e.request_weighted.p99.v1` | client send 到 client 观测 terminal/结构化 close；必须有完整 client 时间对，按 outcome 分层 |

p99 使用 nearest-rank `ceil(0.99*N)`，不插值。FAILED/CANCELLED 请求在失败前已经
形成的 TTFT、ITL、TPOT 样本仍保留。零输出请求没有 TTFT/ITL/TPOT；单输出请求
没有 ITL/TPOT，必须增加对应 censor counter。缺 client timestamp 是
`client_e2e_missing`，不是 0 秒。pre-accept rejection 另计，不能混入 accepted
latency；`DEADLINE_EXCEEDED` 只是保留 enum，本版计数必须为 0。

## 8. 在当前源码上动手

不要复制或复用另一份 checkout 生成的 source-bound environment。先为**当前源码**
重新 bootstrap；Windows 正式 profile 使用 CPython 3.12.14，Linux 使用 3.12.3。
下面是开发学习命令，不授予 qualification：
它们未选择显式policy，运行保留的 C0/schema2 诊断路径；新候选见 §8.1。

```sh
python scripts/fetch_runtime.py --runtime-profile v01-cpu --artifact-cache .illm/cache/runtime
python scripts/bootstrap.py --profile study-v02-current --runtime-profile v01-cpu --artifact-cache .illm/cache/runtime --locked --offline --write-manifest out/study-v02/toolchain.json --write-manifest-digest out/study-v02/toolchain.sha256
python scripts/in_env.py --manifest out/study-v02/toolchain.json --expected-manifest-digest-file out/study-v02/toolchain.sha256 -- python -m illm verify --fixture tiny-decoder
python scripts/in_env.py --manifest out/study-v02/toolchain.json --expected-manifest-digest-file out/study-v02/toolchain.sha256 -- python -m illm run --model fixture:tiny-decoder --prompt hello --max-new-tokens 4 --seed 7 --stream --capture out/study-v02/capture/demo
python scripts/in_env.py --manifest out/study-v02/toolchain.json --expected-manifest-digest-file out/study-v02/toolchain.sha256 -- python -m illm inspect request out/study-v02/capture/demo --request req-0001
python scripts/in_env.py --manifest out/study-v02/toolchain.json --expected-manifest-digest-file out/study-v02/toolchain.sha256 -- python docs/examples/v0_2_streaming.py
```

[Python API 示例](examples/v0_2_streaming.py) 先用独立 engine/request 调用
`generate()`，以 `kv_mode="none"` 一次返回完整结果；再用另一组 engine/request
调用 `stream()`，以 `contiguous` 逐 chunk 交付。示例比较完整 token 与文本结果，
并核对两条路径的 query 长度分别为 `[6, 7, 8, 9]` 和 `[6, 1, 1, 1]`。
这能直接观察“返回方式”和“是否复用 KV”是两件事：`generate()` 并不意味着
必须关闭 KV，`stream()` 也不意味着一定有 KV。示例两条路径都显式安全关闭。

tiny fixture 是确定性合成模型，只用于学习和 correctness；重复字符或非聊天式
输出不代表语言能力。该示例属于现有静态类型检查范围，但不授予 GPU 或发布资格。

### 8.1 显式数值策略的 CPU 学习切片

新的[policy CPU 示例](examples/v0_2_policy_cpu.py)在同一个已绑定当前源码的
环境中运行：

```sh
python scripts/in_env.py --manifest out/study-v02/toolchain.json --expected-manifest-digest-file out/study-v02/toolchain.sha256 -- python docs/examples/v0_2_policy_cpu.py
```

源码改变后必须重新bootstrap，不能复用上个commit的manifest。示例显式传入
`candidate_policy("fp32")`，分别执行B全量重算与C真KV，打印相同policy digest
以及不同query长度。这里`none`只说明KV模式，不再足以识别数值执行方式。
FP32 contraction、RMSNorm、RoPE和softmax的cast边界见
[冻结协议](implementation/v0.2-numerical-policy-protocol.md)。

A仍是未修改的V0.1 reference，B不是自封oracle，C也不能只与共享实现的B
比较就宣布正确。CPU检查另用普通Transformers与独立FP64公式，并保留原golden。
FP64只验证tiny数学语义，不能代替真实BF16/FP16执行。schema3结果包含policy，
旧schema2结果不追认身份；schema3不是R3 replay。

此切片不支持新policy capture，传入capture会在acceptance前明确拒绝；不写
隐藏新语义的schema2。GPU也明确拒绝。本例可学习数值身份与KV工作量，但不是
交互式可视化、性能领先证明或已发布V0.2。

想核验完整CPU开发检查，可按[checkpoint runbook](implementation/v0.2-policy-cpu-checkpoint.md)
在新环境依次运行`inputs`、`cpu`、`verify`。检查stdout/stderr、JUnit和checksum，
不要仅凭目录或一个`pass`字段判断通过。该开发报告仍不是协议要求的完整raw tensor证据。

## 9. 自检问题

1. 为什么 `output_count=4` 时 contiguous KV 可能只有 `prompt+3` 个 valid tokens？
2. 为什么 token 已 sample 但未 commit 时不能发布？
3. 为什么 logical `CANCELLED` 不能直接证明显存已可复用？
4. 为什么 `final_text_delta` 可以非空，却不能增加 output index？
5. 为什么无 client timestamp 时不能用 server terminal 时间补一个 client E2E？
6. 为什么 Request Story 能画时间线，却不能声称支持 R1 state replay？
7. 为什么 B/C 输出相同，仍需要普通 Transformers、独立 FP64 和旧 golden？
8. 为什么相同 KV 模式不等于相同数值策略，schema3 也不等于 R3？

