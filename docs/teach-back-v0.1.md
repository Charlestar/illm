# V0.1：从 token 到下一个 token

本文对应正在开发的 Reference Runner，不是已发布声明。稳定的上一版本是
`v0.0.0`；V0.1 的发布资格以 [SPEC](specs/versions/v0.1-reference-runner.md)
和最终 evidence 为准。

## 1. 先建立正确预期

tiny fixture 的权重是确定性生成的数值测试材料，并未训练语言能力。因此，
`hello` 生成重复字符并不是判断实现对错的标准。正确性要看：相同固定输入与
独立 Transformers oracle 的中间张量是否在预注册 tolerance 内、greedy token
是否完全一致，以及失败路径是否拒绝错误权重。

本版本故意每生成一个 token 就重算完整上下文。不缓存 K/V、不做 batching、
不输出流式响应。这个慢而直接的路径将作为后续优化版本的正确性参照。

## 2. 一层 decoder 的 shape 地图

tiny 模型有2层，hidden size32，intermediate size64，4个 Q heads、2个 KV
heads、head dimension8，vocabulary259。下面 `T` 是当前上下文长度，batch固定1。

| 阶段 | 输出 shape | 为什么需要它 |
|---|---|---|
| token IDs | `[1,T]` | tokenizer 给出的离散索引 |
| embedding | `[1,T,32]` | 把每个 token 映射成向量 |
| input RMSNorm | `[1,T,32]` | 稳定进入 attention 的向量尺度 |
| Q projection | `[1,4,T,8]` | 每个位置的4组查询 |
| K、V projection | 各 `[1,2,T,8]` | 仅2组 K/V，由 Q heads 分组共享 |
| RoPE(Q、K) | shape不变 | 把绝对位置信息注入查询与键 |
| GQA repeat | K/V各 `[1,4,T,8]` | reference 显式展开每个 KV head 的2个使用者 |
| Q×Kᵀ + causal mask | `[1,4,T,T]` | 每个位置只看自己和之前的位置 |
| softmax | `[1,4,T,T]` | 把每一行分数变为注意力概率 |
| attention×V | `[1,4,T,8]` | 按注意力概率汇总 value |
| merge heads、O projection、residual | `[1,T,32]` | 汇回 hidden 维度，并加回输入 |
| post-attention RMSNorm | `[1,T,32]` | 进入 MLP 前归一化 |
| gate、up、SiLU(gate)×up | 各 `[1,T,64]` | 门控 MLP 的中间特征 |
| down projection、residual | `[1,T,32]` | 回到 hidden 维度，送入下一层 |
| 两层结束后的 final RMSNorm | `[1,T,32]` | 进入词表投影前归一化 |
| LM head | `[1,T,259]` | 每个位置对259个 token 的分数 |

生成只使用最后一个位置的 logits：`logits[0,-1,:]`。这不意味着前面的位置没
被计算；它们参与了 full-context attention，也是在之后引入 KV cache 时要减少
的重复工作。

`hello` 的 tiny byte tokenizer 默认加入 BOS，因此输入长度为6。若生成8个
token且没有提前遇到EOS，8次 forward 的上下文长度依次为6到13。

## 3. 权重文件不等于随意可用的参数字典

`model inspect` 展示归一化 spec、tensor shapes 与加载 manifest。loader 在
执行前检查 safetensors header、重复键、缺失/多余 tensor、shape、dtype、storage
范围和 checksum；未知语义 config 不会被静默忽略。

tiny 的27008个参数对应26份 canonical storage。LM head 和 embedding 是
tied weights：两种参数用途引用同一份 storage，不是复制两份后“暂时相等”。
把某个方阵转置后 shape 仍可能正确，因此 shape 检查不能取代逐 op 数值比较。

## 4. logits 如何变成 token

greedy 对 raw FP32 logits 取最大值，相同时选择最小 token ID，不消耗 RNG。
sampling 则按固定顺序处理：temperature → top-k → softmax → top-p → 保留集合
再归一化 → Philox uniform draw。具体 FP32 舍入、FP64 顺序求和和 CDF 边界见
[ADR-005](adr/ADR-005-reference-model-and-sampling.md)，不能替换成 backend 默认行为。

每个请求有自己的 `SamplingSpec` 和状态。输出位置与随机数消费量是两件事：

- greedy 输出8个 token：`next_output_position=8`，`rng_draws_consumed=0`。
- sample 输出8个 token：已提交地址为 `(0,0,0)` 到 `(7,0,0)`，消费8次 draw。
- 只准备 proposal 而没有 commit：两者都不前进。

固定 seed 的意义是复现同一版本数值规则下的抽样，而不是保证更换模型、dtype
或处理器规则后仍生成相同文本。

## 5. 用证据定位首个错误

执行 README 的 `model validate` 命令，会生成20组固定输入的逐 op 报告、greedy
对照和 `fault-transposed-q.json`。错误实验把第0层 Q projection 的方阵权重转置。
预期首个 divergence 是 `layer.0.q_projection`，而不是仅在最终 logits 报错。

在当前 tiny 的公开 `hello` 实验中，单独转置 Q 权重虽然改变了 logits，
greedy token 仍然相同。报告会如实记录这一点；只比较最终 token 并不足以证明
计算正确。另一个明确标注的复合错误实验，同时转置 Q 并将 final norm 置零，
产生实际 token mismatch，仍能把首个失败定位到更早的 Q projection。

混合精度也不代表把所有张量都转成半精度。Qwen2 的参数可以是 BF16/FP16，
RoPE 频率 buffer 仍应保留原始 FP32。先降精度再转回 FP32 无法恢复损失的信息。
真实 GPU 验证曾发现这一对照模型构造问题，修复时没有放宽任何 tolerance。

阅读报告时依次确认：模型/spec/权重 identity → public input fixture → tolerance
profile → 首个失败 op → shape/dtype → absolute/relative error。只有固定 identity
对应的结论才可比较；另一份权重或 dtype 的通过记录不能搬过来使用。

## 6. 捕获、回放和可视化的边界

`generate --telemetry ops --capture <新目录>` 记录 op 名、shape、dtype和有序控制
事件；不伪造逐 op GPU 耗时。结果中的整次 forward wall time 在 CUDA 路径经过
device boundary synchronization，其范围与单 kernel 的 GPU 执行时间不同。
`logits_host_export_wall_time_ns` 单独测量最后一行 logits 的 FP32 转换、传到
CPU及转成 Python list 的时间；`sampling_wall_time_ns` 才是之后的 canonical
采样阶段。它们不是“总耗时全部归因”的承诺，仍有事件记录、HMAC和落盘等开销。

默认 capture 不保存 prompt、输出 token IDs、logits 或 activation 值；相关摘要
使用本次调用的临时 HMAC key，key不会持久化。正常 CLI 返回的文本/token IDs
是用户请求的生成结果，与默认落盘 capture 的隐私规则分开。

R0 是事件播放。R1_REFERENCE 检查并重建模型/采样 identity、请求状态、输出
数量与已提交 draw 地址；**它不会重跑 Transformer，也不证明数值再执行**。
数值证据来自独立 oracle 与明确公开的 golden fixture。Web UI 尚未实现，这些
结构化事实会服务于后续可视化版本。

正式 bundle 的 `screenshots/model-and-diff/` 提供两张静态 SVG 学习图：
`model-structure.svg` 展开两层 decoder 的 shape 路径，
`first-divergence.svg` 并列展示纯转置与复合故障的实际结果。它们是从已验证的
spec、storage manifest 和 numerical reports 派生的图，不是 UI 截屏，也不是
额外执行模型得到的新结论。`provenance.json` 记录输入和输出哈希；portable
reader 会根据原始事实重新生成并逐字节比较，不能只更换图片后重签 checksum。

## 7. 自检问题

1. 为什么 vocabulary projection 的输出最后一维是259，而不是32？
2. 为什么4个 Q heads 可以使用2个 KV heads？
3. 为什么相同 shape 的权重转置仍会被逐 op 检查发现？
4. 为什么 greedy 输出数量不能直接当作 RNG counter？
5. 为什么 R1 通过不能取代 Transformers 数值对照？
6. 如果导出器阻塞，为什么不能把 required capture 一起丢弃？
