---
title: Hopper FA3 Varlen 优化实现与分析
weight: 45
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Hopper FA3 Varlen 优化实现与分析

本文记录基于 FlagTree Triton TLE 实现
`flash_attn_func_varlen` 的一次完整性能优化过程。目标是理解真实模型 shape
中 decode、prefill 和 mixed batch 为什么需要不同 kernel，并说明每项改动
解决了什么硬件瓶颈、如何避免 decode 回退，以及还有哪些差距。

## 结论先行

最终实现由三层组成：

1. FlagTree 编译器提供 native N80 WGMMA、compact KV pipe、精确 pointer
   shuffle lowering 和更窄的 Membar 分析。
2. FlagGems cost model 将 decode、普通 prefill、D256 paged prefill 和
   ragged mixed batch 分流到不同执行计划。
3. Triton autotune 只在满足静态安全契约的候选中实测，并用完整语义 key
   持久化 winner。

在最终 focused gate 中，四个 prefill/mixed case 相对 vLLM 分别达到
`95.59% / 92.87% / 93.48% / 96.62%`，几何均值 `94.63%`；decode
几何均值性能保留为 `99.9966%`。这说明本轮提出的“prefill 和 mixed 至少
达到 vLLM 90%”已经在该 gate 上达到，但项目要求的逐 case 95% 尚未完全
达到，B2 和 B5 仍是主要缺口。

后续 `ACTIVE_WGMMA_N` 泛化提交又完整跑了四组 Qwen3.6 runtime trace：
25/25 个 prefill case 达到 vLLM 的 90%，四组 prefill/mixed phase 的中位数
也都超过 90%；但 329 个 mixed case 中仍有 10 个 extreme-ragged tail
未达 90%。因此“phase 级目标”已经达到，“所有 mixed case 逐条达到 90%”
尚未达到，不能用中位数掩盖尾部。

必须注意：这组最终 gate 的运行时记录显示设备为
`NVIDIA H100 PCIe, 114 SM`，不是最初四个 CSV 所在的 H800。H800 数据用于
发现问题和选择 shape，提交后的跨机器结论仍需在同一台 H800 上复测，不能把
H100 的百分比直接当成 H800 结果。

## 提交清单与依赖

### FlagTree

| Commit | 内容 | 性质 |
| --- | --- | --- |
| `746704102` | WGMMA `active_n`/`active_k` carrier extent、accumulator tail 恢复及 verifier/codegen | 通用 WGMMA 能力 |
| `c814ab090` | 在 WGMMA 指令使用点构造 shared descriptor | descriptor 正确性修复 |
| `299095b61` | compact KV pipe storage、stage/chunk view 与 Python TLE API | Hopper compact storage |
| `a4bdfb464` | compact KV stage/chunk 的精确 shared-memory alias 建模 | Membar 分析基础 |
| `9f2abcc95` | 将精确 paged-pointer broadcast 降为 warp shuffle | 已测性能优化 |
| `90815bacf` | WGMMA 已完成后的精确 release rendezvous 抑制 | 静态有效，运行时收益待复验 |
| `67e44dfb4` | 将 full-carrier `active_n` 规范化为普通 WGMMA | 正值 config 与原 codegen 的兼容层 |

### FlagGems

| Commit | 内容 | 性质 |
| --- | --- | --- |
| `dec24aa0` | 合并四组 Qwen3.6 真实运行时 shape | benchmark 数据 |
| `fd041383` | autotune key 分区与 optional metadata schema 归一化 | 调优基础设施修复 |
| `f59f5485` | mixed Split-K、D256 paged prefill、compact N80 和 decode guard | 运行时集成优化 |
| `6d4cb101` | 将固定 N80 泛化为按 head dim 搜索的 `ACTIVE_WGMMA_N` | head-dim/autotune 泛化 |

依赖关系如下：

```text
FlagTree 746704102 ─> c814ab090 ─> 299095b61 ─> a4bdfb464 ─┐
                                                           ├─> FlagGems f59f5485 ─> 6d4cb101
FlagTree 9f2abcc95 ────────────────────────────────────────┘

FlagTree a4bdfb464 + completed-WGMMA analysis ─> 90815bacf
FlagTree 746704102 ──> FlagTree 67e44dfb4 ──> positive full-N config

FlagGems fd041383 ──> FlagGems f59f5485
FlagGems dec24aa0 ──> benchmark / pre-tune coverage
```

`f59f5485` 使用了 `tle.gpu.alloc_compact_kv`、WGMMA `active_n=80` 和
`active_k=80`，因此不能脱离 `746704102`、`c814ab090`、
`299095b61` 和 `a4bdfb464` 这组基础能力单独运行。
`6d4cb101` 在此基础上把 QK 的 active extent 变为 config constexpr，
并依赖 `746704102` 提供的通用 `active_n` verifier/lowering。

## 从输入 shape 到 kernel 的调用链

运行时决策可以概括为：

```text
flash_attn_varlen_func
    |
    v
prepare_fa3_inputs / validate_fa3_plan
    |
    v
FA3Scheduler.build
    |
    +-- FA3RouteCostModel.analyze
    +-- route
    +-- adaptive work / Split-K plan
    |
    v
launch_fa3
    |
    +-- decode / 近似 decode ----------> direct kernel
    +-- 长且规则的 paged prefill ------> packed persistent kernel
    |                                      +-- compact N80 candidate
    |                                      +-- legacy N64/N128 candidate
    +-- ragged mixed / 长 K ------------> persistent Split-K + combine
```

核心文件：

- `attention_impl/scheduling.py`：cost model、dispatch、autotune pruning；
- `attention_impl/launcher.py`：把执行计划转换为 kernel 参数；
- `attention_impl/persistent.py`：persistent producer/consumer 主体；
- `attention_impl/common.py`：paged KV copy、ragged mapper、Split-K count；
- `attention_impl/split_combine.py`：Split-K partial result 合并；
- `hopper/tune_configs.yaml`：实际参与搜索的候选集合。

建议从 `attention.py::flash_attn_varlen_func` 开始阅读，再依次跟到
`FA3RouteCostModel.analyze`、`FA3Scheduler.route`、
`FA3Scheduler.build`、`launcher.launch_fa3` 和
`persistent.flash_varlen_fwd_v3_tle_kernel`。这样可以先理解“为什么选这个
kernel”，再进入 kernel 内部看“这个 kernel 怎样执行”。

## 为什么 decode、prefill 和 mixed 不能共用一个最优配置

### 原始 H800 CSV 告诉了我们什么

四份 CSV 自身没有 GPU 型号字段；以下“H800”归属来自测试任务的服务器记录。
它们有 1202 行，但只有 562 个唯一 shape。重复行中 one-token decode 很多，
所以“全部行”的中位比例 `2.526` 和几何均值 `1.615` 会高估整体效果。

去重后，中位比例只有 `0.700`、几何均值 `1.001`；进一步只看
FlagGems latency 不小于 0.1 ms 的 345 个主要 case，中位比例为 `0.603`，
没有一个达到 0.95。按 dispatch 路径聚合更能解释差异：

| 原始 H800 路径 | 唯一 case | 中位 vLLM/FA3 | 延迟和比例 |
| --- | ---: | ---: | ---: |
| direct packed-GQA decode | 177 | 2.936 | 2.931 |
| long paged prefill，token-query | 17 | 1.904 | 1.919 |
| persistent Split-K s32 | 19 | 0.852 | 0.835 |
| direct ragged | 319 | 0.600 | 0.669 |
| plain direct | 15 | 0.642 | 0.661 |
| true long prefill | 15 | 0.653 | 0.647 |

这里的比例也是 `vLLM latency / FlagGems latency`，大于 1 表示 FlagGems
更快。175/181 个至少 2x 的 case 来自 packed-GQA token decode，而
`direct_ragged` 占唯一 case 正向延迟缺口的 89.7%。因此最初看到的
“一部分 2x、一部分 0.4～1x”并不矛盾：它们实际在走不同 kernel，也处在
完全不同的算术强度和固定开销区间。

### Decode

Decode 通常满足 `Q=1`，单次工作量小，launch、scheduler ticket、TMA setup
和 partial-buffer combine 都容易成为固定开销。即使 persistent kernel 的
数据复用更好，也可能因为多出的控制路径而变慢。

因此 cost model 保留 direct decode：

- 规则 decode 直接使用 compact direct kernel；
- “接近填满一个 SM wave”的 uniform decode 也优先 direct，避免 persistent
  scheduler 在已经有足够并行度时增加开销；
- decode 不进入 compact N80 prefill 配置，也不启用 `STAGGER_KV`、
  `EARLY_CAST_P` 或 `RESCALE_O_BEFORE_PV`。

这不是只靠一个 `Q == 1` 判断。batch、head 数、SM wave 填充度、K 长度和
Split-K 数量共同决定 direct 是否更便宜。

### Prefill

Prefill 的 Q 和 K 都长，矩阵乘占主导。D256、paged KV、GQA 场景的关键是：

- 多个 Q head 共享同一个 KV head，应使用 `PACK_GQA` 增强 KV 复用；
- page size 16/32 与物理 tile N64/N128 不匹配；
- producer 的 page lookup、pointer broadcast、shared-memory publish 会在
  每个 KV tile 重复；
- persistent kernel 可以把 TMA、WGMMA 和 softmax 放入稳定流水线。

这类 shape 被路由到 `PAGED_D256_PREFILL` profile，再由 autotune 在 legacy
和 compact 候选中选择。

### Mixed batch

Mixed batch 同时包含很短和很长的 query。若按
`batch * ceil(max_q / BLOCK_M)` 建矩形 grid，大量 program 会落在短序列的
无效区域。另一方面，最长 K 又可能需要 Split-K 才能填满 GPU。

优化后的执行计划：

1. 用 ragged prefix work 映射实际有效的 query blocks；
2. 根据 K 长度、batch/head 并行度和 SM wave 计算 adaptive split count；
3. 用 `EXPLICIT_SPLIT_K_CHUNK` 区分普通长 K 与 Q11/Q12 等窄 query 特例；
4. combine kernel 根据 `total_q` 而非纯 `max_q * batch` 选择 compact grid；
5. tiny GQA8 mixed shape 可将 combine `BLOCK_M` 降到 1，避免合并端再次浪费。

针对性 A/B 中，长 mixed case 最高达到约 `8.25x`；另外几组原慢 case
分别约有 `1.19x`、`1.30x` 和 `1.45x` 改善。这里的收益主要来自少做无效
work，而不是单条 WGMMA 变快。

极端 mixed 的单变量结果从 direct-ragged `1.4661 ms` 降到 persistent
split16 `0.1776 ms`；同时单请求长 decode guard 仅从 `0.06837 ms` 变化到
`0.06842 ms`，属于噪声级。combine 自身通过 BM64→BM8 将 NCU duration
从约 31.1 μs 降到 10.3 μs，并把寄存器从 255 降到 71。

## 优化一：真实 Qwen3.6 shape 覆盖

四个源文件为：

- `Qwen3.6-35B-A3B-p1024d1024.yaml`
- `Qwen3.6-35B-A3B-p4096d1024.yaml`
- `Qwen3.6-35B-A3B-p32768d1024.yaml`
- `Qwen3.6-35B-A3B-p65536d6144.yaml`

它们共有 1202 条原始记录，按完整 15-field shape 去重后得到 562 条 trace。
与原有 24 条代表性和边界 case 合并后，
`benchmark/models_shapes/qwen36.yaml` 包含 586 条唯一 case。

去重不能只看 `max_seqlen_q/max_seqlen_k`。以下字段会改变 dispatch 或代码：

- query/KV length RLE；
- query/KV head 数及 GQA ratio；
- head dim；
- causal/local/alibi/softcap；
- paged、page size、block table；
- KV layout；
- Split-K；
- 实际 Q/K 上界。

## 优化二：autotune 搜索面和缓存契约

### 搜索什么

persistent FA3 的性能敏感维度包括：

- `BLOCK_M`、`BLOCK_N`；
- MMA group 和 warp 数；
- Q/KV buffer 数；
- TMA Q/O 与 TMA KV；
- `STAGGER_KV`；
- `ACTIVE_WGMMA_N`；
- `COMPACT_KV80`；
- `EARLY_CAST_P`；
- `RESCALE_O_BEFORE_PV`。

这些维度并非对所有 shape 做无约束笛卡尔积。pruner 先用静态安全契约裁剪：

- compact N80 只允许 FP16、D256、paged、causal、GQA；
- 仅允许 `(page16, GQA4)` 或 `(page32, GQA8)`；
- 必须是 packed prefill，而不能是 decode、Split-K、local、alibi、
  softcap 或 S auxiliary；
- compact path 固定双 MMA group、双 KV buffer、TMA Q/O、staggered KV
  和 early FP16 P；
- 实测保留的 compact 搜索维度只有
  `RESCALE_O_BEFORE_PV={false,true}`。

因此“笛卡尔积”用于发现 winner，“cost model + pruner”用于避免让不合法或
明显不合适的组合进入编译和计时。

### 为什么 autotune key 必须完整

只有 `seqlen_q`、`seqlen_k`、`total_q` 使用 align32 bucket；其他会改变
语义或 pruner 的字段保持 exact，包括：

- batch、Q head、KV head、head dim、page size；
- causal/local/window/alibi/softcap；
- `seqused_k`、paged gather mode；
- pack/ragged/dynamic/split；
- explicit split chunk；
- paged-prefill/dense-TMA profile；
- policy version。

例如 Q255 和 Q256 可能进入同一个长度 bucket，但 dense-TMA profile 不同，
必须由 exact profile key 隔离 winner。

### SQLite schema bug

LibTuner 的一个 winner 表只有一套 value columns。legacy config 不包含
`COMPACT_KV80`、`EARLY_CAST_P` 等 optional metadata；若它先创建表，后来的
compact winner 就可能无法写入。

`_normalize_persistent_config_schema()` 会在搜索集合中发现所有实际出现的
optional fields，并给每个 config 补齐默认值。这样无论 legacy 还是 compact
谁先完成调优，持久化 schema 都一致。

调优时还应使用隔离的 `FLAGGEMS_DB_URL`。全局缓存可能包含旧 policy、不同
温度或被中断搜索得到的 winner，不适合用来做正式 A/B。

## 优化三：用 `ACTIVE_WGMMA_N` 泛化 head-dim 相关的 N tile

### 它是 config constexpr，不是 autotune key

`ACTIVE_WGMMA_N` 表示当前 config 的 QK WGMMA 实际 N extent。它属于
`triton.Config.kwargs`，最终作为 `tl.constexpr` 传入 persistent kernel：

```python
qk = tle.gpu.wgmma(
    q_tile,
    k_tile,
    out_dtype=tl.float32,
    trans_b=True,
    active_n=ACTIVE_WGMMA_N,
)
```

它没有加入 `_PERSISTENT_AUTOTUNE_KEYS`。autotune key 描述一次 workload
调用的输入事实，config metadata 描述本次搜索要比较的实现方案；若把
`ACTIVE_WGMMA_N` 也放进 key，相当于把 N64、N80、N128 拆成互不竞争的缓存
分区，autotuner 就无法为同一 shape 选择 winner。head dim `d` 已经是 exact
key，因此不同 head dim 不会错误复用同一 winner；获胜的
`ACTIVE_WGMMA_N` 则和 `BLOCK_N`、buffer 数等一起写入 value columns。

这次候选集合变化先把 `AUTOTUNE_POLICY_VERSION` 从 10 提升到 11；移除
`ACTIVE_WGMMA_N=0` 哨兵后又提升到 12，避免旧 winner 跨 config 语义复用。
对于旧 YAML/SQLite row，`_normalize_persistent_config_schema()` 根据
`BLOCK_N` 和 `COMPACT_KV80` 补齐正值：ordinary 使用 `BLOCK_N`，compact
使用 80。

### head dim 启发式与候选生成契约

先把运行时 head dim 补齐为 `Dpad`，再构造 1P2C 搜索面：

| `Dpad` 和 eligibility | 进入搜索的逻辑 N | storage |
| --- | --- | --- |
| `Dpad <= 128` | N128、N64 | ordinary N128、ordinary N64 |
| `Dpad == 256` 且 compact profile 合法 | N80、N64 | compact N80、ordinary N64 |
| 其他情况 | N64 | ordinary N64 |

这个选择来自固定 FP16 1P2C payload
`Dpad * (256 + 8 * N)` bytes：D64/D128 可以容纳 N128；D256 的 ordinary
N128 超出当前安全预算，而 compact N80 是已经实测的最大窄 tile，N64 保留
为 no-regression 候选。表中的“compact profile 合法”还包含既有的 FP16、
D256、paged、causal、PackGQA、page/GQA 等契约，不是只看 `Dpad`。
这里 D64/D128 的 N128 与 N64 是两个不同的 ordinary 物理 tile，分别满足
`ACTIVE_WGMMA_N == BLOCK_N`；它们不是“在同一个 N128 backing 中用
active-N64”。当前唯一允许 active extent 小于物理 carrier 的 production
配置仍是 compact D256/N80。

`ACTIVE_WGMMA_N` 不再使用 0 表示另一个 API overload。每个 config 都携带
正的逻辑 N，且满足 `16 <= ACTIVE_WGMMA_N <= BLOCK_N`。当前生成器只产生
N64、N80 和 N128，这三个值在构造候选时已经满足 N16 流水粒度，不在
kernel、pruner 和 config consumer 中重复检查 `% 16`。

这里的“搜索参数”不是把 16 到 `BLOCK_N` 的所有倍数独立地与 storage 做
笛卡尔积。当前合法 config 对由 storage 决定：

- ordinary N64/N128：`ACTIVE_WGMMA_N=BLOCK_N`；
- compact N80：物理 accumulator carrier 为 N128，
  `ACTIVE_WGMMA_N=80`，K/V backing 为精确 N80。

因此 head-dim heuristic 先生成完整、合法的 config 对，再由 autotuner 实测
选择，而不是先生成任意 active N、再在 kernel 内拒绝不一致状态。partial-N80
仍只在 BM128、1P2C、Q1/KV2 的 compact profile 中出现；decode、Split-K、
1P1C、BM64 或 Q 双缓冲使用 ordinary config，其正值
`ACTIVE_WGMMA_N` 等于各自的 `BLOCK_N`。逻辑 tile 直接按

```text
LOGICAL_BLOCK_N = ACTIVE_WGMMA_N
```

计算。因此 producer tile count、offset 和 causal boundary 不再用
`COMPACT_KV80` 特判 N80。这里必须把“计算 extent”和“存储 extent”分开：
ordinary config 中 `ACTIVE_WGMMA_N == BLOCK_N`，SMEM cost model 仍按物理
`BLOCK_N` 计费；只有 `COMPACT_KV80` 的专用 allocator 才按物理 N80
backing 计费。compact config 中 native N80 由 N128 accumulator carrier
的前 80 列承载，`[80,128)` 在 softmax 前恢复为 mask tail，而 K/V backing
本身只存精确的 80 行。将来即使开放 ordinary N96/N112，也不能直接用
`ACTIVE_WGMMA_N` 缩小 SMEM 估算，除非同时实现对应的窄 storage。

### 统一调用与 full-N canonicalization

四个 QK hot site 都直接写：

```python
qk = tle.gpu.wgmma(
    ...,
    trans_b=True,
    active_n=ACTIVE_WGMMA_N,
)
```

FlagTree 的 TLE lowering 将
`active_n == physical N carrier` 规范化为普通
`ttng.warp_group_dot`；只有 `active_n < physical N carrier` 才生成
`ttng.tle_warp_group_dot`。这把“config 中总是正值”和“full-N 必须保持旧
codegen”分离开：FlagGems 不需要数字哨兵或四处重复 constexpr 分支，
ordinary/decode 仍得到原来的同步分析和 PTX，compact N80 才进入 active-N
lowering。编译器回归测试还覆盖 M128 full-N，避免统一接口被 active-N
专用的 M64 约束误伤。

### 为什么暂不开放 ordinary N96

FlagTree 编译器的普通 `active_n` API 接受 Hopper 支持的 8 对齐 N；
FlagGems attention 层进一步要求 16 对齐，因为 paged producer、compact
chunk 和 PV reduction 都以 N16/K16 为流水粒度。即使 compiler 已能生成
`m64n96k16`，当前 kernel 也不能只改一个 constexpr 就安全启用 N96：

- ordinary K/V storage、copy tensor 和 softmax carrier 仍以受支持的物理
  N64/N128 shape 构造；
- compact stage/chunk view 只定义了精确 N80 backing；
- PV 若只归约逻辑 N96，还需要与该 storage 对应的 `active_k=96` contract，
  当前 production compact active-K 路径只验证并开放了 N80。

所以当前安全集合是 ordinary N64/N128 和 compact N80。推广到 N96/N112
需要同时补 storage、copy/view、QK tail、softmax 和 PV active-K 的端到端
契约，不能只依赖硬件 opcode 已存在。

### 本轮正确性、ABBA 与 NCU 验证

首先用 H100 PCIe 对 ordinary D64/D128 做了同进程 full-attention gate。
每个 case 都固定 BM128、1P2C，比较无 `active_n` overload 与
`ACTIVE_WGMMA_N=BLOCK_N` 的 API 语义：

| `Dpad` | `BLOCK_N` / active N | active / legacy 延迟 | active 与 legacy 最大绝对误差 |
| ---: | ---: | ---: | ---: |
| 64 | 64 | 1.01965 | 0 |
| 64 | 128 | 0.98941 | 0 |
| 128 | 64 | 1.02349 | 0 |
| 128 | 128 | 0.99490 | 0 |

4/4 case 输出逐位一致，显式 full-carrier active-N 的最大测得开销低于
2.35%，相对 PyTorch FP32 参考的最大绝对误差不超过
`0.00146484375`。compact N80
另在 small16、small32、B5、B25 和 decode 上通过 5/5 gate：三次重复均
bitwise stable，compact 与 legacy N64 的最大绝对误差为
`0.00048828125`，small case 相对 PyTorch FP32 的最大绝对误差为
`0.00096774101`，decode 为 `0.00004330277`。去掉哨兵后的 decode pruner
仍不保留 compact config，ordinary config 分别携带 N64 或 N128。

最终 compiler canonicalization 的检查比延迟 gate 更严格：B1/B5 ordinary
kernel 的 TTGIR 都只有 11 个 `ttng.warp_group_dot`、没有
`ttng.tle_warp_group_dot`；去掉 debug line 信息后，full-N 正值版与原
overload 版的 PTX SHA-256 逐一相同。compact kernel 则仍包含 TLE active
Op 和 `m64n80k16`，说明 canonicalization 没有吞掉真正的 partial-N。

修改前后还使用相同 H100、相同 shape 和三轮 ABBA 快速 gate 做了 focused
对照：

| Case | before 最优 compact | after 最优 compact | after / before | legacy / 最优 compact：before → after |
| --- | ---: | ---: | ---: | ---: |
| B1 prefill | 4.52024 ms | 4.36138 ms | 0.96486 | 1.08903x → 1.10732x |
| B5 mixed | 0.39897 ms | 0.38779 ms | 0.97198 | 1.09130x → 1.10404x |

B1 和 B5 的 winner 前后都是 `compact_bn80_s1_e1`；其他 shape 上 rescale
仍应交给 per-shape autotune。
表中的最后一列读取逐轮 ABBA 的 paired median，而不是用跨候选的全局
median 相除。
三个 decode case 的 after/before 分别为 1.00410、0.98825 和 1.00039，
几何均值约为 0.9975。这两份记录的
`status=passed-provider`，但 `formal_valid=false`：它们是用于发现明显回退
的探索性 H100 结果，不是独占、锁频条件下的正式 H800 性能证明。
最终 `after_final.json` 记录的 FlagGems commit 为 `6d4cb101`，Hopper
目录 diff 的 SHA-256 是标准空内容摘要，排除了开发中间态污染结果。

NCU 对同一 B1 launch 固定 ordinary N64 和 compact N80 后得到：

| NCU metric | ordinary N64 | compact N80 |
| --- | ---: | ---: |
| Duration | 445.92 us | 439.97 us |
| Dynamic SMEM / block | 197.24 Kbyte | 229.72 Kbyte |
| Registers / thread | 168 | 168 |
| Theoretical / achieved occupancy | 18.75% / 18.75% | 18.75% / 18.75% |
| Local spilling requests | 12,136 | 0 |
| Shared spilling requests | 2,984 | 0 |
| QK HGMMA | 2,146,560 × `m64n64k16` | 1,723,904 × `m64n80k16` |
| PV HGMMA | 536,640 × `m64n256k16` | 538,720 × `m64n256k16` |

N64 的计数对应 134,160 个逻辑 tile，每 tile 为 16 条 QK 和 4 条 PV；
N80 对应 107,744 个逻辑 tile，每 tile 为 16 条 QK 和 5 条 PV。N80 以更宽
的 KV tile 减少约 19.7% 的逻辑 tile 数，虽然每 tile 的 PV K16 repetition
从 4 增至 5，但总 PV 数量基本不变；其 duration 低约 1.33%，且未改变
register 或 occupancy。NCU 在这里证明的是 active-N codegen、资源占用和
性能方向，数值正确性仍由上述 attention/reference gate 证明。

### 四组 Qwen3.6 runtime trace

提交后的同轮运行完成 1202/1202 个 FlagGems FA3 case 和对应的 1202 个
vLLM FA3 case。speedup 定义为 `vLLM latency / FlagGems latency`：

| Shape | Phase | case 数 | speedup 中位数 | ≥0.90 case | 最小 speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| p1024d1024 | Prefill | 4 | 1.06466x | 4/4 | 1.01520x |
| p1024d1024 | Mixed | 70 | 1.32715x | 70/70 | 0.94133x |
| p4096d1024 | Prefill | 3 | 0.99300x | 3/3 | 0.94216x |
| p4096d1024 | Mixed | 153 | 2.69356x | 150/153 | 0.83982x |
| p32768d1024 | Prefill | 8 | 2.57875x | 8/8 | 1.37125x |
| p32768d1024 | Mixed | 68 | 2.61996x | 68/68 | 1.15883x |
| p65536d6144 | Prefill | 10 | 1.53368x | 10/10 | 1.07463x |
| p65536d6144 | Mixed | 38 | 1.41965x | 31/38 | 0.56559x |

全部 25 个 prefill case 逐条达到 90%。10 个未达标的 mixed case 有很强的
共同结构：

- p4096 的 3 个都是 TP1，batch=9/17/37，总 Q=16384，但分别夹有
  4/12/32 个 Q=1 请求；
- p65536 的 7 个主要是 batch=3/4，一个 Q≈16K 请求夹着 Q=1，KV 最长约
  65K；
- 最差 `tp1-trace-ac24f70f8c902b7f` 只有总 Q=67，却要遍历
  K=65546..65560，speedup 为 0.56559x。

这些 tail 的矩阵乘工作不足以摊薄 page walk、persistent scheduler 和
Split-K/combine 固定开销。它们更适合由 cost model 识别
`small total_q + very long/ragged K` 后切到专用 direct/split 方案，而不是
继续扩大 WGMMA N；active-N 解决的是 1P2C prefill 内部 tile 效率，并不会
自动消除调度级固定开销。

同轮四组 decode phase 的 speedup 中位数为 3.06015x、3.79033x、
3.94394x 和 3.39684x。这里说明当前生产 decode 相对 vLLM 有余量；本次
改动的 no-regression 结论仍来自前述同实现 before/after gate，因为两类
对照回答的问题不同。

## 优化四：native compact N80

### 为什么是 N80

目标 D256 paged profile 的最优逻辑 KV tile 为 N80。N64 增加循环次数，
N128 又会为 48 个无效列承担 shared memory、softmax 和 WGMMA carrier
开销。

compact backing allocation 使用物理 `[40, 16, 64]`：

- 2 个 KV stages；
- 每个 stage 20 个 `N16 x D64` chunk；
- N 方向 5 份，D 方向 4 份；
- producer 每解析一个 N16 page row，即可填充四个 D64 panel。

TLE API 通过 stage view 和 chunk view 表达这一布局，Membar alias analysis
仍能证明不同 chunk 的精确区间。

### active-N 和 active-K

QK 使用 `active_n=80`，生成原生 `m64n80k16`；PV 使用
`active_k=80`，只发出 5 个 K16 repetition，而不是物理 N128 所需的 8 个。
最终 production trace 中每个逻辑 tile 是：

- 16 条 `HGMMA.64x80x16` 完成 QK；
- 5 条 `HGMMA.64x256x16` 完成 PV。

raw WGMMA descriptor localization 将 shared byte address 和 descriptor
immediate 在消费 WGMMA 的 asm block 内组合，修复了旧 BN128/KV1 路径可能
使用错误 descriptor 的正确性问题。BN128/KV1 配置本身只有 BN64/KV2 的
约 `0.537x`，因此没有作为性能 winner 保留；compiler fix 保留是为了保证
descriptor 语义正确。

### stagger、early cast 和 rescale

`STAGGER_KV` 让 producer 先发布下一块 K，再补上一拍的 V，使 K/V copy 与
consumer 更好重叠。

softmax 概率 P 在 PV 前尽早转换为 FP16，可显著降低 accumulator live
range。对应实验中 stack frame 从 104 B、spill store 424 B 降到 0，B1 的
动态 spill 从约 40.29M 降到 0。

`RESCALE_O_BEFORE_PV` 不是全局固定值：它在 B2 约有正收益，在 B5 约有负收益，
所以保留为两个 compact candidate 的唯一实测笛卡尔维度，由 per-shape
autotune 决定。

H100 同进程正式 ABBA 中，最终 compact candidate 相对 legacy BN64 的
paired median speedup 为：

| Case | compact / legacy |
| --- | ---: |
| B1 | 1.0479x |
| B2 | 1.1018x |
| B5 | 1.0302x |
| B25 | 1.0582x |

staggered K/V 在 B5/B25 的配对方向也稳定为正；考虑未锁频条件，保守只把
收益表述为约 1.5%～3.5%，而不采用 aggregate 5%～6% 作为承诺。

## 优化五：paged pointer warp shuffle

paged producer 会把一个 64-bit cache pointer 从持有 page id 的 lane 广播到
同一 warp 的其他 lane。原 generic `convert_layout` lowering 使用：

```text
store pointer to shared
bar.sync
load pointer from shared
bar.sync / reuse
```

新的 lowering 只在精确满足以下条件时触发：

- 64-bit pointer；
- rank-2 `[16, 1]` source；
- 目标物理/逻辑布局为已知的 `16 x 8` same-warp broadcast；
- single CTA；
- source 和 destination lane 都在同一 warp。

命中后用两个 32-bit `shfl.sync.idx` 传递低/高半部。类型、shape、layout 或
CTA topology 任何一个不匹配都会回退 generic lowering。

两次无 foreign PID 的 cross-build focused 观察中：

| Case | 旧 lowering | pointer shuffle | latency 改善 |
| --- | ---: | ---: | ---: |
| B1 | 4.7047 ms | 4.5207 ms | 3.91% |
| B5 | 0.4450 ms | 0.4265 ms | 4.15% |

这不是同一进程内的单变量 ABBA。更强的机制性因果证据来自 NCU 的同协议
kernel replay：duration 分别改善 1.06% 和 1.90%。不要把两种协议的绝对
时间混在一起。

动态机制非常清楚：

| 指标 | B1 旧 → 新 | B5 旧 → 新 |
| --- | ---: | ---: |
| `BAR.SYNC.DEFER_BLOCKING` | 17.691M → 5.714M | 1.564M → 0.533M |
| `STS.64` | 6.305M → 0.001M | 0.544M → 0.001M |
| barrier stall / issue | 0.502 → 0.179 | 0.664 → 0.337 |
| eligible warps / cycle | 0.733 → 0.808 | 0.688 → 0.755 |

HGMMA 数量不变，说明收益来自删除 pointer broadcast 的 shared round trip
和 rendezvous，而不是减少 attention 数学工作。

## 优化六：WGMMA release Membar

Membar 原本把 elected consumer 的 release 当成保守的全 shared-memory
依赖，可能在 `wgmma.wait_group` 和 `mbarrier.arrive` 之间插入 CTA
rendezvous。

新分析仅在以下条件全部满足时抑制它：

- single CTA；
- 完整 128-thread consumer warpgroup；
- arrive 为受支持的 elected release；
- shared operand 与 released field 能做精确 alias；
- 目标 WGMMA result 被直接的 wait SSA 命名；
- explicit commit 计数证明该 group 已完成；
- K/V 两次 release 的 barrier allocation 和 released fields 都彼此不相交。

同步边界必须区分：

| 指令/边 | 作用 | 本优化是否删除 |
| --- | --- | --- |
| `wgmma.wait_group` | 等待指定 WGMMA group 完成 | 否 |
| `bar.sync ...,128` | consumer partition 的保守 rendezvous | 仅精确命中时删除 |
| `mbarrier.arrive` | 向 producer 发布 stage 可复用 | 否 |
| `mbarrier.try_wait` | producer/consumer acquire | 否 |
| `fence.proxy.async` | generic/shared 与 async proxy 排序 | 否 |

不能用“已经 wait 了”笼统删除后续 barrier。WGMMA completion、线程会合、
proxy ordering 和 storage-lifetime publication 是四条独立正确性边。

该提交的 focused lit 为 2/2，连同 compact regression 为 4/4。最终静态
结构中：

- PTX `bar.sync` 从 45 降到 41，其中 128-thread barrier 从 36 降到 32；
- SASS `BAR.SYNC` 从 56 降到 52；
- page16/page32 指令行分别减少 19/20；
- registers、shared、stack/local、WGMMA、cp.async、mbarrier 和 proxy
  fence 均保持不变。

small16/small32、B1/B2/B5/B25、decode 和 14 个边界 case 均通过；
compute-sanitizer memcheck 为 0 error，racecheck 为 0 hazard。当前缺少的是
可信的单变量 ABBA 性能闭环，因此它属于“正确性与静态结构已证明有效”，
不能宣称已有确定百分比的运行时加速。

## 最终性能与 decode 回归门

性能百分比定义为：

```text
vLLM latency / TLE latency * 100%
```

### Prefill / mixed

| Case | 类型 | vLLM | TLE | 相对性能 |
| --- | --- | ---: | ---: | ---: |
| B1 | prefill | 4.3729 ms | 4.5749 ms | 95.5852% |
| B2 | prefill | 1.4855 ms | 1.5995 ms | 92.8688% |
| B5 | mixed | 0.4157 ms | 0.4447 ms | 93.4824% |
| B25 | mixed | 1.6952 ms | 1.7546 ms | 96.6171% |
| 几何均值 | - | - | - | 94.6261% |

### Decode 对历史 FlagGems 基线

| Case | 历史 | 当前 | 性能保留 | latency 变化 |
| --- | ---: | ---: | ---: | ---: |
| batch512, K32 | 0.02985 ms | 0.02896 ms | 103.0610% | -2.9701% |
| long, page16 | 0.06581 ms | 0.06644 ms | 99.0525% | +0.9565% |
| long, page32 | 0.09273 ms | 0.09467 ms | 97.9480% | +2.0950% |
| 几何均值 | - | - | 99.9966% | +0.0034% |

严格按“任何 decode case 都不能变慢”，两个 long-decode case 仍有
0.96% 和 2.10% 的小幅回退；按预设 5% 波动门和几何均值，decode 基本持平。
因此不能把当前状态表述成“严格零回退”。

## 为什么 B2/B5 还没有达到 95%

B1/B5 NCU 中，TLE 与 vLLM 的 HGMMA 动态数量完全相同，occupancy 也都约
18.75%。L2/DRAM 流量接近，因此剩余差距不是 Tensor Core 工作量、occupancy
或显存带宽。

相对 vLLM，TLE 每个 QK tile 稳定多出约：

- 30 条 `FADD`；
- 6 条 `FMNMX`；
- 4 条 `MUFU.EX2`；
- 2 条 `FFMA`；
- 4 条 `SHFL.BFLY`；
- 约 23 条 `UIADD3.X`；
- 约 37 条 `UMOV`。

对应地，TLE 的 executed instructions 在 B1/B5 分别多 42.8%/36.3%，
uniform pipe active 约为 vLLM 的 2.62x/2.65x。这说明优先级应放在
softmax carrier、descriptor/address formation 和 page-table state，而不是
继续增加 WGMMA 并行度。

B2/B5 的缺口也与 shape 特征吻合：

- B2 的 batch 内 Q/K 极不均匀，ragged dispatch 已减少无效 tile，但短序列
  和边界 tile 的固定开销占比更高；
- B5 是 mixed batch，矩阵乘总量较小，page lookup、uniform address 和
  scheduler 控制占比比 B1 更高；
- vLLM 的 CUTE fragment 直接以逻辑 N80 做 softmax，而当前 TLE frontend
  仍返回物理 N128 accumulator carrier，再 mask 80..127。

最后一点是基于源码与指令计数的归因，不是已经完成的优化。

## 已排除的实验

以下内容没有进入提交：

- `DEFER_ROW_SUM`：ABBA 约为 0.997x～1.003x，属于噪声；
- active-N80 wait compaction：默认关闭，B1/B5 无实质收益；
- `REUSE_KV_PAGE_STATE`：TritonGPU conversion 出现 unresolved
  materialization，最终重构未完成编译；
- direct `BLOCK_M` 实验旋钮：只服务于已拒绝的扫描，没有 production
  consumer；
- N64+N16 模拟 N80 和 direct pointer layout：没有形成稳定 winner；
- 高温 full-24 run：发生 thermal throttling，不能作为性能证据。

失败实验的价值在于缩小后续搜索空间，但不能以 default-off 代码或环境变量
的形式留在 production 路径。

## 当前最值得继续做的三项优化

### 1. Active-N80 softmax lowering

让逻辑 active extent 穿过 max、exp 和 row sum，而不是在物理 N128 carrier
上处理 tail 48。静态门应保证：

- QK/PV HGMMA 仍为 16/5；
- 至少消掉约 4 `MUFU.EX2` 和一部分约 30 `FADD`/tile；
- 不增加 LDS/STS、spill，register 不超过当前 168。

只延迟 cross-lane row sum 已被证明无收益；必须同时减少 tail elementwise
和 local reduction。

### 2. Page16 K/V page-state 复用

当前 compact producer 每个 N80 tile 为 K 读取 5 个 page entry，staggered
V 又读取 5 个；vLLM 热循环只读取 1 次并保留 page/page-offset 状态。

合理迭代顺序：

1. 先把状态跨 stagger loop carry，使 10 次降到 5 次；
2. 再用 5-lane 或 8-lane carrier 并行读取并 shuffle，使 5 次降到 1 次。

前一版失败实现不能复用，应重新从 loop-carried state 类型和 helper lowering
契约设计。

### 3. Compact WGMMA descriptor 分组预编码

当前每条 WGMMA 都可能重复 raw address 的 `shr/and/add`。可为四个 D64
panel 共用一个 encoded stage base，再用整数 offset 派生 descriptor。

验收门：

- HGMMA 16/5 不变；
- `UIADD3.X` 降到每 tile 接近 2；
- register 不超过 168；
- local memory 与 spill 保持 0。

## 验证与复现

本次提交前的 clean-tree 验证：

- FlagTree `triton-opt` rebuild：通过；
- descriptor、active extent、compact KV、pointer shuffle、Membar：
  18/18 lit 通过；
- compact TLE frontend：9/9 和 6/6 pytest 通过；
- FlagGems 变更模块 `py_compile`：通过；
- 15 个新增调度/调优测试展开为 37/37 pytest 通过；
- rejected marker scan 和 `git diff --check`：通过。

详细命令记录位于：

```text
/home/lutao/zhiyuan/small_exps/fa3_commit_validation_20260729/README.md
```

性能和 NCU 证据位于：

```text
/home/lutao/zhiyuan/small_exps/fa3_active_wgmma_n_heuristic_20260729/
/home/lutao/zhiyuan/small_exps/fa3_paged_ptr_warp_shuffle_20260728/
/home/lutao/zhiyuan/small_exps/fa3_vllm_residual_gap_audit_20260728/
/home/lutao/zhiyuan/small_exps/fa3_autotune_cache_schema_20260728/
/home/lutao/zhiyuan/small_exps/fa3_residual_barrier_audit_20260728/
```

正式 H800 复测应满足：

1. 使用独立 autotune DB 和冷/热缓存两套记录；
2. vLLM 与 TLE 使用同一时钟、同一温度窗口；
3. 监控中 `foreign_sample_count=0`；
4. 同时跑四个 focused case、586 个 Qwen3.6 case 和 decode regression；
5. 分别报告逐 case 90%、逐 case 95% 和几何均值，不能只给平均数；
6. 对任何 compiler barrier 改动补跑 compute-sanitizer memcheck/racecheck。
