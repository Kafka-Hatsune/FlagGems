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

1. FlagTree 编译器提供通用非 2 次幂 logical tiled-SMEM allocation、
   active-N/active-K WGMMA、精确 pointer shuffle lowering 和 tiled alias
   Membar 分析。
2. FlagGems cost model 将 decode、普通 prefill、D256 paged prefill 和
   ragged mixed batch 分流到不同执行计划。
3. Triton autotune 只在满足静态安全契约的候选中实测，并用完整语义 key
   持久化 winner。

本次通用 allocation 重构后的冷启动受控 24-case H100 gate 中，prefill
几何值为 vLLM 的 `98.34%`，最差 B1 为 `91.06%`；mixed 几何值为
`103.18%`，最差 B2 为 `92.10%`，8/11 达到 95%。因此通用 tiled
allocation、N80 和 dispatch 优化已经缩小差距，但“prefill 和 mixed
均达到 vLLM 95%”尚未完成。B1/B2 同进程 ABBA 仍证明生产 N80 相对 N64
分别快 `1.098x`/`1.134x`；它只能证明 N80 方向正确，不能替代与 vLLM
在同温度窗口的正式比较。

decode 没有进入 tiled-N80 候选；policy 12 与 policy 13 的 10 个 decode
case 几何性能保留约 `99.72%`，单 case 波动范围约 `-1.29%～+2.04%`。
因此目前证据支持“没有结构性 decode 回退”，但不把亚百分比温度噪声写成
严格的逐 case 零回退。

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
| `fc364eba4` | `buffered_tensor_type` 与 `tl.block_type` 解耦；logical allocation、tiled stage/tile view 及 verifier | 通用 TLE buffer/frontend |
| `eb2e640ab` | 通用 pipe coverage、memory effect、Membar alias 与 warp-specialization token lowering | 通用流水与同步 |
| `997945d1a` | 从 tiled operand 推导 active WGMMA descriptor、carrier 和 K repetition | NVIDIA Hopper lowering |

### FlagGems

| Commit | 内容 | 性质 |
| --- | --- | --- |
| `f730d7da` | attention producer、persistent kernel、SMEM cost model 和 autotune 迁移到 logical tiled allocation | 运行时迁移 |
| 本文档提交 | frontend/scheduler 回归测试、实验记录和实现说明 | 验证与清理 |

依赖关系如下：

```text
FlagTree fc364eba4
    ├─ logical shape/storage shape + tiled views
    v
FlagTree eb2e640ab
    ├─ pipe coverage + alias/Membar + warp specialization
    v
FlagTree 997945d1a
    ├─ NVIDIA active WGMMA descriptor/codegen
    v
FlagGems f730d7da
    └─ attention producer、cost model、autotune 与 benchmark
```

旧 `tle.gpu.alloc_compact_kv`、compact buffer 类型、compact IR Op 和
专用 verifier 已全部删除，不提供兼容 alias。FlagGems 只描述逻辑 shape：

```python
k_smem = tle.gpu.alloc(
    [NUM_BUFFERS_KV, ACTIVE_WGMMA_N, HEAD_DIM_PADDED],
    INPUT_DTYPE,
)
```

`f730d7da` 因而依赖上面三个 FlagTree 提交，不能只在上游 Triton 环境中
单独运行。改动均位于 TLE 和 `third_party/nvidia` 的 `__TLE__` 路径；
标准 Triton register tensor、标准 `ttg.memdesc` 约束和其他后端行为不变。

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
    |                                      +-- exact tiled N80 candidate
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
- decode 不进入 tiled N80 prefill 配置，也不启用 `STAGGER_KV`、
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
和 tiled-N80 候选中选择。

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
- `EARLY_CAST_P`；
- `RESCALE_O_BEFORE_PV`。

这些维度并非对所有 shape 做无约束笛卡尔积。pruner 先用静态安全契约裁剪：

- tiled N80 只允许 FP16、D256、paged、causal、GQA；
- 仅允许 `(page16, GQA4)` 或 `(page32, GQA8)`；
- 必须是 packed prefill，而不能是 decode、Split-K、local、alibi、
  softcap 或 S auxiliary；
- tiled path 固定双 MMA group、双 KV buffer、TMA Q/O、staggered KV
  和 early FP16 P；
- 实测保留的 tiled-N80 搜索维度只有
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

LibTuner 的一个 winner 表只有一套 value columns。N64 config 不包含
`EARLY_CAST_P` 等 optional metadata；若它先创建表，后来的 N80 winner
就可能无法写入。

`_normalize_persistent_config_schema()` 会在搜索集合中发现所有实际出现的
optional fields，并给每个 config 补齐默认值。这样无论 N64 还是 N80
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

这次 generic tiled-SMEM 改变了 N80 的 allocation 和 producer/WGMMA
指令组合，因此把 `AUTOTUNE_POLICY_VERSION` 从 12 提升到 13，避免复用
旧 physical-N128 backing 测得的 winner。`ACTIVE_WGMMA_N` 没有 0
哨兵；旧 YAML/SQLite row 缺少该列时只按 `BLOCK_N` 补齐。N80 是新的完整
config value，不再从 `COMPACT_KV80` 布尔量推导。

### head dim 启发式与候选生成契约

先把运行时 head dim 补齐为 `Dpad`，再构造 1P2C 搜索面：

| `Dpad` 和 eligibility | 进入搜索的逻辑 N | storage |
| --- | --- | --- |
| `Dpad <= 128` | N128、N64 | ordinary N128、ordinary N64 |
| `Dpad == 256` 且 tiled profile 合法 | N80、N64 | exact tiled N80、ordinary N64 |
| 其他情况 | N64 | ordinary N64 |

这个选择来自固定 FP16 1P2C payload
`Dpad * (256 + 8 * N)` bytes：D64/D128 可以容纳 N128；D256 的 ordinary
N128 超出当前安全预算，而 exact tiled N80 是已经实测的最大窄 tile，N64
保留为 no-regression 候选。表中的“tiled profile 合法”还包含既有的 FP16、
D256、paged、causal、PackGQA、page/GQA 等契约，不是只看 `Dpad`。
这里 D64/D128 的 N128 与 N64 是两个不同的 ordinary 物理 tile，分别满足
`ACTIVE_WGMMA_N == BLOCK_N`；它们不是“在同一个 N128 backing 中用
active-N64”。当前唯一允许 active extent 小于物理 carrier 的 production
配置仍是 tiled D256/N80。

`ACTIVE_WGMMA_N` 不再使用 0 表示另一个 API overload。每个 config 都携带
正的逻辑 N，且满足 `16 <= ACTIVE_WGMMA_N <= BLOCK_N`。当前生成器只产生
N64、N80 和 N128，这三个值在构造候选时已经满足 N16 流水粒度，不在
kernel、pruner 和 config consumer 中重复检查 `% 16`。

这里的“搜索参数”不是把 16 到 `BLOCK_N` 的所有倍数独立地与 storage 做
笛卡尔积。当前合法 config 对由 storage 决定：

- ordinary N64/N128：`ACTIVE_WGMMA_N=BLOCK_N`；
- tiled N80：物理 accumulator carrier 为 N128，
  `ACTIVE_WGMMA_N=80`，K/V backing 为精确 N80。

因此 head-dim heuristic 先生成完整、合法的 config 对，再由 autotuner 实测
选择，而不是先生成任意 active N、再在 kernel 内拒绝不一致状态。partial-N80
仍只在 BM128、1P2C、Q1/KV2 的 tiled profile 中出现；decode、Split-K、
1P1C、BM64 或 Q 双缓冲使用 ordinary config，其正值
`ACTIVE_WGMMA_N` 等于各自的 `BLOCK_N`。逻辑 tile 直接按

```text
LOGICAL_BLOCK_N = ACTIVE_WGMMA_N
```

计算。因此 producer tile count、offset、causal boundary 和 SMEM cost
全部直接使用同一个正值 `ACTIVE_WGMMA_N`。对于 N80，allocation frontend
根据 logical shape 自动选择 exact tiled storage；WGMMA accumulator 仍用
N128 register carrier，前 80 列有效，`[80,128)` 在 softmax 前恢复为
mask tail。这里 carrier padding 只存在于寄存器/SSA 语义中，K/V shared
backing 不含 48 行 padding。将来开放 N96/N112 时也走同一 logical
allocation，无需再增加 shape 专用布尔量或 allocator。

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
`ttng.warp_group_dot` 且不附加属性；partial carrier 也使用同一个标准
Op，只额外附加 `tle.wgmma_active_n` 或 `tle.wgmma_active_k` 私有属性。
因此 TTNG 层不再定义 `ttng.tle_warp_group_dot`。这把“config 中总是正值”
和“full-N 必须保持旧 codegen”分离开：FlagGems 不需要数字哨兵或四处重复
constexpr 分支，ordinary/decode 仍得到原来的同步分析和 PTX；tiled N80
则由 NVIDIA-TLE WGMMA lowering 读取属性，生成 partial native opcode。
编译器回归测试还覆盖 M128 full-N，避免统一接口被 active-N 专用的 M64
约束误伤。

标准 Op 方案的关键不在于简单改名，而是让 active extent 自动继承
`WarpGroupDotOp` 已有的 memory effects、layout anchor、Membar、
canonicalization、commit/wait pipeline 和 LLVM conversion pattern。唯一
需要额外阻止的原生变换是 register/shared WGMMA 的 K-split：对于
`active_k=80`，物理 A carrier 是 K128，但只能发出 5 个 K16 repetition；
若按物理 K128 拆成 8 个标准 dot，就会重新计算被裁掉的 48 行。因此
`splitRSDot` 遇到 `tle.wgmma_active_k` 时保持整体。active-N 不改变 K
repetition，可以正常参与标准 K-split。

### 编译器支持范围与 production 搜索范围

FlagTree generic tiled allocation 首版接受 FP16/BF16、`rows % 16 == 0`、
`cols % 64 == 0`、`cols <= 256`。WGMMA/PTX 回归覆盖 active N/K
`16,32,48,64,80,96,112,128`，因此 N96/N112 已经不再受 storage/view
能力限制：

- K/V logical storage 会分别映射成 6/7 个 N16 row tile；
- QK 使用 `active_n=96/112`；
- PV 使用同值 `active_k`，分别发出 6/7 个 K16 repetition；
- accumulator 的物理 carrier 和 tail restore 由 NVIDIA lowering 管理。

不过“compiler 能生成”不等于“production 一定应搜索”。当前 cost model
只保留 N64/N80/N128：它们覆盖了 D256 的已测最优点以及 D64/D128 的
ordinary no-regression 点。N96/N112 若加入 production，应作为新的
autotune value 与 page size、GQA、rescale 等一起实测，而不是凭 shared
容量公式直接假设更快。这样把能力边界与性能策略边界分开：编译器是通用的，
FlagGems 搜索面仍保持小而有证据。

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
`0.00146484375`。tiled N80
另在 small16、small32、B5、B25 和 decode 上通过 5/5 gate：三次重复均
bitwise stable，tiled N80 与 legacy N64 的最大绝对误差为
`0.00048828125`，small case 相对 PyTorch FP32 的最大绝对误差为
`0.00096774101`，decode 为 `0.00004330277`。去掉哨兵后的 decode pruner
仍不保留 tiled config，ordinary config 分别携带 N64 或 N128。

最终 compiler canonicalization 的检查比延迟 gate 更严格：ordinary
kernel 的 TTGIR 只有标准 `ttng.warp_group_dot`，且没有 active extent
属性；其 TTGIR、LLVM IR、PTX 和 cubin 在标准 Op 重构前后逐字节相同。
tiled N80 kernel 也不再包含独立 TLE Op，而是 34 个标准
`ttng.warp_group_dot`，其中 6 个携带 `tle.wgmma_active_n=80`、6 个携带
`tle.wgmma_active_k=80`。最终 PTX 仍有 96 条 `m64n80k16`，PV 仍只发出
5 个 K16 repetition，说明完整 pass pipeline 没有丢失 partial extent。

标准 pipeline 还为循环内 active op 正常插入
`ttng.warp_group_dot_commit`。所选 tiled kernel 的
`fence.proxy.async`、WGMMA fence/mma/commit/wait、mbarrier 和
`membar.cta` 指令数量重构前后不变；PTX diff 只消除了 4 个位于
`wgmma.wait_group 1` 与 release `mbarrier.arrive` 之间的冗余
`bar.sync 128`。五组实际 GPU 正确性均通过，三次重复 bitwise stable；
短时 ABBA 中 B1 不回退、B5 基本中性，三个 decode kernel 的 cubin 与
重构前完全相同。

修改前后还使用相同 H100、相同 shape 和三轮 ABBA 快速 gate 做了 focused
对照：

| Case | before 最优 N80 | after 最优 N80 | after / before | N64 / 最优 N80：before → after |
| --- | ---: | ---: | ---: | ---: |
| B1 prefill | 4.52024 ms | 4.36138 ms | 0.96486 | 1.08903x → 1.10732x |
| B5 mixed | 0.39897 ms | 0.38779 ms | 0.97198 | 1.09130x → 1.10404x |

B1 和 B5 的历史 harness winner 前后都是 `compact_bn80_s1_e1`（该名称仅
是实验 variant label，当前 API 已无 compact 类型）；其他 shape 上 rescale
仍应交给 per-shape autotune。
表中的最后一列读取逐轮 ABBA 的 paired median，而不是用跨候选的全局
median 相除。
三个 decode case 的 after/before 分别为 1.00410、0.98825 和 1.00039，
几何均值约为 0.9975。这两份记录的
`status=passed-provider`，但 `formal_valid=false`：它们是用于发现明显回退
的探索性 H100 结果，不是独占、锁频条件下的正式 H800 性能证明。
最终 `after_final.json` 记录的 FlagGems commit 为 `6d4cb101`，Hopper
目录 diff 的 SHA-256 是标准空内容摘要，排除了开发中间态污染结果。

NCU 对同一 B1 launch 固定 ordinary N64 和 exact tiled N80 后得到：

| NCU metric | ordinary N64 | exact tiled N80 |
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

## 优化四：通用非 2 次幂 tiled-SMEM

### 为什么是 N80

目标 D256 paged profile 的最优逻辑 KV tile 为 N80。N64 增加循环次数，
N128 又会为 48 个无效列承担 shared memory、softmax 和 WGMMA carrier
开销。

调用方现在只写逻辑 allocation：

```python
k_smem = tle.gpu.alloc([2, 80, 256], tl.float16)
```

frontend 在编译期推导：

```text
capacity = 2
row_tiles = 80 / 16 = 5
col_tiles = 256 / 64 = 4
tiles_per_stage = 5 * 4 = 20
storage = [2 * 20, 16, 64] = [40,16,64]
```

这是元素一一映射，不是先分配 `[2,128,256]` 再 reshape：

```text
logical elements = 2 * 80 * 256 = 40,960
storage elements = 40 * 16 * 64 = 40,960
```

因此 K 和 V 各自的 backing allocation 都使用物理 `[40,16,64]`：

- 2 个 KV stages；
- 每个 stage 20 个 `N16 x D64` atom；
- N 方向 5 份，D 方向 4 份；
- producer 每解析一个 N16 page row，即可填充四个 D64 panel。

`slot(stage)` 暴露一个逻辑 `[80,256]` stage，producer tile view 暴露一个
`[16,64]` atom；两者都只是 root allocation 的 alias，不产生第二份
shared memory。flat tile 由
`stage * row_tiles * col_tiles + row_tile * col_tiles + col_tile`
计算。pipe coverage 用 BitVector 验证一个 stage 的 20 个 atom 恰好各写
一次，并验证总 copy bytes 等于 `80*256*2=40,960` bytes。Membar 根据
root、stage 和 tile interval 建模，不再识别固定 `[40,16,64]` 或 20-bit
mask。

对当前 1P2C、Q 单缓冲、KV 双缓冲 kernel，核心 payload 为：

```text
Q: 2 consumers * 1 Q stage * [64,256] * 2 bytes = 64 KiB
K: 2 KV stages * [80,256] * 2 bytes = 80 KiB
V: 2 KV stages * [80,256] * 2 bytes = 80 KiB
payload total = 224 KiB
```

编译 metadata 实测 dynamic SMEM 为 `229,716` bytes；再加 `1,024` bytes
static SMEM，总计 `230,740` bytes，距离 Hopper 单 block 上限仍有
`1,708` bytes。多出的部分来自 barrier、pipe state 等控制结构，不是
N128 carrier padding。

### active-N 和 active-K

QK 使用 `active_n=80`，生成原生 `m64n80k16`；PV 使用
`active_k=80`，只发出 5 个 K16 repetition，而不是物理 N128 所需的 8 个。
最终 production trace 中每个逻辑 tile 是：

- 16 条 `HGMMA.64x80x16` 完成 QK；
- 5 条 `HGMMA.64x256x16` 完成 PV。

这里需要区分两个“物理”概念：

- shared storage physical shape 是 exact `[40,16,64]`，没有 N128；
- accumulator/descriptor carrier 为了兼容 Triton register layout 可以是
  N128，但 lowering 只对前 80 列构造 WGMMA，tail 不参与数学计算。

因此 active extent 的目的确实达到了：shared payload 随 N80 缩小，PV
也从 8 次 K repetition 缩到 5 次；carrier 只承担 pass pipeline 中的类型
和 accumulator 布局，不会重新分配一块 N128 shared tensor。

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
所以保留为两个 tiled-N80 candidate 的唯一实测笛卡尔维度，由 per-shape
autotune 决定。

H100 同进程正式 ABBA 中，最终 tiled-N80 candidate 相对 legacy BN64 的
paired median speedup 为：

| Case | tiled N80 / N64 |
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

同步边界必须区分。下面的 owner 指生成该 PTX 的 IR op 或 pass，而不是源码
中恰好出现它的 Python helper：

| PTX instruction | purpose | issuing threads | owner op/pass | required? |
| --- | --- | --- | --- | --- |
| `wgmma.wait_group.sync.aligned N` | WGMMA async completion：保证被等待 group 的 accumulator/SMEM 读取完成 | 完整 consumer warpgroup | `ttng.warp_group_dot_wait` → WGMMA LLVM lowering | 必须保留 |
| `bar.sync id,128` | rendezvous：consumer warpgroup 内线程会合；不表示 WGMMA 完成或 stage publication | consumer 的 128 threads | Membar analysis / warp-specialization barrier insertion | 仅在 completion SSA、participation 和 alias 都已精确证明时可删 |
| `mbarrier.arrive...` | mbarrier publication：consumer 发布 stage 已可被 producer 复用，或 producer 发布 copy 已完成 | elected thread 或规定的 participant set | `tle.gpu.barrier_arrive` / producer-consumer pipe lowering | 必须保留 |
| `mbarrier.try_wait...` | acquire：等待对应 phase 的 publication 后才访问或覆盖 stage | waiter partition | `tle.gpu.barrier_wait` / pipe lowering | 必须保留 |
| `cp.async.bulk...mbarrier::complete_tx::bytes` | 发起异步 TMA/cp.async transaction，并将完成字节记入 stage barrier | producer elected lane/warp，取决于 copy 形式 | `tle.gpu.copy` → `ttg.tma_copy`/cp.async lowering | copy 存在时必须保留 |
| `cp.async.wait_group N` | async copy completion：约束普通 cp.async group 的完成，不等价于 CTA rendezvous | 发起 copy 的 producer warp | async-wait lowering | cp.async 路径必须保留 |
| `fence.proxy.async.shared::cta` | proxy ordering：generic shared 写与 async/TMA proxy 观察顺序 | 负责 publish 的 producer/consumer threads | proxy fence insertion / TLE pipe lowering | 跨 proxy 可见性需要时必须保留 |
| `membar.cta` | CTA 内普通 memory ordering；不代替 barrier participation | 相关 partition threads | Membar pass | 仅精确 alias 证明不相交时可删 |

不能用“已经 wait 了”笼统删除后续 barrier。WGMMA completion、线程会合、
proxy ordering 和 storage-lifetime publication 是四条独立正确性边。

generic tiled alias 初版曾把 storage root 的 rank-3/rank-2 顺序识别错，
导致 4 个 completed-WGMMA release rendezvous 回来：旧 production baseline
为 41 条 `bar.sync`，错误版本为 45 条。修复
`isTiledSMEMStorageRoot` 并按 `(root,stage,tile interval)` 建模后，最终为
40 条，其中 barrier id 2/3 分别是 10/9 条，与旧版 10/9 一致；额外少的
1 条来自 generic atom interval 能证明的新 disjoint case，不是扩大
completed-WGMMA 特判。register、shared、stack/local、WGMMA、cp.async、
mbarrier 和 proxy fence 没有随此修复变化。

focused lit 覆盖 tiled view、pipe coverage、active extent、pipeline safety
与 completed-WGMMA release，共 13/13 通过；small16/small32、B1/B2/B5/B25
和 decode 正确性均通过。这里的结论是同步结构与 alias 精度正确，不能把
少一条 barrier 直接换算成固定百分比的运行时加速。

## 最终性能与 decode 回归门

性能百分比定义为：

```text
vLLM latency / TLE latency * 100%
```

### 24-case 冷启动受控 gate

同一份 shape、同为 CUDA Graph、warmup 20、iteration 200。FlagGems
从 43°C 开始、60°C 结束；vLLM 从 45°C 开始、69°C 结束，运行期间均未
触发 thermal throttle。结果为：

| Phase | case 数 | 几何均值 | 最差 | ≥95% |
| --- | ---: | ---: | ---: | ---: |
| prefill | 3 | 98.34% | 91.06% | 2/3 |
| mixed | 11 | 103.18% | 92.10% | 8/11 |
| decode（相对 vLLM） | 10 | 148.87% | 84.52% | 8/10 |

decode 表中的最差值不是本次改造引起的回退：该路径没有 N80 config，且同一
FlagGems 实现 policy 12→13 的几何性能保留约 99.72%。它只说明某个既有
decode shape 本来就不如 vLLM，不应把“相对 vLLM <95%”和“本次回退”
混为一谈。

低于 95% 的主要长路径绝对值如下：

| Case | FlagGems | vLLM | vLLM / FlagGems |
| --- | ---: | ---: | ---: |
| B1 long prefill | 4.7281 ms | 4.3052 ms | 91.06% |
| B2 ragged mixed | 1.5902 ms | 1.4646 ms | 92.10% |
| B5 mixed | 0.4358 ms | 0.4061 ms | 93.19% |

### 临界 case 的 ABBA

早期跨进程 gate 曾使 GPU 达到 84°C，并累计 `SW Thermal Slowdown`
约 19.8 秒，因此该轮 95% 结论作废。作为 N64/N80 内部选择证据，B1/B2
另使用同进程 CUDA Graph ABBA：

| Case | winner | winner median | paired N64/N80 speedup |
| --- | --- | ---: | ---: |
| B1 long prefill | N80, stagger, early-cast | 4.928 ms | 1.0977x |
| B2 mixed | N80, stagger, early-cast | 1.607 ms | 1.1338x |

B2 曾有另一温度窗口的 vLLM 绝对值 1.575 ms，不能与这份 ABBA 的
FlagGems 数值拼接成正式百分比。本文以冷启动受控 gate 的 92.10% 为准，
不用挑选最好的一轮覆盖它。

## 为什么 B1/B2/B5 还没有达到 95%

通用重构后的 B1 NCU 因旧 policy-12 DB 命中了 N64，可用于解释 N64
为什么仍落后于 vLLM N80，而不能冒充 N80 profile：

| Metric | FlagGems N64 | vLLM N80 |
| --- | ---: | ---: |
| duration（NCU base clock） | 4.98 ms | 4.21 ms |
| registers/thread | 168 | 168 |
| achieved occupancy | 18.75% | 18.75% |
| executed instructions | 1.016B | 0.708B |
| chip instructions | 1.160B | 1.027B |
| eligible warps/cycle | 0.78 | 0.65 |
| compute SOL | 67.9% | 78.9% |
| memory SOL | 65.3% | 59.8% |
| barrier stall/issue | 0.362 | 0.342 |
| long-scoreboard stall/issue | 0.923 | 0.925 |

两边 global/L2 sectors 已接近，occupancy 相同，而且 FlagGems eligible
warps 反而更多；所以 N64 的主要问题不是等待不足、barrier 或显存请求数，
而是更多 tile 带来的控制、softmax 和 address instructions。N80 的价值
正是少做约 20% logical KV tile，而不是提高理论 occupancy。

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

B1/B2/B5 的缺口也与 shape 特征吻合：

- B1 是规则长 prefill，矩阵乘已被 N80 压缩，但 per-tile softmax carrier
  和 descriptor/address 指令仍比 vLLM 多；
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
- shared-ring `REUSE_KV_PAGE_STATE`：最终可编译且 small16/small32/B1/B2
  正确，但 24-case 中 B1 `4.7281→4.8781 ms`、B2
  `1.5902→1.6021 ms`、B5 `0.4358→0.4378 ms`，因此完整撤回；
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

当前 tiled producer 每个 N80 tile 为 K 读取 5 个 page entry，staggered
V 又读取 5 个；vLLM 热循环只读取 1 次并保留 page/page-offset 状态。

已经排除“用 `[NUM_BUFFERS_KV,8]` shared ring 在 K/V 间传递 page id”
的直接方案：它虽然只增加 64 B SMEM、保持 168 registers 和 0 spill，
但 shared round trip 与 acquire 前的状态物化依赖超过了省下的 5 次读取。

若继续研究，合理迭代顺序是：

1. 让 producer loop 以 register/SSA tuple carry K 的 5 个 page id，避免
   shared round trip；
2. 让 paged copy helper 原生消费通用 page-id vector；
3. 再用 5-lane 或 8-lane carrier 并行读取并 shuffle，使 5 次降到 1 次。

前一版 shared-ring 实现不能复用，应重新从 loop-carried state 类型和
helper lowering 契约设计。

### 3. Tiled WGMMA descriptor 分组预编码

当前每条 WGMMA 都可能重复 raw address 的 `shr/and/add`。可为四个 D64
panel 共用一个 encoded stage base，再用整数 offset 派生 descriptor。

验收门：

- HGMMA 16/5 不变；
- `UIADD3.X` 降到每 tile 接近 2；
- register 不超过 168；
- local memory 与 spill 保持 0。

## 验证与复现

本次提交的 focused 验证：

- FlagTree `libtriton.so` 与工具增量 rebuild：通过；
- tiled view/pipe/Membar、active extent 与 pipeline safety：
  13/13 lit 通过；
- generic tiled-SMEM frontend：55/55 pytest 通过；
- FlagGems 变更模块 `py_compile`：通过；
- scheduler/autotune focused tests：7/7 pytest 通过；
- production attention small16/small32/decode correctness：通过；
- active N/K 16..128（16 的倍数）runtime/PTX smoke：通过；
- rejected marker scan 和 `git diff --check`：通过。

详细命令记录位于：

```text
/home/lutao/zhiyuan/small_exps/tle_generic_tiled_smem_20260729/README.md
```

性能和 NCU 证据位于：

```text
/home/lutao/zhiyuan/small_exps/fa3_active_wgmma_n_heuristic_20260729/
/home/lutao/zhiyuan/small_exps/tle_generic_tiled_smem_20260729/
/home/lutao/zhiyuan/small_exps/fa3_paged_state_reuse_20260729/
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
