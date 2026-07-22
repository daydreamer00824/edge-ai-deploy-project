# TensorRT Device Buffer复用前后性能对比

## 1. 对比目的

本实验用于验证TensorRT C++推理中，将device buffer从“每次推理动态申请和释放”改为“按容量申请并跨推理复用”后，对运行阶段延迟、端到端延迟和吞吐量的实际影响。

对比的核心变量为device buffer生命周期：

### 复用前

每次调用`TrtRunner::run()`时执行：

```text
cudaMalloc输入buffer
→ cudaMalloc输出buffer
→ H2D
→ TensorRT推理
→ D2H
→ cudaFree输入buffer
→ cudaFree输出buffer
```

### 复用后

仅在当前buffer容量不足时申请或扩容：

```text
检查buffer容量
→ 必要时申请或扩容
→ H2D
→ TensorRT推理
→ D2H
→ 后续推理继续复用
→ TrtRunner析构时统一释放
```

---

## 2. 测试条件

复用前后使用相同测试条件：

| 项目         | 配置                                             |
| ---------- | ---------------------------------------------- |
| 模型         | ResNet18                                       |
| 输入尺寸       | N × 3 × 224 × 224                              |
| TensorRT精度 | FP32、FP16                                      |
| Batch      | 1、8、32                                         |
| 图片数量       | 244                                            |
| Warmup     | 3                                              |
| Repeat     | 10                                             |
| GPU阶段计时    | CUDA Event                                     |
| CPU墙钟计时    | `std::chrono::steady_clock`                    |
| 测试阶段       | preprocess、TensorRT run、postprocess、end-to-end |

GPU阶段划分为：

```text
H2D
→ GPU inference
→ D2H
```

并验证：

```text
GPU total ≈ H2D + GPU inference + D2H
```

---

## 3. 代码与结果来源

### 复用前基线

复用前代码基于提交：

```text
f40358b benchmark: add four-backend performance comparison
```

在本地实验分支中移植统一的CUDA Event计时代码：

```text
00392ea experiment: add unified timing to pre-reuse TensorRT baseline
```

该版本仍在每次`run()`中执行局部：

```cpp
cudaMalloc(...)
cudaFree(...)
```

复用前原始结果位于：

```text
pre_reuse/trt_stage_timing_pre_reuse.csv
```

### 复用后版本

Buffer复用功能提交：

```text
f5ab7f5 perf: reuse TensorRT device buffers
```

GPU分阶段计时代码提交：

```text
f78d032 feat: add TensorRT GPU stage timing benchmark
```

正式结果提交：

```text
2e0c3cb docs: add TensorRT stage timing results and validation
```

复用后原始结果位于：

```text
../trt_stage_timing/trt_stage_timing.csv
```

---

## 4. 核心对比结果

| 精度   | Batch | TensorRT run降低 | 端到端延迟降低 |  P95降低 |  FPS提升 |
| ---- | ----: | -------------: | ------: | -----: | -----: |
| FP32 |     1 |         14.95% |   6.64% |  5.92% |  7.11% |
| FP32 |     8 |          9.65% |   2.67% |  0.55% |  2.75% |
| FP32 |    32 |         14.17% |   1.27% | 15.15% |  1.28% |
| FP16 |     1 |         27.58% |  11.05% |  8.75% | 12.43% |
| FP16 |     8 |         13.04% |   1.12% |  0.42% |  1.14% |
| FP16 |    32 |         13.89% |   3.09% |  2.27% |  3.19% |

完整结构化结果见：

```text
trt_buffer_reuse_comparison.csv
```

---

## 5. TensorRT运行阶段收益

六组测试中，复用后的`avg_trt_run_wall_ms`均低于复用前。

| 精度   | Batch |          复用前 |          复用后 |          降低 |
| ---- | ----: | -----------: | -----------: | ----------: |
| FP32 |     1 |  2.914397 ms |  2.478710 ms | 0.435687 ms |
| FP32 |     8 |  7.988833 ms |  7.217615 ms | 0.771218 ms |
| FP32 |    32 | 22.050418 ms | 18.926348 ms | 3.124070 ms |
| FP16 |     1 |  2.139776 ms |  1.549688 ms | 0.590088 ms |
| FP16 |     8 |  5.291534 ms |  4.601688 ms | 0.689846 ms |
| FP16 |    32 | 15.486136 ms | 13.335142 ms | 2.150994 ms |

TensorRT运行阶段降幅范围为：

```text
9.65%～27.58%
```

这说明移除每次推理主路径中的重复显存申请和释放，可以稳定降低`TrtRunner::run()`的CPU墙钟时间。

---

## 6. Host端额外开销变化

使用：

```text
avg_trt_run_wall_ms - avg_gpu_total_ms
```

作为Host端及GPU阶段之外额外开销的观察指标。

该差值包含但不限于：

* `cudaMalloc/cudaFree`；
* TensorRT CPU侧接口调用；
* CUDA Event提交和读取；
* stream同步；
* Host端容器及其他少量管理开销。

复用前后该差值降低：

| 精度   | Batch |          降低 |
| ---- | ----: | ----------: |
| FP32 |     1 | 0.438032 ms |
| FP32 |     8 | 1.022804 ms |
| FP32 |    32 | 1.457014 ms |
| FP16 |     1 | 0.465946 ms |
| FP16 |     8 | 0.725801 ms |
| FP16 |    32 | 1.527786 ms |

该指标不能被表述为“纯`cudaMalloc/cudaFree`耗时”，但结合复用前后的代码差异，可以判断运行阶段收益主要来自显存生命周期优化及其相关Host端开销减少。

---

## 7. 为什么端到端收益小于TensorRT运行阶段收益

Buffer复用使TensorRT运行阶段降低约：

```text
9.65%～27.58%
```

但端到端延迟只降低约：

```text
1.12%～11.05%
```

原因是端到端链路还包括CPU图像预处理和后处理。

特别是在较大batch下，CPU预处理占端到端延迟的大部分，因此TensorRT运行阶段的优化会被整体链路中的其他阶段稀释。

这说明后续进一步优化端到端性能时，应重点评估：

* 多线程预处理；
* Pinned Memory；
* 异步H2D；
* 数据预取；
* CPU与GPU流水线重叠。

---

## 8. Batch=1收益更明显

FP16、batch=1的收益最大：

```text
TensorRT run降低：27.58%
端到端延迟降低：11.05%
FPS提升：12.43%
```

batch较小时，GPU计算时间较短，固定的显存申请、释放及Host端管理开销占总运行时间的比例更高。

随着batch增大，模型计算和预处理耗时提高，固定开销在总时间中的占比下降，因此端到端收益比例通常减小。

---

## 9. GPU推理时间的解释边界

本实验不能证明buffer复用直接加速了GPU模型计算。

原因是：

* Buffer复用主要改变显存申请和释放策略；
* CUDA Event测得的GPU inference时间存在运行状态波动；
* 测试过程中没有同步记录GPU频率、温度、功耗和后台负载；
* WSL调度和GPU动态频率可能影响不同轮次结果。

因此准确结论是：

> Buffer复用稳定降低了TensorRT完整运行阶段的墙钟时间，收益主要来自避免每次推理重复申请和释放device buffer；GPU模型计算本身没有证据表明因buffer复用而产生稳定加速。

---

## 10. FP32 batch=8计时异常及复核

复用后正式结果中，FP32、batch=8出现：

```text
wall_minus_gpu_total_ms = -0.038818 ms
```

其绝对值约为GPU总时间的0.53%。

为验证该现象，额外进行了5次独立复测：

```text
batch=8
warmup=5
repeat=30
```

五次复测的`wall_minus_gpu_total_ms`均恢复为正：

```text
0.283752 ms
0.224113 ms
0.275672 ms
0.261794 ms
0.188926 ms
```

因此该负值未稳定复现，判断为CPU墙钟计时与CUDA Event计时之间的微小测量波动，而不是Event顺序或同步逻辑错误。

复测证据位于：

```text
../trt_stage_timing/fp32_batch8_recheck/
```

原始异常值未被删除或替换。

---

## 11. 最终结论

在当前ResNet18 TensorRT C++推理项目和测试环境中，device buffer复用取得以下效果：

1. 六组测试的TensorRT运行墙钟时间均降低；
2. TensorRT运行阶段降低约9.65%～27.58%；
3. 端到端延迟降低约1.12%～11.05%；
4. 吞吐量提升约1.14%～12.43%；
5. Batch=1场景收益最明显；
6. 较大batch下，CPU预处理成为影响端到端收益的主要因素；
7. Buffer复用降低的是显存生命周期及相关Host端开销，不能表述为直接加速GPU模型计算。

因此，device buffer成员化复用属于有效且必要的TensorRT C++工程优化。

---

## 12. 复现方式

### 运行复用后测试

```bash
bash scripts/run_trt_stage_timing_benchmark.sh
```

### 运行复用前基线

在本地实验worktree中执行：

```bash
bash scripts/run_trt_stage_timing_pre_reuse.sh
```

### 生成前后对比CSV

```bash
python3 scripts/compare_trt_buffer_reuse.py \
  --pre \
  results/benchmark/trt_buffer_reuse_comparison/pre_reuse/trt_stage_timing_pre_reuse.csv \
  --post \
  results/benchmark/trt_stage_timing/trt_stage_timing.csv \
  --output \
  results/benchmark/trt_buffer_reuse_comparison/trt_buffer_reuse_comparison.csv
```
