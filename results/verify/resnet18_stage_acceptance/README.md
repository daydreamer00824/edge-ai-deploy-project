# ResNet18部署项目阶段验收

## 1. 验收范围

本次验收针对ResNet18端侧推理部署项目，主要验证：

* clean build；
* ORT CPU推理；
* ORT CUDA推理；
* TensorRT FP32推理；
* TensorRT FP16推理；
* 动态batch 1、8、32；
* 四后端统一benchmark；
* TensorRT device buffer复用；
* H2D、GPU inference、D2H分阶段计时；
* CSV、日志和异常复测证据完整性。

测试使用244张图片，统一设置：

```text
warmup = 3
repeat = 10
batch = 1、8、32
```

---

## 2. Clean build结果

删除原有`cpp/build`目录后，重新执行CMake配置和完整编译。

结果：

```text
CMake配置成功
所有目标编译成功
ORT与TensorRT可执行文件正常生成
未依赖旧构建缓存
```

构建日志：

```text
cmake_configure.log
clean_build.log
```

---

## 3. 四后端功能回归

完成以下12组测试：

```text
ORT CPU：batch 1、8、32
ORT CUDA：batch 1、8、32
TensorRT FP32：batch 1、8、32
TensorRT FP16：batch 1、8、32
```

结果：

```text
12组测试全部完成
未发生程序崩溃
未发生CUDA错误
未发生Provider加载错误
CSV字段和行数正确
12份独立日志完整保存
```

结构化结果：

```text
four_backend_recheck.csv
```

---

## 4. TensorRT阶段计时验证

TensorRT推理使用CUDA Event划分：

```text
H2D
→ GPU inference
→ D2H
```

并验证：

```text
GPU total ≈ H2D + GPU inference + D2H
```

六组FP32、FP16及batch 1、8、32测试中，阶段计时均闭合。

---

## 5. Device buffer复用结果

将device buffer从每次推理执行`cudaMalloc/cudaFree`，改为容量不足时申请并跨推理复用。

对比结果：

```text
TensorRT run墙钟时间降低：9.65%～27.58%
端到端延迟降低：1.12%～11.05%
吞吐量提升：1.14%～12.43%
```

收益主要来自减少显存申请、释放及相关Host端管理开销，不能表述为直接加速GPU模型计算。

完整证据：

```text
results/benchmark/trt_buffer_reuse_comparison/
```

---

## 6. FP32 batch=32性能波动

clean build回归中，TensorRT FP32 batch=32出现GPU推理时间和P95延迟升高。

进行了以下验证：

1. 5次独立复测；
2. 每次warmup=5、repeat=30；
3. 采集GPU温度、频率、性能状态、功耗和利用率；
4. 检查其他GPU计算进程。

遥测结果显示：

```text
温度：42～45℃
功耗：约20～50 W，未接近170 W上限
其他CUDA进程：未发现
前半段性能状态：P2
前半段SM频率：1807 MHz
后半段性能状态：P3/P5
后半段SM频率：607～1042 MHz
```

因此未发现热降频、功耗限制或其他CUDA进程竞争。

运行后半段存在明显GPU动态频率下降。该性能波动更符合GPU动态电源管理和间歇性负载状态变化，不能归因于已证实的代码回退。

遥测证据：

```text
fp32_batch32_telemetry/gpu_telemetry.csv
fp32_batch32_telemetry/stage.csv
fp32_batch32_telemetry/run.log
```

---

## 7. 最终验收结论

| 验收项           | 结论                      |
| ------------- | ----------------------- |
| Clean build   | 通过                      |
| ORT CPU       | 通过                      |
| ORT CUDA      | 通过                      |
| TensorRT FP32 | 功能通过，batch=32存在环境相关性能波动 |
| TensorRT FP16 | 通过                      |
| 动态batch       | 通过                      |
| Buffer复用      | 通过                      |
| GPU阶段计时       | 通过                      |
| 四后端benchmark  | 通过                      |
| CSV与日志证据      | 通过                      |
| 异常复测与遥测       | 通过                      |

最终结论：

> ResNet18端侧推理部署项目的核心工程链路、四后端推理、动态batch、性能测试、buffer复用和GPU阶段计时均完成验收。FP32 batch=32存在GPU动态频率相关的运行时性能波动，但未发现功能错误或能够归因于代码的稳定性能回退。

ResNet18阶段可以停止继续扩展功能，进入README封装、项目介绍和后续YOLO检测部署阶段。
