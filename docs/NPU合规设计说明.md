# NPU 合规设计说明（TI F28P55x / C2000）

本文回答一个问题：**这个 32×32 手写数字 CNN，凭什么说它符合 TI F28P55x NPU 要求？**
依据全部来自本机已实测的 TI 官方资料与工具链（路径可查证），编译证据见第 5 节。

---

## 1. 平台与依据来源

| 项 | 位置 / 说明 |
|---|---|
| NPU 支持层与参数约束 | `E:\desk\ti-npu\tinyml-tensorlab-main\tinyml-tensorlab-main\tinyml-modelzoo\docs\NPU_CONFIGURATION_GUIDELINES.md`（官方《NPU Configuration Guidelines for TI NNC》） |
| 转换/编译工具 | `E:\desk\ti-npu\saw2sin_npu\torch2tinpu`（本工程自建，saw2sin / charge_mtl 两次实战验证） |
| 部署示例（本模型） | `E:\desk\ti-npu\saw2sin_npu\examples\mnist_cnn\`（adapter + job） |
| 运行环境 | conda env `ti-npu`：`E:\anaconda\envs\ti-npu\python.exe`；`C2000_CG_ROOT` 由工具自动注入 |
| 历史经验 | 用户记忆 `ti-f28p55x-npu.md`（batch=1 / TI-QAT 必须 / 通道 m4 / FC iC≥16 / 坑位清单） |

## 2. 硬约束 → 本模型逐条对照

TI NNC 用户指南的核心原则：**"Layers that don't meet these constraints will fall back to software execution"**
（不满足约束的层会被静默退回 CPU 执行——不报错，但没加速）。因此设计阶段就要对齐：

| # | 约束（官方原文口径） | 本模型设计 | 状态 |
|---|---|---|---|
| 1 | **FCONV**（首层，iC=1）：oC 为 4 的倍数 | Conv2d(1→16, k3×3, pad1) | ✅ oC=16=m4 |
| 2 | **GCONV**（中间层）：iC、oC 均 m4 | Conv(16→32)、Conv(32→32) | ✅ 全 m4 |
| 3 | **MAXPOOL**：至少一维核尺寸 ∈1–4 | MaxPool2d(2×2) | ✅ |
| 4 | **FC**：输入特征 ≥16（8bit 权重） | Flatten 512 → Linear64 → Linear10 | ✅ 512、64 |
| 5 | **batch 必须静态 =1** | `get_example_input()` = (1,1,32,32) | ✅ |
| 6 | **只能用 TI-QAT**（CPU 量化/float 模型只能在 CPU 跑） | torch2tinpu QAT（15 epoch 自蒸馏） | ✅ 见第 5 节 |
| 7 | 图形态：PyTorch 的 MatMul+独立Add 不被识别（FC 回落） | 导出 ONNX 为 `Conv/Relu/MaxPool/Flatten/**Gemm**`，无 MatMul | ✅ 审计通过 |
| 8 | 编译配置 `quantization: 2`、纯 ASCII、opset 18 | 工具自动保障（防呆） | ✅ |

> 注：约束 7 是历史第一大坑（saw2sin "零卸载"）。本模型因为 `Flatten` 后是 2D 张量，
> PyTorch 直接导出为 **Gemm**（TI 认的 FC 形态），**无需**像纯 MLP 那样做 Linear→1×1 卷积重写。

## 3. 模型数字（runs/cnn32/model.onnx 口径）

```
输入 (1,1,32,32) [0,1] float
Conv2d(1→16,3x3,p1)+ReLU+MaxPool2   -> (16,16,16)
Conv2d(16→32,3x3,p1)+ReLU+MaxPool2  -> (32,8,8)
Conv2d(32→32,3x3,p1)+ReLU+MaxPool2  -> (32,4,4)
Flatten -> Linear(512→64)+ReLU -> Linear(64→10) -> logits (1,10)
总参数量 47,530 (~46 KB int8 权重) | 总算力 1.95 MMACs
```

- 算力/内存量级参考：charge_mtl（285k 参数 MLP）编译后 mod.a Total ≈16.9 KB → 本模型同为 6 万参数级，flash 完全无压力。
- 输入输出口径：C 侧按 **NCHW** 顺序填 float32（本机历史实测：头文件 I/O 即 NCHW 语义，转换由 NNC 内部处理）；
  `output_int=false` → 输出反量化回 float，**softmax/argmax 放 C 端**。

## 4. 怎么跑（部署命令）

```powershell
# 在 E:\desk\ti-npu\saw2sin_npu 下 (ti-npu 环境, 绝对路径 python)
E:\anaconda\envs\ti-npu\python.exe -m torch2tinpu check examples\mnist_cnn\job_mnist_cnn.yaml
E:\anaconda\envs\ti-npu\python.exe -m torch2tinpu all   examples\mnist_cnn\job_mnist_cnn.yaml
# all = convert(QAT→int8.onnx, 审计, 验收) → compile(mod.a) → deploy(集成包+C模板)
# 产物: examples\mnist_cnn\out\mnist_cnn\ (int8.onnx / audit_report.md / verify_report.json /
#        compile\summary.md / deploy\artifacts\mod.a + tvmgen_default.h + INTEGRATION.md)
```

Adapter（`examples\mnist_cnn\adapter_mnist_cnn.py`）直接复用 CNN_TWO 的 `model.py` 架构代码与
`runs\cnn32\best.pt` 权重——**PC 端与部署端单一事实源**；QAT 为自蒸馏（量化学生逼近 float 教师输出，loss=mse）。

## 5. 编译与量化结果（2026-10-09 实测完成）

### 5.1 量化验收（verify_report.json，n=200）

| 指标 | 值 |
|---|---|
| float 模型 acc | 0.995 |
| int8（TI-QAT）acc | 0.990 |

- QAT：15 epoch 自蒸馏（loss=mse，3.70e-2 → 3.13e-2）；⚠️ 本机在 CPU 上跑了约 84 min（下次把 `quant.device` 改 `cuda`）。
- 审计（audit_report.md）：opset=18、输入 [1,1,32,32] / 输出 [1,10]、量化格式 TI-NPU 风格，**0 error / 0 warning**。

### 5.2 编译与 NPU 卸载（compile/summary.md）

编译模式：**type=hard（真 NPU，非 CPU 软库）**；警告 0 条。

| 层 | 数量 | 上 NPU |
|---|---|---|
| FCONV | 1 | ✅ |
| GCONV | 2 | ✅ |
| FC | 2 | ✅（Gemm 形态被识别） |
| MAXPOOL | 3 | ✅ |
| PAD | 2 | ✅ |
| add/mul/floor/clip/cast/reshape/nn.pad×2/flatten/dequant/layout_transform | 12 | CPU（量化胶水 + 数据搬运，成本可忽略） |

mod.a 内存：Code 1.76 KB + RO 41.88 KB（权重）+ RW 24.98 KB（激活/工作区）= **总计 68.63 KB**；
F28P55（1.1 MB Flash + 共享 RAM）完全放得下，板端链接段照抄 `E:\desk\CCS\npu_and_charge`。

### 5.3 产物

- int8 ONNX：`examples\mnist_cnn\out\mnist_cnn\mnist_cnn_int8.onnx`
- 部署包：`...\deploy\`：`artifacts\mod.a + tvmgen_default.h`、`INTEGRATION.md`、`mnist_cnn_npu.h/.c`（I/O = float32，1024 入 → 10 出）

### 5.4 板端实测（2026-10-09，F28P550SJ9 LaunchPad）

工程 `E:\desk\CCS\cnn_and_number`（golden 验证 + 计时 + 文本串口；构建/烧录说明见其 `MNIST_CNN_README.md`）：

- **精度**：12 张 golden MNIST **全部 argmax 正确**（0 MISMATCH）；板端 int8 vs PC float logits 的 max|d| = **0.591**
  （与 PC 端量化验收 0.695 同量级 → 纯量化噪声，无功能性偏差）。
- **耗时**（口径 = `tvmgen_default_run + wait`，CPUTimer1@150MHz，每样本 32 次取统计）：

  | 版本 | 结构 | 参数/MACs | 板端耗时 (avg) | PC 测试集 float |
  |---|---|---|---|---|
  | 完整版 | (16,32,32) | 47,530 / 1.95M | **10,950.6 µs** | 99.23% |
  | 轻量版 | (8,16,16) | 20,666 / 0.53M | **4,908.6 µs** | 98.59% |

- **优化结论（两点粗拟合）**：`T ≈ 2.6 ms + 4.3 ns × MACs` → 该 NPU 属"低算力密度"型
  （边际 ≈ 1.5 MAC/周期 @150MHz）；**提速第一杠杆 = 减 MACs**（轻量版 MACs -73% → 时间 -55%）。
- 冷启动 = 稳态（10,950.2 vs 10,950.6 µs，无预热效应）；耗时抖动 ±8 µs（NPU 为确定性流水）。

## 6. 坑位备忘（本工程视角）

1. **口径一致性优先于调参**：训练/手写板/固件三处预处理必须是 `preproc.py` 同一套；QAT 校准数据必须与训练同分布（adapter 里直接取 MNIST 缓存）。
2. **输出缩放工程（÷128）**：本模型单头输出，不涉及 charge_mtl 那种"两段输出共享 scale"的问题，无需缩放；若以后加第二输出头要重新审视。
3. **不要手写 NHWC 转换**：C 侧按 NCHW 填数即可。
4. **板端集成**：抄 `E:\desk\CCS\npu_and_charge`（段配置 `.rodata.tvm→FLASH`、`.bss.noinit.tvm→RAMGS`、NPU 时钟/中断、`while(!tvmgen_default_finished)`）；**不要调用 `TI_NPU_init()`**。
5. **target_module 命名坑（本次实测）**：modelmaker 的合法模块名是 `timeseries / vision / audio / radar`（**不是工具注释里的 `image`**；`job.py` 不校验该字段值，写错要到编译才爆）；且 `vision` 模块对 F28P55 的支持任务列表为空 → **本案例用 `timeseries` + `generic_timeseries_regression` 通用预设编译图像 CNN 成功**（输出 float，softmax/argmax 在 C 端）。
6. **DSLite 烧录必须带 `-u/--run`**：不带时烧完 CPU 停住不跑（现象 = 串口静默；日志无 "Running… Success"）。
7. **大模型需要连续 GS RAM**：`.bss.noinit.tvm` 从 8KB（小模型时代）涨到 25.5KB → lnk 中合并 `RAMGS0..3` 为 `RAMGS0123`（32KB，物理连续，手法同 C2000Ware arc_fault 例程）；同时主程序 `.bss`（含 1024 点输入缓冲）从 RAMLS5(2KB) 挪到 RAMLS8(8KB)。
8. **换 mod.a 后注意时间戳**：`Copy-Item` 保留源文件时间戳 → gmake 可能判 "up to date" 不重链（板上还是旧模型）→ 手动刷新 `artifacts\mod.a` 时间戳后再 build。

## 7. 下一步（板端阶段建议）

1. 用 `deploy\INTEGRATION.md` + `mnist_cnn_npu.h/.c` 模板把 mod.a 集成进 CCS（F28P550SJ9 LaunchPad）。
2. 板端预处理流水线对齐 `preproc.py`（包围盒→20px→质心居中→32×32→/255）；用 golden vector 自检（PC 生成输入+输出对拍）。
3. 计时：CPUTimer 打点，目标量级参考 saw2sin（MLP 637µs/次）；CNN 1.95 MMACs 与 MLP 对比后实测为准。
