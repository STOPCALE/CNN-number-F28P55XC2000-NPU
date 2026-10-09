# cnn_and_number — MNIST CNN (32×32) F28P55 NPU 板端验证工程

> 从 `npu_and_charge` 脚手架改造（2026-10-09）：模型换成 CNN_TWO 训练的 **mnist_cnn**（TI-QAT int8），
> 主程序重写为「golden 验证 + 耗时统计 + 文本串口打印」。charge_mtl 时代产物已归档到 `_backup_charge_mtl\`。

## 1. 硬件与连接

- 开发板：TMS320F28P550SJ9 LaunchPad（XDS110 调试器 + 虚拟串口）
- 串口：**COM6 @ 115200-8N1**（SCIA，GPIO28/29，由 SysConfig 生成）
- 调试配置：`TMS320F28P550SJ9_LaunchPad.ccxml`

## 2. 模型与产物

- 模型：mnist_cnn（CNN_TWO 工程；Conv3x3 ×3 + MaxPool ×3 + FC；输入 1×1×32×32 → 输出 1×10）
- NPU 库：`artifacts\mod.a` + `artifacts\tvmgen_default.h`（type=hard；卸载 FCONV/GCONV×2/FC×2/MAXPOOL×3；0 警告）

| 版本 | 状态 | 产物来源 / 备份 |
|---|---|---|
| **完整版** (16,32,32) | **[当前板上]** | artifacts 已装；备份 `_backup_mnist_base\`（mod.a / tvmgen_default.h / cnn_and_number_base.out） |
| 轻量版 (8,16,16) | 实验存档 | 编译源 `E:\desk\ti-npu\saw2sin_npu\examples\mnist_cnn_lean\out\mnist_cnn_lean\deploy\artifacts\`；备份 `_backup_mnist_lean\`（含 .out） |

## 3. 构建（gmake 直编）

```powershell
E:\ccs\ccs\utils\bin\gmake.exe -C E:\desk\CCS\cnn_and_number\CPU1_LAUNCHXL_FLASH -j4 all
```

产物：`CPU1_LAUNCHXL_FLASH\cnn_and_number.out`（map：`cnn_and_number.map`）

## 4. 烧录（⚠️ 必须带 `-u`）

```powershell
E:\ccs\ccs\ccs_base\DebugServer\bin\DSLite.exe flash --config=E:\desk\CCS\cnn_and_number\TMS320F28P550SJ9_LaunchPad.ccxml -u E:\desk\CCS\cnn_and_number\CPU1_LAUNCHXL_FLASH\cnn_and_number.out
```

- **`-u / --run` 不能省**：不带时烧完 CPU 停住不运行（现象 = 串口永远静默）。
- 完整走完的日志特征：`Erasing Flash … Loading Program … "Setting PC to entry point" … Running… Success`。

## 5. 串口读取

```powershell
E:\anaconda\envs\torch\python.exe E:\desk\CCS\cnn_and_number\_diag\read_serial_cnn.py --seconds 10 --port COM6
```

- 输出存 `_diag\serial_cnn_capture.txt`；已归档记录：`serial_capture_baseline.txt`（完整版）/ `serial_capture_lean.txt`（轻量版）
- 运行前关闭 VOFA+ / 串口助手等占用串口的工具

## 6. 板端程序功能（`empty_driverlib_main.c`，v2：golden + 交互）

**阶段 1 — 开机 golden 自检（跑一遍）**：上电 0.5s → 冷启动计时 → 12 张 golden MNIST 逐张推理
（每张预热 2 次后计时 32 次）→ 打印每张结果 + 汇总（同 v1 格式；`read_serial_cnn.py` 仍可解析）。

**阶段 2 — 交互模式（常驻）**：轮询串口，收到 PC 帧即推理并回发结果。

**协议**：

| 方向 | 内容 |
|---|---|
| PC → 板 | `A5 5A` + 1024 字节 uint8 像素（32×32 行主序，白字黑底；板端 /255 → [0,1]） |
| 板 → PC | 文本 2 行：`[PC] pred=<d> conf=<f>% t_avg=<f>us t_min=<f>us` + `     logits: ...`（10 维） |

- 每次响应用 **4 次推理**取 avg/min；空闲时完全静默（省带宽，方便交互）
- **耗时口径** = `tvmgen_default_run() + while(!tvmgen_default_finished)` 的微秒数
  （含图内 CPU 胶水算子；不含打印/后处理；CPUTimer1 150MHz 递减计数）。

**PC 侧两种用法**（协议已封装进 `CNN_TWO\draw_ui.py` 的 `BoardLink`）：

```powershell
# 1) 手写板 GUI: 画完 -> "板端推理" 按钮 (右侧同时显示本地/板端结果与 NPU 耗时)
conda run --no-capture-output -n torch python E:\desk\DeepStudy\CNN_TWO\draw_ui.py
# 2) 无界面联测: 发一张 MNIST 测试图给板端
conda run --no-capture-output -n torch python E:\desk\DeepStudy\CNN_TWO\draw_ui.py --boardtest 7
```

## 7. 实测结果（2026-10-09）

| | 完整版 | 轻量版 |
|---|---|---|
| PC 测试集 acc (float) | 99.23% | 98.59% |
| int8 verify（200 样本，PC 侧） | 0.990 | 0.990 |
| 板端 golden 12 张 | **12/12 全对** | 12/12 全对 |
| 板端推理耗时 (avg, 每轮 n=384) | **10,950.6 µs** | **4,908.6 µs** |
| 板端 int8 vs PC float logits | max\|d\| = 0.591 | （跨模型比较，该行忽略） |
| mod.a 内存 | 68.63 KB | 36.12 KB |

- 冷启动 = 稳态（10,950.2 µs）；耗时抖动仅 ±8 µs（NPU 为确定性流水）
- 两点粗拟合：`T ≈ 2.6 ms + 4.3 ns × MACs` → **加速第一杠杆 = 减 MACs**（轻量版 MACs -73% → 时间 -55%）
- **交互模式实测**（PC 发帧 → 回读）：digit 7 → `pred=7` 100%、**10,908.8 µs**；digit 3（歧义样本 idx=18）→
  PC 与板端**一致判 8**（85.3% vs 86.2%）、**10,922.6 µs**。记录：`_diag\board_link_test_digit{3,7}.txt`

## 8. 切换模型版本（例：切轻量版）

```powershell
$p='E:\desk\CCS\cnn_and_number'
Copy-Item "$p\_backup_mnist_lean\mod.a"            "$p\artifacts\mod.a" -Force
Copy-Item "$p\_backup_mnist_lean\tvmgen_default.h" "$p\artifacts\tvmgen_default.h" -Force
(Get-Item "$p\artifacts\mod.a").LastWriteTime = Get-Date   # ⚠️ 防时间戳坑（见 9.4）
E:\ccs\ccs\utils\bin\gmake.exe -C "$p\CPU1_LAUNCHXL_FLASH" -j4 all
# 再执行第 4 节烧录命令（记得 -u）
```

（切回完整版同理，用 `_backup_mnist_base\`）

## 9. 坑位记录（全部 2026-10-09 实测）

1. **DSLite 必须 `-u/--run`**：否则烧完 CPU 停住，串口静默（本工程第一天踩的就是它）。
2. **GS RAM 要连续**：CNN 的 `.bss.noinit.tvm` = 25.5 KB，单块 RAMGS3（8KB）放不下 →
   `28p55x_generic_flash_lnk.cmd` 中合并 `RAMGS0..3` 为 `RAMGS0123`（32KB，物理连续；手法同 C2000Ware arc_fault 例程）。
3. **主程序 `.bss` 扩容**：输入缓冲 `g_in[1024]`（4KB）→ `.bss` 从 RAMLS5（2KB）换到 RAMLS8（8KB）。
4. **Copy-Item 时间戳坑**：换 `mod.a` 后 gmake 可能判 "up to date" 不重链（板上还是旧模型）→ 手动刷新时间戳再 build。
5. **不要调用 `TI_NPU_init()`**（新版 mod.a 无该符号，调用会链接报错；初始化在运行库内部完成）。
6. **旧固件残留特征**：上电后串口持续流出旧协议数据（如 VOFA 二进制帧）＝ 新固件没有跑起来。
7. 更换 golden 数据：在 CNN_TWO 跑 `export_golden.py`（重新生成 `golden_mnist.h`）→ gmake → 烧录。
8. **交互模式空闲静默**：v2 固件空闲时不周期输出（golden 只在开机跑一遍）——`read_serial_cnn.py`
   要在刚复位后 ~3s 内启动才能抓到 golden 段；平时用 `draw_ui.py` 的"板端推理"/`--boardtest` 交互。

## 10. 目录速查

```
artifacts\               mod.a + tvmgen_default.h   （当前 = 完整版）
_backup_mnist_base\      完整版存档（mod.a / tvmgen_default.h / cnn_and_number_base.out）
_backup_mnist_lean\      轻量版存档（mod.a / tvmgen_default.h / cnn_and_number_lean.out）
_backup_charge_mtl\      charge_mtl 脚手架时代存档（旧 mod.a / 旧主程序 / 旧 README / saw2sin 遗留）
golden_mnist.h           12 张 golden 样本（由 CNN_TWO\export_golden.py 生成）
empty_driverlib_main.c   板端主程序（golden 验证 + 计时 + 串口打印）
_diag\read_serial_cnn.py 串口文本读取工具；serial_capture_*.txt 为实测记录
28p55x_generic_flash_lnk.cmd  链接脚本（含 RAMGS0123 合并与 .bss→RAMLS8）
```
