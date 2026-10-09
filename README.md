# CNN_TWO —— 手写数字识别 (32x32) PC 学习工程

一条龙：**数据集 → CNN 训练 → 评估验收 → 手写板 UI → ONNX 导出 → TI F28P55x NPU 编译验证**。
网络从设计第一天起就按 TI NPU 约束打造（首层 FCONV 通道 4 倍数、无 BN/Dropout、batch=1 静态形状）。

> 📚 **学习向文档**：`docs/01_教学篇_CNN原理与本工程拆解.md`（原理 → 逐层拆解 → 实验手册）+ `docs/02_复述题库.md`（合上资料复述给 AI 听）+ `docs/03_能力阶梯_实验分级与猜想验证.md`（实验难度分级 L1~L6 + "猜想-设计-验证"能力拆解与升段任务）。

## 关键数字（本机实测，2026-10-09）

| 项目 | 数值 |
|---|---|
| 测试集准确率 | **0.9923**（10000 张，best = epoch 14） |
| 模型规模 | 47,530 参数（≈46 KB int8 权重），1.95 MMACs |
| 训练开销 | 15 epoch / 2.0 min（RTX 4060 laptop，cuda） |
| 权重 int8 粗量化模拟 | 0.9921（≈无损） |
| 平移 +2px / 旋转 ±10° | 0.981 / 0.988~0.991 |
| 字体渲染代理测试 | **0.9933**（5 字体 × 10 数字 × 3 字号 = 150 张） |
| 手写板管线自检 | 线→1、圆→0、S 形→5 全部 100% 置信 |
| ONNX 对拍 | max\|torch−onnx\| = 9.5e-06（逐张 batch=1） |
| NPU 编译（实测） | **type=hard，全层上 NPU（0 警告）**；mod.a 68.6 KB；int8 验收 acc 0.990（详见 `docs/NPU合规设计说明.md`） |
| **板端实测（F28P550SJ9）** | **10.95 ms/次**（golden 12/12 全对；int8 vs PC float max\|d\|=0.591；轻量版实验 4.91 ms） |

## 快速开始（conda 环境 `torch`）

```powershell
# 0) 第一次跑: 下载 MNIST 并转成 32x32 [0,1] (需要代理)
$env:HTTP_PROXY='http://127.0.0.1:7890'; $env:HTTPS_PROXY='http://127.0.0.1:7890'
conda run --no-capture-output -n torch python dataset.py

# 1) 训练 (~2 min, 自动用 GPU)
conda run --no-capture-output -n torch python train.py

# 2) 验收报告 (准确率/混淆矩阵/鲁棒性/字体代理/int8模拟)
conda run --no-capture-output -n torch python evaluate.py

# 3) 手写板 UI —— 鼠标写数字, 松开即预测
conda run --no-capture-output -n torch python draw_ui.py
#    右侧"板端推理"按钮: 把当前输入经串口发给 F28P55 开发板, 回读板端 NPU 结果与耗时
#    无界面联测: conda run --no-capture-output -n torch python draw_ui.py --boardtest 7

# 4) 导出 ONNX (batch=1, opset18) + 数值对拍
conda run --no-capture-output -n torch python export_onnx.py
```

手写板快捷键：回车=预测，Esc=清空；右侧实时显示"模型看到的 32×32"和 0-9 概率条；
可勾选"松开自动预测"；"保存自绘样本"把当前笔迹存进 `data/my_draws.npz`（可攒个人数据集做微调）。
"板端推理"按钮把同一个 32×32 输入经 **COM6 串口**发给开发板（协议 `A5 5A`+1024B），
回读板端 NPU 的识别结果与推理耗时（≈10.9 ms）；串口被 VOFA 等工具占用时会提示。

## 目录结构

```
CNN_TWO/
├─ config.py        # 全局配置(唯一事实源): 尺寸/通道/超参/增广参数
├─ dataset.py       # MNIST 下载+IDX解析 -> 32x32 [0,1]; 批增广(warp/粗细)
├─ preproc.py       # 预处理唯一口径: 笔迹图 -> 32x32 (UI/评估/固件共用)
├─ model.py         # DigitCNN (NPU 合规, 见 docs/NPU合规设计说明.md)
├─ train.py         # 训练 -> runs/cnn32/best.pt + history/report.json
├─ evaluate.py      # 验收: 混淆矩阵/鲁棒性/字体代理/int8模拟 -> eval_report.json
├─ draw_ui.py       # 手写板 UI (Tkinter); --selftest 无窗口自检
├─ export_onnx.py   # 导出 ONNX(opset18, batch=1) + onnxruntime 对拍
├─ export_golden.py # 导出板端 golden 向量(12 张 MNIST) -> cnn_and_number/golden_mnist.h
├─ data/            # mnist_32.npz (缓存), preview_grid.png, my_draws.npz
├─ runs/cnn32/      # best.pt / model.onnx / eval_report.json / deploy_info.json / 图
└─ docs/
   ├─ 01_教学篇_CNN原理与本工程拆解.md   # ← 学 CNN 从这里开始（原理+拆解+实验手册）
   ├─ 02_复述题库.md                     # 复述自测（题答分离，答给 AI 听）
   ├─ 03_能力阶梯_实验分级与猜想验证.md   # 实验难度分级 + 工程直觉（猜想-设计-验证）拆解
   └─ NPU合规设计说明.md                 # 合规设计与编译实测
```

## 数据口径（全工程一致，改任何一处都要三思）

1. **来源**：MNIST 官方 28×28 → 四周补 2px 黑边 → **32×32**，像素 /255 → **float32 [0,1]**（即"0-1 格式"）。
   补边不改变笔迹粗细；数字仍装在 20px 内容盒里（MNIST 官方做派）。
2. **手写输入**（`preproc.py` 唯一实现）：包围盒裁剪 → 长边缩放到 20px → 质心居中放进 28×28 → 补边到 32×32 → /255。
   手写板、评估脚本、未来固件（C 端）必须用同一套，否则"模型看到的是没见过的分布"。
3. **白字黑底**：笔迹=1、背景=0（MNIST 惯例）。手写板显示是白底黑笔，内部渲染时统一为"墨迹亮"。

## 原理与易错点（学习向）

- **为什么 32×32、为什么补边**：明确"32x32 的 0-1 数据"这个接口；补边是最不破坏原始笔画分布的放大方式（相比拉伸 28→32）。
- **为什么无 BN**：NPU 约束 + 量化友好（少一类需要折叠/回落的算子）；MNIST 这个规模不用 BN 也能到 99%+。
- **增广为什么在 GPU 上按 batch 做**：`dataset.py` 的 `warp()` 用 `affine_grid+grid_sample` 实现旋转/平移/缩放，
  `random_thick_batch()` 用 maxpool/minpool 模拟笔尖粗细——全部走张量运算，扩充样本不花 CPU 时间。
- **易错点 1**：`torch.load` 读含配置的 checkpoint 需 `weights_only=False`（本工程已处理）。
- **易错点 2**：ONNX 是 batch=1 静态形状（NPU 要求），用 onnxruntime 对拍时要逐张喂。
- **易错点 3**：不要用别处的代码重写预处理——一切以 `preproc.py` 为准（历史经验：口径不一致 = 上板精度崩）。
- **易错点 4**：conda 环境必须用 `conda run` 方式跑（裸调 envs 里的 python.exe 会踩 MKL 崩溃坑，本机实测）。

## NPU 合规速览（详见 docs/NPU合规设计说明.md）

| 约束（TI NNC 指南） | 本模型 | 结论 |
|---|---|---|
| FCONV（首层）iC=1, oC 为 4 倍数 | Conv(1→16) | ✅ |
| 中间卷积 iC/oC 为 m4 | 16→32→32 | ✅ |
| MaxPool 至少一维 ≤4 | 2×2 | ✅ |
| FC 输入特征 ≥16 | 512 / 64 | ✅ |
| batch=1、静态形状 | (1,1,32,32) | ✅ |
| 只能用 TI-QAT（通用 PTQ 上不了 NPU） | torch2tinpu QAT 15ep 自蒸馏 | ✅ |
| 算子形态（禁用 MatMul+独立 Add） | ONNX: Conv/Relu/MaxPool/Flatten/**Gemm** | ✅ |

> 实测（2026-10-09）：`type=hard` 编译通过、**0 警告**；FCONV×1 / GCONV×2 / FC×2 / MAXPOOL×3 全部上 NPU；mod.a 68.63 KB；
> int8 验收 acc 0.990（float 0.995）。产物：`E:\desk\ti-npu\saw2sin_npu\examples\mnist_cnn\out\mnist_cnn\deploy\`。

**板端部署已完成（2026-10-09）**：工程 `E:\desk\CCS\cnn_and_number`（golden 12/12 全对、板端 int8 vs PC float max|d|=0.591、
NPU 推理 **10.95 ms**；轻量版 (8,16,16) 实验 **4.91 ms**、PC acc 98.59%）。构建/烧录/串口/切模型说明见该工程
`MNIST_CNN_README.md`；板端数据与优化分析见 `docs/NPU合规设计说明.md` §5.4。
