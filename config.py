# -*- coding: utf-8 -*-
"""
全局配置 —— 唯一事实源
========================
全工程统一口径 (训练 / 评估 / 手写板 / 未来固件):
  * 输入: 32x32 单通道, float32, 取值 [0,1], "白字黑底"(笔迹=1, 背景=0)
  * 模型: 只用 Conv2d / ReLU / MaxPool2d / Linear(Flatten), 无 BN、无 Dropout
    —— 按 TI F28P55x NPU 约束设计, 依据见 docs/NPU合规设计说明.md
  * 一张手写图怎么变成 32x32 输入: 只认 preproc.py (禁止别处重写!)

想改任何超参, 只改这里; 其余文件全部从这里读, 避免"三处口径不一致"。
"""
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass
class DataCfg:
    data_dir: str = "data"
    img_size: int = 32            # 模型输入边长 (MNIST 28x28 四周 pad 2px)
    num_classes: int = 10
    batch_size: int = 128
    test_batch_size: int = 512


@dataclass
class ModelCfg:
    # 通道全部是 4 的倍数 (TI NPU m4 约束, 见 NPU_CONFIGURATION_GUIDELINES.md)
    conv_channels: Tuple[int, int, int] = (16, 32, 32)
    fc_hidden: int = 64


@dataclass
class TrainCfg:
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"          # auto | cpu | cuda
    seed: int = 0
    out_dir: str = "runs/cnn32"
    # 训练增广 (按 batch 随机, 见 dataset.py; 模拟"手写风格差异")
    aug_rot_deg: float = 10.0                     # 随机旋转 正负 10 度
    aug_shift_px: float = 2.0                     # 随机平移 正负 2 像素
    aug_scale: Tuple[float, float] = (0.9, 1.1)   # 随机缩放 +-10%
    aug_thick_p: float = 0.25                     # 笔迹加粗/变细概率


class _CFG:
    data = DataCfg()
    model = ModelCfg()
    train = TrainCfg()


CFG = _CFG()


def cfg_to_dict():
    """转成 json 可序列化的 dict (存档到权重/报告里)"""
    return {k: asdict(getattr(CFG, k)) for k in ("data", "model", "train")}
