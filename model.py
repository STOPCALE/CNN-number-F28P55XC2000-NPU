# -*- coding: utf-8 -*-
"""
CNN 网络 —— 按 TI F28P55x NPU 约束设计
========================================
结构 (输入 32x32 单通道):
    Conv(1->16, 3x3, pad1) + ReLU + MaxPool2x2   -> 16 x 16 x 16
    Conv(16->32, 3x3, pad1) + ReLU + MaxPool2x2  -> 32 x 8 x 8
    Conv(32->32, 3x3, pad1) + ReLU + MaxPool2x2  -> 32 x 4 x 4
    Flatten 512  ->  Linear(512->64) + ReLU  ->  Linear(64->10)

NPU 合规要点 (依据 tinyml-modelzoo/docs/NPU_CONFIGURATION_GUIDELINES.md):
  1. 首层 iC=1 + oC=16           -> 走 FCONV, oC 必须 4 的倍数 [OK]
  2. 中间层 iC/oC = 16/32/32     -> 全部 4 的倍数 (m4) [OK]
  3. MaxPool 2x2                 -> 至少一维 <=4 [OK]
  4. FC 输入特征 512 / 64        -> 都 >=16 [OK]
  5. 算子只用 Conv/ReLU/MaxPool/Linear, 无 BN、无 Dropout、无自定义算子 [OK]
  6. batch=1 导出、输入静态 32x32x1 (训练 batch 与导出无关)
  上板时需要把 Linear 改写为等价的 1x1 卷积形态 (见 NPU适配说明), 数学等价。
"""
import torch
import torch.nn as nn

from config import CFG


class DigitCNN(nn.Module):
    def __init__(self, channels=(16, 32, 32), fc_hidden=64, num_classes=10):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, fc_hidden), nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(channels=None, fc_hidden=None):
    return DigitCNN(channels or CFG.model.conv_channels,
                    fc_hidden or CFG.model.fc_hidden,
                    CFG.data.num_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def summary(model=None, input_shape=(1, 1, 32, 32)):
    """打印逐层输出形状 / 参数量 / MACs 估算 (评估 NPU 算力与内存量级用)"""
    model = model or build_model()
    model.eval()
    rows = []

    def hook(mod, inp, out):
        macs = 0
        if isinstance(mod, nn.Conv2d):
            macs = (out.shape[1] * out.shape[2] * out.shape[3]
                    * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1])
        elif isinstance(mod, nn.Linear):
            macs = mod.in_features * mod.out_features
        rows.append((type(mod).__name__, tuple(out.shape[1:]),
                     sum(p.numel() for p in mod.parameters()), macs))

    dev = next(model.parameters()).device
    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    model(torch.zeros(*input_shape, device=dev))
    for h in handles:
        h.remove()

    total_p = sum(r[2] for r in rows)
    total_m = sum(r[3] for r in rows)
    print(f"{'layer':<10}{'output':<16}{'params':>8}{'MACs':>10}")
    for name, shape, p, macs in rows:
        print(f"{name:<10}{str(shape):<16}{p:>8}{macs:>10}")
    print(f"total: {total_p:,} params (int8 weights ~ {total_p / 1024:.1f} KB), "
          f"{total_m / 1e6:.2f} MMACs")
    return {"params": total_p, "macs": total_m, "layers": rows}


if __name__ == "__main__":
    summary()
