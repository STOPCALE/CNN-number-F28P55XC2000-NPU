# -*- coding: utf-8 -*-
"""
数据管道: MNIST -> 32x32 [0,1]
================================
* 来源: MNIST 官方镜像 (依次尝试多个 URL; 需要代理时先设 HTTP_PROXY/HTTPS_PROXY)
* 解析: 手写 IDX 格式解析 (纯 numpy + gzip, 不依赖 torchvision)
* 尺寸: 官方 28x28 -> 四周补 2px 黑边 -> 32x32;  像素 /255 -> float32 [0,1]
        (与"32x32 的 0-1 格式数据"口径一致; 补边不改变笔迹粗细分布)
* 缓存: data/mnist_32.npz (压缩) + data/preview_grid.png
* 增广 (训练时按 batch 随机):
      random_affine_batch()  旋转/平移/缩放 —— 模拟手写位置、倾斜、大小差异
      random_thick_batch()   笔迹加粗/变细 —— 模拟笔尖粗细差异
* warp() 固定变换也被 evaluate.py 复用, 保证测试口径与训练一致。
"""
import gzip
import math
import os
import struct
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from config import CFG

_BASE_URLS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",       # PyTorch 官方镜像
    "https://storage.googleapis.com/cvdf-datasets/mnist/",  # Google 原版
    "https://raw.githubusercontent.com/fgnt/mnist/master/",  # GitHub 镜像(需代理)
]
_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


# --------------------------------------------------------------- 下载 / 解析
def _download(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    for name in _FILES.values():
        dst = os.path.join(raw_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[skip] exists: {name}")
            continue
        ok = False
        for base in _BASE_URLS:
            url = base + name
            try:
                print(f"[download] {url}")
                urllib.request.urlretrieve(url, dst)
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [fail] {e}")
        if not ok:
            raise RuntimeError(
                f"download {name} failed. check network/proxy "
                f"(PowerShell: $env:HTTPS_PROXY='http://127.0.0.1:7890')")


def _parse_images(path):
    """IDX3 格式:  magic(2051) | N | H | W | uint8 像素"""
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        return np.frombuffer(f.read(), np.uint8).reshape(n, h, w)


def _parse_labels(path):
    """IDX1 格式:  magic(2049) | N | uint8 标签"""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic}"
        return np.frombuffer(f.read(), np.uint8).reshape(n)


def _save_preview(x, y, path, n=100, seed=0):
    """存一张 10x10 采样网格图, 肉眼检查数据长什么样"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), n, replace=False)
    fig, axes = plt.subplots(10, 10, figsize=(6, 6))
    for k, ax in enumerate(axes.flat):
        ax.imshow(x[idx[k]], cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(int(y[idx[k]])), fontsize=6)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_mnist_32(data_dir=None):
    """返回 (train_x, train_y, test_x, test_y)
    x: float32 [N,32,32] 取值 [0,1];  y: int64 [N]
    第一次运行会下载 + 转换 + 缓存; 以后直接读缓存。"""
    data_dir = data_dir or CFG.data.data_dir
    npz = os.path.join(data_dir, "mnist_32.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["train_x"], d["train_y"], d["test_x"], d["test_y"]

    raw = os.path.join(data_dir, "raw")
    _download(raw)
    trx = _parse_images(os.path.join(raw, _FILES["train_x"]))
    try_ = _parse_labels(os.path.join(raw, _FILES["train_y"]))
    tex = _parse_images(os.path.join(raw, _FILES["test_x"]))
    tey = _parse_labels(os.path.join(raw, _FILES["test_y"]))

    pad = (CFG.data.img_size - 28) // 2          # 32 -> 四周各补 2px
    assert pad >= 0
    trx = np.pad(trx, ((0, 0), (pad, pad), (pad, pad))).astype(np.float32) / 255.0
    tex = np.pad(tex, ((0, 0), (pad, pad), (pad, pad))).astype(np.float32) / 255.0
    try_ = try_.astype(np.int64)
    tey = tey.astype(np.int64)

    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(npz, train_x=trx, train_y=try_, test_x=tex, test_y=tey)
    _save_preview(trx, try_, os.path.join(data_dir, "preview_grid.png"))
    return trx, try_, tex, tey


# --------------------------------------------------------- 批变换 (torch, GPU 友好)
def _as_batch(v, b, device, dtype):
    t = torch.as_tensor(v, dtype=dtype, device=device)
    return t.expand(b) if t.dim() == 0 else t


def warp(x, deg=0.0, dx_px=0.0, dy_px=0.0, scale=1.0):
    """仿射变换 (旋转/平移/缩放), 批内每张图参数可不同; 边界补 0 (黑底)
    x: (B,1,H,W) 取值 [0,1];  deg: 度;  dx/dy: 像素;  scale>1 = 内容放大

    实现要点: affine_grid 生成"目标->源"采样网格 (归一化坐标, 平移量 = 像素*2/边长);
    数值上等价于把图旋转 deg 度。增广时角度取对称随机分布, 正负号方向不影响统计。
    """
    b, _, h, w = x.shape
    deg = _as_batch(deg, b, x.device, x.dtype)
    dx = _as_batch(dx_px, b, x.device, x.dtype)
    dy = _as_batch(dy_px, b, x.device, x.dtype)
    sc = _as_batch(scale, b, x.device, x.dtype)

    r = deg * (math.pi / 180.0)
    c, s = torch.cos(r), torch.sin(r)
    theta = torch.stack(
        [c, -s, dx * 2.0 / w,
         s,  c, dy * 2.0 / h], dim=1).view(b, 2, 3)
    theta[:, :2, :2] = theta[:, :2, :2] / sc.view(b, 1, 1)   # 放大内容 <=> 缩小采样范围
    grid = F.affine_grid(theta, (b, 1, h, w), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def random_affine_batch(x, max_deg=10.0, max_shift_px=2.0, scale_range=(0.9, 1.1)):
    b = x.shape[0]
    rnd = lambda lo, hi: lo + (hi - lo) * torch.rand(b, device=x.device)  # noqa: E731
    return warp(x,
                deg=rnd(-max_deg, max_deg),
                dx_px=rnd(-max_shift_px, max_shift_px),
                dy_px=rnd(-max_shift_px, max_shift_px),
                scale=rnd(scale_range[0], scale_range[1]))


def random_thick_batch(x, p=0.25):
    """笔迹粗细扰动: maxpool 近似加粗(膨胀), minpool 近似变细(腐蚀)"""
    r = torch.rand(x.shape[0], device=x.device)
    view = (-1, 1, 1, 1)
    dil = (r < p).view(*view)
    ero = ((r >= p) & (r < 2 * p)).view(*view)
    if bool(dil.any()):
        x = torch.where(dil, F.max_pool2d(x, 3, 1, 1), x)
    if bool(ero.any()):
        x = torch.where(ero, -F.max_pool2d(-x, 3, 1, 1), x)
    return x.clamp(0, 1)


# ------------------------------------------------------------------- Dataset
class Digits32(torch.utils.data.Dataset):
    """简单封装: (1,32,32) float32 [0,1] + int64 标签"""

    def __init__(self, x, y):
        self.x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_loaders():
    trx, try_, tex, tey = load_mnist_32()
    tr = Digits32(trx, try_)
    te = Digits32(tex, tey)
    tl = torch.utils.data.DataLoader(
        tr, batch_size=CFG.data.batch_size, shuffle=True, num_workers=0, drop_last=False)
    vl = torch.utils.data.DataLoader(
        te, batch_size=CFG.data.test_batch_size, shuffle=False, num_workers=0)
    return tl, vl


def main():
    trx, try_, tex, tey = load_mnist_32()
    print(f"train_x {trx.shape} {trx.dtype} range [{trx.min():.1f}, {trx.max():.1f}]")
    print(f"test_x  {tex.shape} {tex.dtype}")
    print("train class counts:", np.bincount(try_, minlength=10).tolist())
    print("preview:", os.path.join(CFG.data.data_dir, "preview_grid.png"))


if __name__ == "__main__":
    main()
