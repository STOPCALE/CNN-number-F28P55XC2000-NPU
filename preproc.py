# -*- coding: utf-8 -*-
"""
预处理唯一口径 —— 手写板 / 评估 / 未来固件三处共用, 禁止重写!
=============================================================
把"画板笔迹"或任意灰度笔迹图 -> 模型输入 32x32 float32 [0,1]:

  1) 取墨迹包围盒 (裁掉四周空白)
  2) 长边等比缩放到 20px 内容盒 (MNIST 官方做法: 数字装进 20x20)
  3) 放进 28x28 画布, 再按"墨迹质心"对齐到画布中心 (14,14)
  4) 四周补 2px 黑边 -> 32x32;  像素 /255 -> [0,1]

为什么这么做: 训练数据 (MNIST) 就是"20 内容盒 + 质心居中"的做派;
手写输入必须走同一条流水线, 否则模型看到的是"没见过的分布"。
"""
import numpy as np
from PIL import Image, ImageDraw

THRESH = 20          # 墨迹灰度阈值 (0-255)


def strokes_to_gray(strokes, canvas=320, brush=24):
    """笔迹列表 [[(x,y), ...], ...] -> 灰度图 (uint8, 墨迹亮=255, 背景=0)
    画板内部统一用"墨迹亮"表示 (与 MNIST 白字黑底一致),
    所以传进来之前, 白底黑笔的显示层要自行翻转 (见 draw_ui.py)。"""
    img = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(img)
    r = max(1, brush // 2)
    for st in strokes:
        if len(st) == 1:
            x, y = st[0]
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        elif len(st) >= 2:
            d.line([(float(x), float(y)) for x, y in st],
                   fill=255, width=brush, joint="curve")
    return np.asarray(img)


def _shift(a, dy, dx):
    """平移 (不环绕), 空出区域为 0"""
    h, w = a.shape
    out = np.zeros_like(a)
    sy0, sy1 = max(0, -dy), h - max(0, dy)
    sx0, sx1 = max(0, -dx), w - max(0, dx)
    dy0, dx0 = max(0, dy), max(0, dx)
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = a[sy0:sy1, sx0:sx1]
    return out


def gray_to_input32(gray):
    """灰度笔迹图 (墨迹亮) -> (img32 (32,32) float32 [0,1], meta dict)
    meta: empty(bool), preview32(uint8), preview28(uint8)
    空笔迹 -> 全 0 图 + meta['empty']=True"""
    g = np.asarray(gray, dtype=np.uint8)
    mask = g > THRESH
    if not mask.any():
        return np.zeros((32, 32), np.float32), {
            "empty": True, "preview32": np.zeros((32, 32), np.uint8)}

    ys, xs = np.where(mask)
    crop = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 1) 长边缩放进 20px 内容盒 (LANCZOS 抗锯齿, 和 MNIST 的灰度过渡一致)
    h, w = crop.shape
    k = 20.0 / max(h, w)
    nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
    small = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.Resampling.LANCZOS))

    # 2) 放进 28x28
    canvas28 = np.zeros((28, 28), np.uint8)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas28[top:top + nh, left:left + nw] = small

    # 3) 质心对齐: 把墨迹质心平移到 (14,14)
    m = canvas28 > THRESH
    if m.any():
        cy, cx = np.argwhere(m).mean(axis=0)
        dy, dx = int(round(14 - cy)), int(round(14 - cx))
        if dy or dx:
            canvas28 = _shift(canvas28, dy, dx)

    # 4) 补边到 32x32 + /255
    img32 = np.zeros((32, 32), np.uint8)
    img32[2:30, 2:30] = canvas28
    return img32.astype(np.float32) / 255.0, {
        "empty": False, "preview32": img32, "preview28": canvas28}
