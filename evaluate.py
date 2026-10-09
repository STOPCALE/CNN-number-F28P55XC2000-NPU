# -*- coding: utf-8 -*-
"""
评估 / 鲁棒性 / 字体渲染代理测试
==================================
用法: conda run --no-capture-output -n torch python evaluate.py
产物: runs/cnn32/eval_report.json + confusion.png + errors.png + font_samples.png

内容:
  1. 测试集准确率 / 混淆矩阵 / 每类准确率
  2. 平移、旋转鲁棒性 (用与训练增广同一套 warp 实现, 口径一致)
  3. 字体渲染代理测试: 用系统字体渲染 0-9 -> 走与手写板完全相同的预处理
     (这是在"非 MNIST 分布"上的摸底; 手写板才是最终考验)
  4. 权重 int8 粗量化模拟 (正式量化由 TI-QAT 完成, 这里只估量级)
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from config import CFG
from dataset import load_mnist_32, warp
from model import build_model
from preproc import gray_to_input32

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
_FONT_TRIES = ["arial.ttf", "arialbd.ttf", "times.ttf", "georgia.ttf", "segoeui.ttf"]


def load_model(ckpt=None, device="cpu"):
    ckpt = ckpt or os.path.join(CFG.train.out_dir, "best.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    return model, ck


def predict_logits(model, x, device="cpu", bs=512):
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(model(x[i:i + bs].to(device)).cpu())
    return torch.cat(outs)


def confusion_matrix(y_true, y_pred, k=10):
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def render_digit(ch, font_path, size, canvas=320, xy=(60, 60)):
    """字符 -> "墨迹亮"灰度图 (255 on 0), 与画板内部表示一致"""
    img = Image.new("L", (canvas, canvas), 0)
    ImageDraw.Draw(img).text(xy, ch, fill=255, font=ImageFont.truetype(font_path, size))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true", help="跳过字体测试")
    args = ap.parse_args()

    out_dir = CFG.train.out_dir
    model, ck = load_model(args.ckpt, args.device)
    print(f"[load] {args.ckpt or out_dir + '/best.pt'} (epoch {ck.get('epoch')}, "
          f"test_acc {ck.get('test_acc'):.4f})")

    _, _, tex, tey = load_mnist_32()
    x = torch.from_numpy(tex).unsqueeze(1)          # (10000,1,32,32)

    # ---------------------------------------------------------- 1. 基础指标
    pred = predict_logits(model, x, args.device).argmax(1).numpy()
    acc = float((pred == tey).mean())
    cm = confusion_matrix(tey, pred)
    per_class = [float(cm[i, i] / cm[i].sum()) for i in range(10)]
    print(f"[test] acc = {acc:.4f} (n = {len(tey)})")
    print("[per-class] " + " ".join(f"{i}:{per_class[i]:.4f}" for i in range(10)))

    # ---------------------------------------------------------- 2. 鲁棒性
    shift_res = {}
    for s in (1, 2, 3, 4):
        xs = warp(x, dx_px=s, dy_px=s)
        a = float((predict_logits(model, xs, args.device).argmax(1).numpy() == tey).mean())
        shift_res[str(s)] = a
        print(f"[shift +{s}px] acc = {a:.4f}")
    rot_res = {}
    for dgr in (-20, -10, 10, 20):
        xs = warp(x, deg=dgr)
        a = float((predict_logits(model, xs, args.device).argmax(1).numpy() == tey).mean())
        rot_res[str(dgr)] = a
        print(f"[rotate {dgr:+d}deg] acc = {a:.4f}")

    # ---------------------------------------------------------- 3. 字体代理
    font_res = None
    if not args.quick:
        fonts = [os.path.join(_FONT_DIR, f) for f in _FONT_TRIES]
        fonts = [f for f in fonts if os.path.exists(f)]
        if fonts:
            imgs, labels = [], []
            for fp in fonts:
                for digit in range(10):
                    for size in (110, 140, 170):
                        gray = render_digit(str(digit), fp, size)
                        img32, _ = gray_to_input32(gray)
                        imgs.append(img32)
                        labels.append(digit)
            fx = torch.from_numpy(np.stack(imgs)).unsqueeze(1)
            fpred = predict_logits(model, fx, args.device).argmax(1).numpy()
            flabels = np.array(labels)
            facc = float((fpred == flabels).mean())
            fper = [float((fpred[flabels == i] == i).mean()) for i in range(10)]
            font_res = {"n": len(labels), "acc": facc, "per_digit": fper,
                        "fonts": [os.path.basename(f) for f in fonts]}
            print(f"[font-proxy] n = {len(labels)}, acc = {facc:.4f}  fonts={font_res['fonts']}")

            fig, axes = plt.subplots(1, 10, figsize=(15, 2))
            for i, ax in enumerate(axes):
                ax.imshow(imgs[i * 3], cmap="gray")
                ax.set_title(f"{labels[i * 3]}->{fpred[i * 3]}", fontsize=9)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "font_samples.png"), dpi=120)
            plt.close(fig)
        else:
            print("[font-proxy] 没找到系统字体, 跳过")

    # ---------------------------------------------------------- 4. int8 模拟
    m2 = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for p in m2.parameters():
            s = p.abs().max() / 127.0
            if float(s) > 0:
                p.copy_(torch.round(p / s).clamp(-127, 127) * s)
    p2 = predict_logits(m2, x).argmax(1).numpy()
    sim_acc = float((p2 == tey).mean())
    print(f"[int8-sim] 权重粗量化后 acc = {sim_acc:.4f}")

    # ---------------------------------------------------------- 保存
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_title(f"confusion matrix (acc={acc:.4f})")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            v = int(cm[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                        color="white" if v > cm.max() * 0.5 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion.png"), dpi=130)
    plt.close(fig)

    wrong = np.where(pred != tey)[0][:25]
    fig, axes = plt.subplots(5, 5, figsize=(7, 7))
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < len(wrong):
            i = int(wrong[k])
            ax.imshow(tex[i], cmap="gray")
            ax.set_title(f"{tey[i]}->{pred[i]}", fontsize=9, color="red")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "errors.png"), dpi=120)
    plt.close(fig)

    report = {"test_acc": acc, "n_test": int(len(tey)),
              "per_class_acc": per_class, "confusion": cm.tolist(),
              "shift_px": shift_res, "rot_deg": rot_res,
              "font_test": font_res, "int8_weight_sim_acc": sim_acc,
              "ckpt": args.ckpt or os.path.join(out_dir, "best.pt")}
    with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[save] eval_report.json / confusion.png / errors.png / font_samples.png")


if __name__ == "__main__":
    main()
