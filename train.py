# -*- coding: utf-8 -*-
"""
训练脚本 (PC, 支持 GPU)
========================
用法 (conda 环境 torch, 必须用 conda run 见 README):
  conda run --no-capture-output -n torch python train.py               # 完整训练 (15 epoch)
  conda run --no-capture-output -n torch python train.py --epochs 3    # 指定轮数
  conda run --no-capture-output -n torch python train.py --device cpu
  conda run --no-capture-output -n torch python train.py --smoke       # 冒烟: 1 epoch + 4k 样本
产物: runs/cnn32/best.pt (最优权重) + history.json + report.json
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import CFG, cfg_to_dict
from dataset import get_loaders, random_affine_batch, random_thick_batch
from model import build_model, count_params, summary


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    n, correct = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        n += yb.numel()
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CFG.train.epochs)
    ap.add_argument("--device", type=str, default=CFG.train.device)
    ap.add_argument("--smoke", action="store_true", help="冒烟测试: 1 epoch + 4k 样本")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--channels", type=str, default=None,
                    help="覆盖 conv_channels, 如 8,16,16 (优化实验用)")
    args = ap.parse_args()

    out_dir = args.out_dir or ("runs/_smoke" if args.smoke else CFG.train.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(CFG.train.seed)

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_loaders()
    epochs = 1 if args.smoke else args.epochs
    if args.smoke:
        sub = Subset(train_loader.dataset, range(4096))
        train_loader = DataLoader(sub, batch_size=CFG.data.batch_size, shuffle=True)

    channels = tuple(int(v) for v in args.channels.split(",")) if args.channels else None
    model = build_model(channels=channels).to(device)
    n_par = count_params(model)
    summary(model)
    print(f"[info] device={device} epochs={epochs} params={n_par:,} out={out_dir}")

    opt = torch.optim.Adam(model.parameters(), lr=CFG.train.lr,
                           weight_decay=CFG.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc, best_epoch, history = -1.0, -1, []
    t_start = time.time()
    for ep in range(1, epochs + 1):
        lr_now = opt.param_groups[0]["lr"]
        model.train()
        t0 = time.time()
        loss_sum, n, correct = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            # ---- 增广 (按 batch 随机, GPU 上做, 成本可忽略) ----
            xb = random_affine_batch(xb, CFG.train.aug_rot_deg,
                                     CFG.train.aug_shift_px, CFG.train.aug_scale)
            xb = random_thick_batch(xb, CFG.train.aug_thick_p)
            # ---- 前向/反向 ----
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).sum().item()
            n += yb.numel()
        sched.step()
        test_acc = eval_acc(model, test_loader, device)
        dt = time.time() - t0
        history.append({"epoch": ep, "lr": lr_now,
                        "train_loss": loss_sum / n, "train_acc": correct / n,
                        "test_acc": test_acc, "sec": dt})
        print(f"E{ep:02d}/{epochs} | loss {loss_sum / n:.4f} | train {correct / n:.4f} "
              f"| test {test_acc:.4f} | {dt:.1f}s")
        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, ep
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "test_acc": test_acc, "cfg": cfg_to_dict()},
                       os.path.join(out_dir, "best.pt"))

    total_min = (time.time() - t_start) / 60.0
    report = {"best_test_acc": best_acc, "best_epoch": best_epoch,
              "final_test_acc": history[-1]["test_acc"],
              "params": n_par, "device": str(device), "epochs": epochs,
              "minutes": round(total_min, 2), "out_dir": out_dir}
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[done] best_acc={best_acc:.4f} (epoch {best_epoch}), {total_min:.1f} min")
    print(f"[save] {out_dir}/best.pt | history.json | report.json")


if __name__ == "__main__":
    main()
