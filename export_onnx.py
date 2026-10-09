# -*- coding: utf-8 -*-
"""
导出 ONNX (给 TI F28P55x NPU 工具链 torch2tinpu 用)
====================================================
用法: conda run --no-capture-output -n torch python export_onnx.py
产物: runs/cnn32/model.onnx (opset18, batch=1) + deploy_info.json
      (+ onnxruntime 数值对拍: torch 与 onnx 是否逐位一致)
上 NPU 的下一步: torch2tinpu 工具 (ti-npu 环境), 见
      E:\\desk\\ti-npu\\saw2sin_npu\\README.md 与 examples\\mnist_cnn
"""
import json
import os

import numpy as np
import torch

from config import CFG, cfg_to_dict
from dataset import load_mnist_32
from model import build_model, count_params


def main():
    out_dir = CFG.train.out_dir
    ck = torch.load(os.path.join(out_dir, "best.pt"),
                    map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ck["model"])
    model.eval()

    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        model, torch.zeros(1, 1, 32, 32), onnx_path, opset_version=18,
        input_names=["input"], output_names=["output"],
        do_constant_folding=True)
    print(f"[export] {onnx_path}")

    # 图里都有什么算子 (上 NPU 前先自查形态: 有 MatMul 就要警惕, 见 NPU 文档)
    import onnx
    m = onnx.load(onnx_path)
    ops = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print("[ops]", ops)

    # 数值对拍: torch vs onnxruntime
    diff = None
    acc256 = None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        _, _, tex, tey = load_mnist_32()
        n = 256
        x = torch.from_numpy(tex[:n]).unsqueeze(1)
        with torch.no_grad():
            a = model(x).numpy()
        # 注意: 导出的 ONNX 是 batch=1 静态形状 (NPU 要求), 对拍要逐张喂
        b = np.concatenate([sess.run(None, {"input": x[i:i + 1].numpy()})[0]
                            for i in range(n)], axis=0)
        diff = float(np.abs(a - b).max())
        acc256 = float((b.argmax(1) == tey[:n]).mean())
        print(f"[onnx] batch1 vs torch max|diff| = {diff:.2e}, acc256 = {acc256:.4f}")
    except ImportError as e:  # noqa: BLE001
        print("[onnx] skip parity:", e)

    info = {
        "onnx": onnx_path,
        "input": {"shape": [1, 1, 32, 32], "layout": "NCHW", "dtype": "float32",
                  "value_range": [0.0, 1.0],
                  "note": "白字黑底; 手写输入预处理见 preproc.py"},
        "output": {"shape": [1, 10], "note": "logits; 上板时 softmax/argmax 放 C 端"},
        "params": count_params(model),
        "onnx_ops": ops,
        "torch_vs_onnx_max_abs_diff": diff,
        "npu_next_step": r"E:\desk\ti-npu\saw2sin_npu\examples\mnist_cnn (torch2tinpu)",
        "cfg": cfg_to_dict(),
    }
    with open(os.path.join(out_dir, "deploy_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("[save] deploy_info.json")


if __name__ == "__main__":
    main()
