# -*- coding: utf-8 -*-
"""
手写数字验证 UI (Tkinter 画板)
================================
用法:
  conda run --no-capture-output -n torch python draw_ui.py                  # 打开画板
  conda run --no-capture-output -n torch python draw_ui.py --selftest       # 无窗口自检
  conda run --no-capture-output -n torch python draw_ui.py --boardtest 7    # 串口板端联测(无界面)
玩法: 白底黑笔写数字 -> 松开鼠标自动预测 (可关) -> 右侧显示"模型看到的 32x32"
      与 0-9 概率条; 回车=预测, Esc=清空; 可"保存自绘样本"攒个人数据集。

板端串口推理 (F28P55 NPU):
  点"板端推理"-> 把当前 32x32 输入经串口发给开发板 -> 回读板端 NPU 结果与耗时。
  协议: PC->板  `A5 5A` + 1024 字节 uint8 像素 (板端 /255 还原 [0,1]);
        板->PC  文本 `[PC] pred=.. conf=..% t_avg=..us t_min=..us` + 一行 10 维 logits。
  注意: 串口独占 —— 使用前关闭 VOFA+/串口助手; 板端开机自检期间(~2s)不响应。
"""
import argparse
import os

import numpy as np
import torch

CANVAS = 320          # 画板边长 (像素)
BRUSH = 24            # 笔刷宽度 (像素); 缩到 32x32 后约 2px, 接近 MNIST 笔迹粗细
ZOOM = 6              # 32x32 预览放大倍数


def load_model():
    from config import CFG
    from model import build_model
    path = os.path.join(CFG.train.out_dir, "best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}, 请先运行 train.py 训练")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def predict_probs(model, img32):
    x = torch.from_numpy(img32)[None, None]        # (1,1,32,32)
    with torch.no_grad():
        logits = model(x)
    return torch.softmax(logits, dim=1)[0].numpy()


BOARD_MAGIC = b"\xA5\x5A"


class BoardLink:
    """板端串口链路 (cnn_and_number 固件协议)

    PC -> 板: 0xA5 0x5A + 1024 字节 uint8 像素 (32x32, 白字黑底)
    板 -> PC: 文本行 `[PC] pred=.. conf=..% t_avg=..us t_min=..us` + logits 行
    """

    def __init__(self, port="COM6", baud=115200):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.2)

    def close(self):
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass

    def infer(self, pix_u8, timeout=4.0):
        """发送一帧并等待回读; 返回 dict 或 None(超时)"""
        import re
        import time

        self.ser.reset_input_buffer()
        self.ser.write(BOARD_MAGIC + np.asarray(pix_u8, np.uint8).tobytes())
        buf = bytearray()
        t0 = time.time()
        while time.time() - t0 < timeout:
            buf += self.ser.read(512)
            if b"logits:" in buf and b"\r\n" in buf.split(b"logits:", 1)[1]:
                break
        text = buf.decode("ascii", errors="replace")
        m = re.search(r"\[PC\] pred=(\d+)\s+conf=([\d.]+)%\s+t_avg=([\d.]+)us\s+t_min=([\d.]+)us", text)
        if not m:
            return None
        lg = re.search(r"logits:\s*([-0-9.\s]+)", text)
        logits = [float(v) for v in lg.group(1).split()] if lg else []
        return {"pred": int(m.group(1)), "conf": float(m.group(2)),
                "t_avg": float(m.group(3)), "t_min": float(m.group(4)),
                "logits": logits}


def run_gui(port="COM6"):
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    from config import CFG
    from preproc import gray_to_input32, strokes_to_gray

    model, ck = load_model()

    class App:
        def __init__(self, root):
            self.root = root
            root.title(f"手写数字验证 - CNN 32x32 (test_acc={ck.get('test_acc', float('nan')):.4f})")
            root.resizable(False, False)

            left = tk.Frame(root)
            left.pack(side="left", padx=10, pady=10)
            right = tk.Frame(root)
            right.pack(side="left", padx=4, pady=10, anchor="n")

            self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="white",
                                    cursor="pencil", highlightthickness=1,
                                    highlightbackground="#999")
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.on_press)
            self.canvas.bind("<B1-Motion>", self.on_move)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)

            tk.Label(right, text="模型看到的输入 (32x32, 白=笔迹)",
                     font=("Microsoft YaHei", 9)).pack(anchor="w")
            self.preview = tk.Canvas(right, width=32 * ZOOM, height=32 * ZOOM, bg="black",
                                     highlightthickness=1, highlightbackground="#999")
            self.preview.pack(pady=(2, 8))

            tk.Label(right, text="0-9 概率", font=("Microsoft YaHei", 9)).pack(anchor="w")
            self.prob_canvas = tk.Canvas(right, width=310, height=10 * 22 + 4,
                                         highlightthickness=0)
            self.prob_canvas.pack(pady=(2, 8))

            btns = tk.Frame(right)
            btns.pack(fill="x")
            tk.Button(btns, text="预测 (回车)", command=self.predict_now).pack(side="left")
            tk.Button(btns, text="清空 (Esc)", command=self.clear).pack(side="left", padx=6)
            self.auto = tk.BooleanVar(value=True)
            tk.Checkbutton(btns, text="松开自动预测", variable=self.auto).pack(side="left")

            sp = tk.Frame(right)
            sp.pack(fill="x", pady=(6, 0))
            tk.Label(sp, text="样本标签:").pack(side="left")
            self.label_var = tk.StringVar(value="0")
            ttk.Combobox(sp, textvariable=self.label_var, width=3, state="readonly",
                         values=[str(i) for i in range(10)]).pack(side="left")
            tk.Button(sp, text="保存自绘样本", command=self.save_sample).pack(side="left", padx=6)

            bd = tk.Frame(right)
            bd.pack(fill="x", pady=(6, 0))
            tk.Label(bd, text="板端(串口):").pack(side="left")
            self.port_var = tk.StringVar(value=port)
            ttk.Combobox(bd, textvariable=self.port_var, width=6,
                         values=["COM3", "COM4", "COM5", "COM6", "COM7", "COM8"]).pack(side="left")
            tk.Button(bd, text="板端推理", command=self.board_infer).pack(side="left", padx=6)
            self.board_status = tk.Label(right, text="板端: 未连接 (画完点\"板端推理\")",
                                         anchor="w", justify="left", wraplength=310,
                                         font=("Microsoft YaHei", 9), fg="#555")
            self.board_status.pack(fill="x", pady=(4, 0))

            self.status = tk.Label(right, text="写一个数字, 松开鼠标即预测", anchor="w",
                                   justify="left", wraplength=310,
                                   font=("Microsoft YaHei", 10))
            self.status.pack(fill="x", pady=(8, 0))

            root.bind("<Return>", lambda e: self.predict_now())
            root.bind("<Escape>", lambda e: self.clear())
            root.protocol("WM_DELETE_WINDOW", self._on_close)

            self.strokes = []          # [[(x,y),...], ...] 相当于矢量墨迹
            self.cur = None
            self._imgref = None        # 必须持有 PhotoImage 引用, 否则预览变白
            self._last_img32 = None
            self._last_u8 = None       # 板端串口发送用 (uint8 32x32)
            self._last_top = None      # 本地预测结果 (与板端对照)
            self.board = None          # BoardLink (懒加载)

        # -------------------------------------------------------- 鼠标事件
        def on_press(self, e):
            self.cur = [(e.x, e.y)]
            self.strokes.append(self.cur)
            r = BRUSH // 2
            self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r,
                                    fill="black", outline="black")

        def on_move(self, e):
            if self.cur is None:
                return
            x0, y0 = self.cur[-1]
            self.cur.append((e.x, e.y))
            self.canvas.create_line(x0, y0, e.x, e.y, fill="black", width=BRUSH,
                                    capstyle="round", joinstyle="round", smooth=True)

        def on_release(self, e):
            self.cur = None
            if self.auto.get():
                self.predict_now()

        # -------------------------------------------------------- 核心
        def predict_now(self):
            if not self.strokes:
                self.status.config(text="画板是空的, 先写一个数字吧")
                return
            # 显示层(白底黑笔) -> 统一内部表示(墨迹亮) -> preproc 唯一口径
            gray = strokes_to_gray(self.strokes, CANVAS, BRUSH)
            img32, meta = gray_to_input32(gray)
            self._last_img32 = img32
            self._last_u8 = meta["preview32"]      # 板端串口发送用 (uint8 32x32)
            self._show_preview(img32)
            if meta["empty"]:
                self.status.config(text="没检测到笔迹")
                return
            probs = predict_probs(model, img32)
            self._show_probs(probs)
            order = np.argsort(probs)[::-1]
            top = int(order[0])
            self._last_top = top
            self.status.config(
                text=(f"预测: {top}   概率 {probs[top]:.1%}\n"
                      + "  ".join(f"{int(i)}:{probs[i]:.1%}" for i in order[1:4])))

        def _show_preview(self, img32):
            arr = (img32 * 255).astype(np.uint8)
            im = Image.fromarray(arr).resize((32 * ZOOM, 32 * ZOOM), Image.NEAREST)
            self._imgref = ImageTk.PhotoImage(im)
            self.preview.delete("all")
            self.preview.create_image(0, 0, anchor="nw", image=self._imgref)

        def _show_probs(self, probs):
            c = self.prob_canvas
            c.delete("all")
            top = int(np.argmax(probs))
            for i in range(10):
                y = i * 22 + 2
                c.create_text(12, y + 9, text=str(i), font=("Consolas", 11))
                c.create_rectangle(26, y, 246, y + 18, outline="#cccccc", fill="#f5f5f5")
                w = int(probs[i] * 220)
                if w > 0:
                    c.create_rectangle(26, y, 26 + w, y + 18, outline="",
                                       fill="#2f7fd0" if i == top else "#9db8d6")
                c.create_text(252, y + 9, anchor="w", text=f"{probs[i]:.3f}",
                              font=("Consolas", 10))

        def clear(self):
            self.strokes = []
            self.cur = None
            self.canvas.delete("all")
            self.preview.delete("all")
            self.prob_canvas.delete("all")
            self._last_img32 = None
            self.status.config(text="已清空, 重新写一个数字")

        def save_sample(self):
            if self._last_img32 is None:
                self.status.config(text="先写一个数字并预测, 再保存")
                return
            path = os.path.join(CFG.data.data_dir, "my_draws.npz")
            os.makedirs(CFG.data.data_dir, exist_ok=True)
            label = int(self.label_var.get())
            if os.path.exists(path):
                d = np.load(path)
                xs, ys = list(d["x"]), list(d["y"])
            else:
                xs, ys = [], []
            xs.append(self._last_img32)
            ys.append(label)
            np.savez_compressed(path, x=np.stack(xs),
                                y=np.array(ys, np.int64))
            self.status.config(text=f"已保存自绘样本: 共 {len(xs)} 张 -> {path}")

        # -------------------------------------------------------- 板端串口
        def board_infer(self):
            if self._last_u8 is None or int(self._last_u8.max()) == 0:
                self.board_status.config(text="板端: 先画一个数字(并预测)再发送")
                return
            try:
                if self.board is None:
                    self.board = BoardLink(self.port_var.get())
            except Exception as e:  # noqa: BLE001
                self.board = None
                self.board_status.config(text=f"串口打开失败: {e} (VOFA/串口助手占用?)")
                return
            self.board_status.config(text="板端: 已发送, 等待回读...")
            self.root.update_idletasks()
            try:
                res = self.board.infer(self._last_u8)
            except Exception as e:  # noqa: BLE001
                self.board_status.config(text=f"串口通信错误: {e}")
                return
            if res is None:
                self.board_status.config(text="板端无响应 (可能正在开机自检, 稍后重试)")
                return
            match = "一致 ✓" if res["pred"] == self._last_top else "不一致 ⚠"
            self.board_status.config(
                text=(f"板端: {res['pred']} ({res['conf']:.1f}%) | NPU {res['t_avg'] / 1000:.2f} ms "
                      f"(min {res['t_min'] / 1000:.2f} ms) | 本地 {self._last_top} {match}"))

        def _on_close(self):
            if self.board is not None:
                self.board.close()
            self.root.destroy()

    root = tk.Tk()
    App(root)
    root.mainloop()


def selftest():
    """无窗口自检: 合成笔迹 -> 预处理 -> 推理, 验证整条管线"""
    import math

    from preproc import gray_to_input32, strokes_to_gray

    model, ck = load_model()
    print(f"[selftest] loaded best.pt (test_acc={ck.get('test_acc', float('nan')):.4f})")

    cases = {
        "vertical-line(1)": [[(160, 60), (160, 260)]],
        "circle(0)": [[(int(160 + 90 * math.cos(t)), int(160 + 90 * math.sin(t)))
                       for t in np.linspace(0.35, 2 * math.pi - 0.35, 40)]],
        "s-shape(5)": [[(205, 75), (140, 75), (130, 115), (185, 140),
                        (205, 185), (150, 240), (105, 225)]],
    }
    for name, strokes in cases.items():
        gray = strokes_to_gray(strokes)
        img32, meta = gray_to_input32(gray)
        assert img32.shape == (32, 32) and img32.dtype == np.float32
        assert 0.0 <= float(img32.min()) and float(img32.max()) <= 1.0
        probs = predict_probs(model, img32)
        order = np.argsort(probs)[::-1]
        print(f"[{name}] pred={int(order[0])} ({probs[order[0]]:.1%}) "
              f"top3={[(int(i), round(float(probs[i]), 3)) for i in order[:3]]}")
    print("[selftest] pipeline OK")


def boardtest(digit=7, port="COM6"):
    """无界面联测: 取一张 MNIST 测试图 -> 本地推理 -> 串口发板 -> 回读板端结果与耗时"""
    from config import CFG

    model, _ = load_model()
    d = np.load(os.path.join(CFG.data.data_dir, "mnist_32.npz"))
    tex, tey = d["test_x"], d["test_y"]
    idx = int(np.where(tey == digit)[0][0])
    img32 = tex[idx]
    pix_u8 = np.clip(np.round(img32 * 255.0), 0, 255).astype(np.uint8)

    probs = predict_probs(model, img32)
    local = int(probs.argmax())
    print(f"[local] test digit={digit} idx={idx} -> PC pred={local} ({probs[local]:.1%})")

    try:
        link = BoardLink(port)
    except Exception as e:  # noqa: BLE001
        print(f"[board] 串口打开失败: {e} (VOFA/占用?)")
        return 1
    res = link.infer(pix_u8)
    link.close()
    if res is None:
        print("[board] 无响应 (开机自检中? VOFA 占用串口?)")
        return 1
    print(f"[board] pred={res['pred']} conf={res['conf']:.1f}% "
          f"t_avg={res['t_avg']:.1f}us t_min={res['t_min']:.1f}us "
          f"({res['t_avg'] / 1000:.2f} ms)")
    print(f"[board] logits: {['%.3f' % v for v in res['logits']]}")
    print(f"[match] {'OK' if res['pred'] == local else 'MISMATCH'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="无窗口自检")
    ap.add_argument("--boardtest", type=int, default=None, metavar="DIGIT",
                    help="串口联测: 发一张该数字的 MNIST 测试图给板端 (如 --boardtest 7)")
    ap.add_argument("--port", type=str, default="COM6", help="板端串口 (默认 COM6)")
    args = ap.parse_args()
    if args.boardtest is not None:
        raise SystemExit(boardtest(args.boardtest, args.port))
    if args.selftest:
        selftest()
    else:
        run_gui(port=args.port)


if __name__ == "__main__":
    main()
