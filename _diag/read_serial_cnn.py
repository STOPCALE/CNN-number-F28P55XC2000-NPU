# -*- coding: utf-8 -*-
"""
read_serial_cnn.py -- 读取 cnn_and_number 板端文本输出 (SCIA 115200)
====================================================================
用法 (torch 环境已装 pyserial):
  E:\\anaconda\\envs\\torch\\python.exe read_serial_cnn.py --seconds 8 --port COM6

[WARN] 串口独占: 运行前请关闭 VOFA+/串口助手等占用串口的工具!
输出: 控制台摘要 + 原始文本存至本目录 serial_cnn_capture.txt
"""
import argparse
import os
import re
import sys
import time


def main():
    ap = argparse.ArgumentParser(description="cnn_and_number serial (text) reader")
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    import serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except Exception as e:  # noqa: BLE001
        print("[FAIL] cannot open %s: %s" % (args.port, e))
        print("       -> close other serial tools (VOFA etc.) first")
        sys.exit(2)

    print("[..] reading %s @ %d for %.1fs ..." % (args.port, args.baud, args.seconds))
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < args.seconds:
        buf += ser.read(8192)
    ser.close()

    text = bytes(buf).decode("ascii", errors="replace")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "serial_cnn_capture.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    runs = len(re.findall(r"==== RUN #", text))
    oks = len(re.findall(r"OK \| t_avg", text))
    mis = len(re.findall(r"MISMATCH", text))
    sums = re.findall(r"timing n=(\d+): min=([\d.]+)us avg=([\d.]+)us max=([\d.]+)us", text)
    colds = re.findall(r"cold-start first inference: ([\d.]+) us", text)
    gm = re.findall(r"max\|d\|=([\d.]+)", text)

    print("[OK] %d bytes -> %s" % (len(buf), out))
    print("runs=%d  sample-OK lines=%d  mismatch=%d" % (runs, oks, mis))
    for s in sums:
        print("timing n=%s min=%s avg=%s max=%s us" % s)
    if colds:
        print("cold-start: %s us" % colds[0])
    if gm:
        print("board-vs-PC max|d|: %s" % gm[0])


if __name__ == "__main__":
    main()
