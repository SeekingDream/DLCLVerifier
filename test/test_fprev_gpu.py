"""Reveal the K-accumulation order of the GPU GEMM kernels with FPRev and check the orders the
certifier assumes for them.

The certifier treats the Triton and TileLang kernels in kernels.py as a sequential FMA chain over
k (from a zero accumulator). This script reveals each kernel's order at several K, checks it is
that chain, checks determinism and bit-exact emulation (collect_evidence), and probes FMA use.
cuBLAS is probed too; its order is reported, not assumed.

    CUDA_VISIBLE_DEVICES=7 python test/test_fprev_gpu.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import compilers  # noqa: E402
from dlclv import fpmodel as fpm  # noqa: E402
from dlclv.fprev import RevealError, collect_evidence, detect_fma, matmul_sum_fn  # noqa: E402


def canon(t):
    return t if isinstance(t, int) else frozenset(canon(c) for c in t)


def main():
    compilers.strict_fp32()
    kernels = {
        "triton": lambda A, B: compilers.triton_bmm(A[None], B[None])[0],
        "tilelang": lambda A, B: compilers.tilelang_bmm(A[None], B[None])[0],
        "cublas": lambda A, B: A @ B,
    }
    fails = 0
    for name, mm in kernels.items():
        for K in (16, 48, 96):
            fn = matmul_sum_fn(mm, K, M=4, N=4, row=1, col=2)
            try:
                ev = collect_evidence(fn, K, kernel=f"{name} K={K}", trials=25, runs=3)
            except RevealError as e:
                print(f"{name:9s} K={K:3d}: order not established ({e})")
                if name != "cublas":
                    fails += 1
                continue
            seq = fpm.SumTree.sequential(K, zero_init=False)
            is_seq = canon(ev.tree.to_nested()) == canon(seq.to_nested())
            depth = _depth(ev.tree)
            print(f"{name:9s} K={K:3d}: validated tree, sequential={is_seq}, depth={depth}, "
                  f"nodes={len(ev.tree.children)}")
            if name != "cublas" and not is_seq:
                fails += 1
        fma = detect_fma(mm)
        print(f"{name:9s} FMA probe: {fma}")
        if name != "cublas" and fma != "fma":
            fails += 1
    print("SUMMARY", "PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


def _depth(t):
    total = t.n_total
    d = {}
    for k, ch in enumerate(t.children):
        d[total + k] = 1 + max(d.get(c, 0) for c in ch)
    return max(d.values()) if d else 0


if __name__ == "__main__":
    main()
