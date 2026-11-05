"""DSL kernels used by compilers.py: a batched float32 GEMM in Triton and in TileLang.

Kept apart from compilers.py because both DSLs resolve names in kernel signatures from module
globals, which `from __future__ import annotations` would turn into unresolvable strings.
Import this module only when one of the two paths is used.
"""
from functools import lru_cache

import tilelang
import tilelang.language as T
import triton
import triton.language as tl


@lru_cache(None)
def triton_bmm_kernel():
    @triton.jit
    def bmm_kernel(A, B, C, M, N, K, k_per_split,
                   sab, sam, sak, sbb, sbk, sbn, scb, scs, scm, scn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   SPLIT_K: tl.constexpr):
        pid_m, pid_n, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        bat, ks = pid_z // SPLIT_K, pid_z % SPLIT_K
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        k_lo = ks * k_per_split
        k_hi = tl.minimum(k_lo + k_per_split, K)
        a_ptr = A + bat * sab + rm[:, None] * sam
        b_ptr = B + bat * sbb + rn[None, :] * sbn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(k_lo, k_hi, BLOCK_K):
            kk = k + rk
            a = tl.load(a_ptr + kk[None, :] * sak, mask=(rm[:, None] < M) & (kk[None, :] < k_hi),
                        other=0.0)
            b = tl.load(b_ptr + kk[:, None] * sbk, mask=(kk[:, None] < k_hi) & (rn[None, :] < N),
                        other=0.0)
            acc = tl.dot(a, b, acc, input_precision="ieee")
        c_ptr = C + bat * scb + ks * scs + rm[:, None] * scm + rn[None, :] * scn
        tl.store(c_ptr, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    return bmm_kernel


@lru_cache(None)
def tilelang_bmm_kernel(N, K, trans_b, block_m, block_n, block_k, num_stages, threads):
    Bt, M = T.dynamic("bt"), T.dynamic("m")
    b_shape = (Bt, N, K) if trans_b else (Bt, K, N)
    bs_shape = (block_n, block_k) if trans_b else (block_k, block_n)
    f32 = "float32"

    # Explicit fp32 multiply-add: T.gemm would use TF32 for float32 inputs.
    @T.prim_func
    def bmm_nn(A: T.Tensor((Bt, M, K), f32), B: T.Tensor(b_shape, f32), C: T.Tensor((Bt, M, N), f32)):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), Bt, threads=threads) as (bx, by, bz):
            As = T.alloc_shared((block_m, block_k), f32)
            Bs = T.alloc_shared(bs_shape, f32)
            Cl = T.alloc_fragment((block_m, block_n), f32)
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[bz, by * block_m, ko * block_k], As)
                T.copy(B[bz, ko * block_k, bx * block_n], Bs)
                for k in T.serial(block_k):
                    for i, j in T.Parallel(block_m, block_n):
                        Cl[i, j] += As[i, k] * Bs[k, j]
            T.copy(Cl, C[bz, by * block_m, bx * block_n])

    @T.prim_func
    def bmm_nt(A: T.Tensor((Bt, M, K), f32), B: T.Tensor(b_shape, f32), C: T.Tensor((Bt, M, N), f32)):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), Bt, threads=threads) as (bx, by, bz):
            As = T.alloc_shared((block_m, block_k), f32)
            Bs = T.alloc_shared(bs_shape, f32)
            Cl = T.alloc_fragment((block_m, block_n), f32)
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[bz, by * block_m, ko * block_k], As)
                T.copy(B[bz, bx * block_n, ko * block_k], Bs)
                for k in T.serial(block_k):
                    for i, j in T.Parallel(block_m, block_n):
                        Cl[i, j] += As[i, k] * Bs[j, k]
            T.copy(Cl, C[bz, by * block_m, bx * block_n])

    return tilelang.compile(bmm_nt if trans_b else bmm_nn, out_idx=[2], target="cuda")
