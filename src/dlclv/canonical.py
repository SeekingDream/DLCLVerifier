"""Lower a source graph to a canonical graph of logical operations.

Matching (`matching.py`) compares canonical graphs, so two programs that compute the same thing
in real arithmetic should lower to the same canonical structure even when one of them fuses,
splits, or reorders operations. Lowering therefore

1. maps every source op (ATen, ONNX, ONNX Runtime contrib kernels) to logical ops, splitting fused
   kernels into their parts (`FusedConv` -> conv, add, relu);
2. applies only rewrites that are exact in real arithmetic:
   * no-op removal (identity, dropout in eval, reshape to the same shape, ...);
   * folding of exact layout ops on constants (reshape, transpose, slice, concat, gather, cast);
   * reshape / transpose / slice chain composition;
   * broadcast elision into elementwise consumers;
   * `x - y -> x + (-y)`, `x / c -> x * (1/c)` with an exact rational `1/c`;
   * n-ary `add` and `mul` (associativity and commutativity), with scalar factors multiplied
     exactly and scalar factors moved out of `matmul` (linearity);
   * recognition of the erf and tanh GELU expressions as `gelu` (their constants are compared
     with a relative tolerance of 1e-6, recorded in `meta["approximate_constants"]`).

Every canonical node records in `meta["origins"]` the source nodes it stands for. Source ops
without a lowering rule become `opaque` nodes, which only match the identical source op.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Dict, List, Optional

import numpy as np

from .frontends import aten_name
from .graph import GraphError, Node, TensorGraph, Value

COMMUTATIVE = {"add", "mul", "max", "min", "eq", "ne", "and", "or"}
NARY = {"add", "mul"}
BINARY = {"add", "sub", "mul", "div", "pow", "max", "min", "eq", "ne", "lt", "le", "gt", "ge",
          "and", "or"}
COMPARE = {"eq", "ne", "lt", "le", "gt", "ge", "and", "or"}
UNARY = {"relu", "tanh", "exp", "log", "sqrt", "rsqrt", "neg", "sigmoid", "erf", "abs",
         "reciprocal", "sin", "cos", "silu", "not"}
ELEMENTWISE = BINARY | UNARY | {"where", "gelu", "clamp"}
EXACT_FOLD = {"reshape", "permute", "broadcast_to", "slice", "concat", "cast", "gather", "neg",
              "where", "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"}


class LoweringError(GraphError):
    pass


def _f32(x) -> float:
    return float(np.float32(x))


def exact_scalar(v: Value) -> Optional[Fraction]:
    """Exact value of a scalar constant, or None."""
    if v.kind != "const" or v.numel != 1:
        return None
    if v.exact is not None:
        return v.exact
    x = v.data.reshape(()).item()
    if isinstance(x, (bool, np.bool_)):
        return Fraction(int(x))
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return Fraction(x)


# --------------------------------------------------------------------------------------------
# Lowering context
# --------------------------------------------------------------------------------------------

class Lowerer:
    def __init__(self, src: TensorGraph):
        self.src = src
        self.k = TensorGraph(src.name, "canonical")
        self.k.meta.update(src.meta)
        self.vmap: Dict[str, str] = {}
        self.node: Optional[Node] = None
        for vid in src.inputs:
            v = src.values[vid]
            self.k.add_value(Value(vid, v.shape, v.dtype, "input", input_index=v.input_index,
                                   name=v.name, origin=vid))
            self.vmap[vid] = vid

    # ---- values ----

    def val(self, src_vid: Optional[str]) -> Optional[str]:
        if src_vid is None:
            return None
        if src_vid not in self.vmap:
            v = self.src.values[src_vid]
            if v.kind != "const":
                raise LoweringError(f"value {src_vid} used before it was lowered")
            self.vmap[src_vid] = self.k.add_const(v.data, name=v.name, vid=src_vid, origin=src_vid)
        return self.vmap[src_vid]

    def V(self, cid) -> Value:
        return self.k.values[cid]

    def shape(self, cid):
        return self.k.values[cid].shape

    def dtype(self, cid):
        return self.k.values[cid].dtype

    def const_data(self, cid):
        v = self.k.values[cid]
        if v.kind != "const":
            raise LoweringError(f"{self.node.source}: expected a constant operand")
        return v.data

    def scalar(self, x, dtype) -> str:
        dt = np.dtype(dtype)
        arr = np.asarray(x, dtype=dt)
        ex = None
        if dt.kind == "f":
            ex = Fraction(float(arr))
        return self.k.add_const(arr, exact=ex)

    def out_shape(self, slot=0):
        return self.src.values[self._slot_vid(slot)].shape

    def out_dtype(self, slot=0):
        return self.src.values[self._slot_vid(slot)].dtype

    def _slot_vid(self, slot):
        slots = self.node.meta.get("output_slots", list(range(len(self.node.outputs))))
        if slot not in slots:
            raise LoweringError(f"{self.node.source}: output slot {slot} not present")
        return self.node.outputs[slots.index(slot)]

    def has_slot(self, slot):
        return slot in self.node.meta.get("output_slots", list(range(len(self.node.outputs))))

    # ---- emission ----

    def emit(self, op, inputs, shape, dtype, attrs=None, meta=None) -> str:
        cid = self.k.fresh("t")
        self.k.add_value(Value(cid, tuple(shape), str(dtype)))
        m = {"origins": {self.node.id}}
        m.update(meta or {})
        self.k.add_node(Node(self.k.fresh("k"), op, list(inputs), [cid], dict(attrs or {}),
                             source=self.node.source, meta=m))
        return cid

    def emit_multi(self, op, inputs, outs, attrs=None) -> List[str]:
        cids = []
        for shape, dtype in outs:
            cid = self.k.fresh("t")
            self.k.add_value(Value(cid, tuple(shape), str(dtype)))
            cids.append(cid)
        self.k.add_node(Node(self.k.fresh("k"), op, list(inputs), cids, dict(attrs or {}),
                             source=self.node.source, meta={"origins": {self.node.id}}))
        return cids

    def opaque(self, reason=""):
        n = self.node
        outs = [(self.src.values[o].shape, self.src.values[o].dtype) for o in n.outputs]
        attrs = {"source": n.source, "attrs": _attr_key(_strip_inputs(n.attrs))}
        cids = self.emit_multi("opaque", [self.val(i) for i in n.inputs], outs, attrs)
        self.k.nodes[self.k.values[cids[0]].producer].meta["reason"] = reason or "no lowering rule"
        slots = n.meta.get("output_slots", list(range(len(n.outputs))))
        return dict(zip(slots, cids))

    # ---- shared builders ----

    def cast(self, cid, dtype):
        return cid if self.dtype(cid) == str(dtype) else self.emit(
            "cast", [cid], self.shape(cid), dtype, {"to": str(dtype)})

    def binary(self, op, a, b, out_dtype=None):
        """Elementwise binary op with explicit casts; Python numbers become scalar consts."""
        ta = a if isinstance(a, str) else None
        tb = b if isinstance(b, str) else None
        ref = ta or tb
        if op in COMPARE:
            dts = [self.dtype(x) for x in (ta, tb) if x is not None]
            calc = str(np.result_type(*[np.dtype(d) for d in dts])) if len(dts) == 2 else dts[0]
        else:
            calc = out_dtype or self.dtype(ref)
        a = self.scalar(a, calc) if ta is None else self.cast(ta, calc)
        b = self.scalar(b, calc) if tb is None else self.cast(tb, calc)
        shape = tuple(np.broadcast_shapes(self.shape(a), self.shape(b)))
        res_dtype = "bool" if op in COMPARE else calc
        return self.emit(op, [a, b], shape, res_dtype)

    def unary(self, op, x, attrs=None):
        return self.emit(op, [x], self.shape(x), self.dtype(x), attrs)

    def reshape(self, x, shape):
        shape = tuple(int(d) for d in shape)
        return x if self.shape(x) == shape else self.emit(
            "reshape", [x], shape, self.dtype(x), {"shape": shape})

    def permute(self, x, perm):
        perm = tuple(int(p) % len(self.shape(x)) for p in perm)
        if perm == tuple(range(len(perm))):
            return x
        shape = tuple(self.shape(x)[p] for p in perm)
        return self.emit("permute", [x], shape, self.dtype(x), {"perm": perm})

    def transpose2(self, x, d0, d1):
        r = len(self.shape(x))
        perm = list(range(r))
        perm[d0 % r], perm[d1 % r] = perm[d1 % r], perm[d0 % r]
        return self.permute(x, perm)

    def broadcast(self, x, shape):
        shape = tuple(shape)
        return x if self.shape(x) == shape else self.emit(
            "broadcast_to", [x], shape, self.dtype(x), {"shape": shape})

    def slice(self, x, specs):
        """specs: list of (axis, start, end, step) with step > 0."""
        shape = list(self.shape(x))
        norm = []
        for axis, start, end, step in specs:
            axis = axis % len(shape)
            n = shape[axis]
            if step is None:
                step = 1
            if step <= 0:
                raise LoweringError("slice with non-positive step")
            s, e, _ = slice(start, end, step).indices(n)
            if (s, e, step) == (0, n, 1):
                continue
            shape[axis] = len(range(s, e, step))
            norm.append((axis, s, e, step))
        if not norm:
            return x
        return self.emit("slice", [x], shape, self.dtype(x), {"spec": tuple(sorted(norm))})

    def matmul(self, a, b):
        sa, sb = self.shape(a), self.shape(b)
        if len(sa) < 1 or len(sb) < 1:
            raise LoweringError("matmul of a scalar")
        a2 = sa if len(sa) > 1 else (1,) + sa
        b2 = sb if len(sb) > 1 else sb + (1,)
        batch = tuple(np.broadcast_shapes(a2[:-2], b2[:-2]))
        shape = batch + (a2[-2], b2[-1])
        if len(sa) == 1:
            shape = shape[:-2] + shape[-1:]
        if len(sb) == 1:
            shape = shape[:-1]
        return self.emit("matmul", [a, b], shape, self.dtype(a))

    def conv(self, x, w, b, stride, pads, dilation, groups, out_shape):
        nsp = len(self.shape(x)) - 2
        attrs = {"stride": tuple(stride), "pads": tuple(pads), "dilation": tuple(dilation),
                 "groups": int(groups)}
        if len(attrs["stride"]) != nsp or len(attrs["pads"]) != 2 * nsp:
            raise LoweringError("conv attribute rank mismatch")
        y = self.emit("conv", [x, w], out_shape, self.dtype(x), attrs)
        if b is not None:
            bb = self.reshape(b, (self.shape(b)[0],) + (1,) * nsp)
            y = self.binary("add", y, bb)
        return y

    def reduce(self, op, x, axes, keepdim, out_shape):
        r = len(self.shape(x))
        axes = tuple(sorted({int(a) % r for a in axes})) if axes else tuple(range(r))
        return self.emit(op, [x], out_shape, self.dtype(x), {"axes": axes,
                                                            "keepdim": bool(keepdim)})

    def layer_norm(self, x, w, b, axes, eps):
        r = len(self.shape(x))
        axes = tuple(sorted(int(a) % r for a in axes))
        return self.emit("layer_norm", [x, w, b], self.shape(x), self.dtype(x),
                         {"axes": axes, "eps": _f32(eps)})


def _strip_inputs(attrs):
    return {k: v for k, v in attrs.items() if not _has_input_ref(v)}


def _has_input_ref(v):
    if isinstance(v, dict) and "$in" in v:
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_input_ref(x) for x in v)
    return False


def _attr_key(v):
    """Deterministic, hashable form of an attribute value."""
    import hashlib
    if isinstance(v, dict):
        return tuple(sorted((k, _attr_key(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_attr_key(x) for x in v)
    if isinstance(v, np.ndarray):
        return ("array", v.dtype.name, v.shape, hashlib.sha1(np.ascontiguousarray(v).tobytes())
                .hexdigest())
    if isinstance(v, float):
        return Fraction(v) if math.isfinite(v) else repr(v)
    if isinstance(v, (np.generic,)):
        return _attr_key(v.item())
    return v


# --------------------------------------------------------------------------------------------
# ATen rules
# --------------------------------------------------------------------------------------------

class AtenArgs:
    def __init__(self, L: Lowerer):
        self.L = L
        self.attrs = L.node.attrs

    def __call__(self, name, default=None):
        return self._resolve(self.attrs.get(name, default))

    def _resolve(self, v):
        if isinstance(v, dict) and "$in" in v:
            return self.L.val(self.L.node.inputs[v["$in"]])
        if isinstance(v, (list, tuple)):
            return [self._resolve(x) for x in v]
        return v


def _pairs(v, n):
    if v is None:
        return [None] * n
    if isinstance(v, int):
        return [v] * n
    v = list(v)
    return v * n if len(v) == 1 else v


def _aten_binary(op):
    def rule(L, A):
        a, b = A("self"), A("other") if "other" in L.node.attrs else A("exponent")
        if op == "div" and A("rounding_mode") is not None:
            return L.opaque("div with rounding_mode")
        alpha = A("alpha", 1)
        if alpha not in (None, 1) and op in ("add", "sub"):
            b = L.binary("mul", b, alpha) if isinstance(b, str) else b * alpha
        if op == "rsub":
            return {0: L.binary("sub", b, a, L.out_dtype())}
        return {0: L.binary(op, a, b, None if op in COMPARE else L.out_dtype())}
    return rule


def _aten_unary(op, **attrs):
    def rule(L, A):
        x = L.cast(A("self"), L.out_dtype()) if op not in ("not",) else A("self")
        extra = {k: (A(k, v) if k in L.node.attrs else v) for k, v in attrs.items()}
        return {0: L.unary(op, x, extra or None)}
    return rule


def _aten_alias(L, A):
    return {0: A("self") if "self" in L.node.attrs else A("input")}


def _aten_to(L, A):
    return {0: L.cast(A("self"), L.out_dtype())}


def _aten_reshape(L, A):
    return {0: L.reshape(A("self") if "self" in L.node.attrs else A("input"), L.out_shape())}


def _aten_expand(L, A):
    return {0: L.broadcast(A("self"), L.out_shape())}


def _aten_dropout(L, A):
    if A("train", False):
        return L.opaque("dropout in training mode")
    return {0: A("input")}


def _aten_linear(L, A):
    x, w, b = A("input"), A("weight"), A("bias")
    y = L.matmul(x, L.permute(w, (1, 0)))
    return {0: L.binary("add", y, b) if b is not None else y}


def _aten_addmm(L, A):
    beta, alpha = A("beta", 1), A("alpha", 1)
    y = L.matmul(A("mat1"), A("mat2"))
    if alpha != 1:
        y = L.binary("mul", y, alpha)
    c = A("self")
    if beta != 1:
        c = L.binary("mul", c, beta)
    return {0: L.binary("add", y, c)}


def _aten_baddbmm(L, A):
    beta, alpha = A("beta", 1), A("alpha", 1)
    y = L.matmul(A("batch1"), A("batch2"))
    if alpha != 1:
        y = L.binary("mul", y, alpha)
    c = A("self")
    if beta != 1:
        c = L.binary("mul", c, beta)
    return {0: L.binary("add", y, c)}


def _aten_matmul(L, A):
    a = A("self")
    b = A("other") if "other" in L.node.attrs else A("mat2")
    return {0: L.matmul(a, b)}


def _aten_conv(L, A):
    x, w, b = A("input"), A("weight"), A("bias")
    if A("transposed", False):
        return L.opaque("transposed convolution")
    nsp = len(L.shape(x)) - 2
    padding = A("padding", 0)
    if isinstance(padding, str):
        if padding != "valid":
            return L.opaque(f"padding={padding!r}")
        padding = 0
    pads = _pairs(padding, nsp)
    return {0: L.conv(x, w, b, _pairs(A("stride", 1), nsp), pads + pads,
                      _pairs(A("dilation", 1), nsp), A("groups", 1), L.out_shape())}


def _aten_batch_norm(L, A):
    if A("training", False):
        return L.opaque("batch_norm in training mode")
    x = A("input")
    y = L.emit("batch_norm", [x, A("weight"), A("bias"), A("running_mean"), A("running_var")],
               L.shape(x), L.dtype(x), {"eps": _f32(A("eps"))})
    return {0: y}


def _aten_layer_norm(L, A):
    x = A("input")
    k = len(A("normalized_shape"))
    r = len(L.shape(x))
    return {0: L.layer_norm(x, A("weight"), A("bias"), range(r - k, r), A("eps"))}


def _aten_max_pool(L, A):
    x = A("self")
    nsp = len(L.shape(x)) - 2
    k = _pairs(A("kernel_size"), nsp)
    s = A("stride") or k
    p = _pairs(A("padding", 0), nsp)
    y = L.emit("max_pool", [x], L.out_shape(), L.dtype(x),
               {"kernel": tuple(k), "stride": tuple(_pairs(s, nsp)), "pads": tuple(p + p),
                "dilation": tuple(_pairs(A("dilation", 1), nsp)),
                "ceil_mode": bool(A("ceil_mode", False))})
    out = {0: y}
    if L.has_slot(1):
        out[1] = L.opaque("max_pool indices")[1]
    return out


def _aten_avg_pool(L, A):
    x = A("self")
    if A("divisor_override") is not None:
        return L.opaque("avg_pool divisor_override")
    nsp = len(L.shape(x)) - 2
    k = _pairs(A("kernel_size"), nsp)
    s = A("stride") or k
    p = _pairs(A("padding", 0), nsp)
    return {0: L.emit("avg_pool", [x], L.out_shape(), L.dtype(x),
                      {"kernel": tuple(k), "stride": tuple(_pairs(s, nsp)), "pads": tuple(p + p),
                       "ceil_mode": bool(A("ceil_mode", False)),
                       "count_include_pad": bool(A("count_include_pad", True))})}


def _aten_adaptive_avg_pool(L, A):
    x = A("self")
    if all(d == 1 for d in L.out_shape()[2:]):
        return {0: L.reduce("reduce_mean", x, range(2, len(L.shape(x))), True, L.out_shape())}
    return L.opaque("adaptive pooling to a size other than 1")


def _aten_reduce(op):
    def rule(L, A):
        x = A("self")
        dims = A("dim") if "dim" in L.node.attrs else None
        if isinstance(dims, int):
            dims = [dims]
        if A("dtype") is not None:
            x = L.cast(x, A("dtype"))
        return {0: L.reduce(op, x, dims or range(len(L.shape(x))), A("keepdim", False),
                            L.out_shape())}
    return rule


def _aten_softmax(op):
    def rule(L, A):
        x = A("self")
        if A("dtype") is not None:
            x = L.cast(x, A("dtype"))
        axis = A("dim") % len(L.shape(x))
        return {0: L.emit(op, [x], L.shape(x), L.out_dtype(), {"axis": axis})}
    return rule


def _aten_permute(L, A):
    return {0: L.permute(A("self"), A("dims"))}


def _aten_transpose(L, A):
    return {0: L.transpose2(A("self"), A("dim0"), A("dim1"))}


def _aten_t(L, A):
    x = A("self")
    return {0: L.permute(x, (1, 0)) if len(L.shape(x)) == 2 else x}


def _aten_slice(L, A):
    return {0: L.slice(A("self"), [(A("dim", 0), A("start"), A("end"), A("step", 1))])}


def _aten_select(L, A):
    x = A("self")
    d = A("dim") % len(L.shape(x))
    i = A("index") % L.shape(x)[d]
    return {0: L.reshape(L.slice(x, [(d, i, i + 1, 1)]), L.out_shape())}


def _aten_narrow(L, A):
    s = A("start")
    return {0: L.slice(A("self"), [(A("dim"), s, s + A("length"), 1)])}


def _aten_split(L, A):
    x = A("self")
    d = A("dim", 0) % len(L.shape(x))
    out, start = {}, 0
    for slot in range(max(L.node.meta["output_slots"]) + 1):
        size = L.out_shape(slot)[d]
        out[slot] = L.slice(x, [(d, start, start + size, 1)])
        start += size
    return out


def _aten_unbind(L, A):
    x = A("self")
    d = A("dim", 0) % len(L.shape(x))
    out = {}
    for slot in L.node.meta["output_slots"]:
        out[slot] = L.reshape(L.slice(x, [(d, slot, slot + 1, 1)]), L.out_shape(slot))
    return out


def _aten_cat(L, A):
    xs = [L.cast(x, L.out_dtype()) for x in A("tensors")]
    axis = A("dim", 0) % len(L.out_shape())
    return {0: L.emit("concat", xs, L.out_shape(), L.out_dtype(), {"axis": axis})}


def _aten_stack(L, A):
    out = L.out_shape()
    axis = A("dim", 0) % len(out)
    xs = []
    for x in A("tensors"):
        s = list(L.shape(x))
        s.insert(axis, 1)
        xs.append(L.reshape(x, s))
    return {0: L.emit("concat", xs, out, L.out_dtype(), {"axis": axis})}


def _aten_where(L, A):
    c, a, b = A("condition"), A("self"), A("other")
    dt = L.out_dtype()
    a = L.scalar(a, dt) if not isinstance(a, str) else L.cast(a, dt)
    b = L.scalar(b, dt) if not isinstance(b, str) else L.cast(b, dt)
    return {0: L.emit("where", [c, a, b], L.out_shape(), dt)}


def _aten_masked_fill(L, A):
    x, m, v = A("self"), A("mask"), A("value")
    dt = L.out_dtype()
    v = L.scalar(v, dt) if not isinstance(v, str) else L.cast(v, dt)
    return {0: L.emit("where", [m, v, x], L.out_shape(), dt)}


def _aten_embedding(L, A):
    w, idx = A("weight"), A("indices")
    return {0: L.emit("gather", [w, idx], L.out_shape(), L.dtype(w), {"axis": 0})}


def _aten_index_select(L, A):
    x = A("self")
    return {0: L.emit("gather", [x, A("index")], L.out_shape(), L.dtype(x),
                      {"axis": A("dim") % len(L.shape(x))})}


def _aten_pad(L, A):
    x = A("self") if "self" in L.node.attrs else A("input")
    mode = A("mode", "constant")
    if mode != "constant":
        return L.opaque(f"pad mode {mode}")
    pad = list(A("pad"))
    r = len(L.shape(x))
    begins, ends = [0] * r, [0] * r
    for i in range(len(pad) // 2):
        begins[r - 1 - i], ends[r - 1 - i] = pad[2 * i], pad[2 * i + 1]
    v = L.scalar(A("value", 0.0) or 0.0, L.dtype(x))
    return {0: L.emit("pad", [x, v], L.out_shape(), L.dtype(x),
                      {"pads": tuple(begins + ends)})}


def _aten_clamp(L, A):
    x = A("self")
    lo, hi = A("min"), A("max")
    lo = None if lo is None else (lo if isinstance(lo, str) else L.scalar(lo, L.dtype(x)))
    hi = None if hi is None else (hi if isinstance(hi, str) else L.scalar(hi, L.dtype(x)))
    y = x
    if lo is not None:
        y = L.binary("max", y, lo)
    if hi is not None:
        y = L.binary("min", y, hi)
    return {0: y}


def _aten_gelu(L, A):
    return {0: L.unary("gelu", A("self"), {"approximate": A("approximate", "none")})}


def _aten_type_as(L, A):
    return {0: L.cast(A("self"), L.dtype(A("other")))}


ATEN_RULES: Dict[str, Callable] = {
    **{f"aten.{n}": _aten_binary(op) for n, op in [
        ("add", "add"), ("sub", "sub"), ("mul", "mul"), ("div", "div"), ("pow", "pow"),
        ("maximum", "max"), ("minimum", "min"), ("eq", "eq"), ("ne", "ne"), ("lt", "lt"),
        ("le", "le"), ("gt", "gt"), ("ge", "ge"), ("logical_and", "and"), ("logical_or", "or"),
        ("bitwise_and", "and"), ("bitwise_or", "or"), ("rsub", "rsub")]},
    **{f"aten.{n}": _aten_unary(op) for n, op in [
        ("relu", "relu"), ("tanh", "tanh"), ("exp", "exp"), ("log", "log"), ("sqrt", "sqrt"),
        ("rsqrt", "rsqrt"), ("neg", "neg"), ("sigmoid", "sigmoid"), ("erf", "erf"),
        ("abs", "abs"), ("reciprocal", "reciprocal"), ("sin", "sin"), ("cos", "cos"),
        ("silu", "silu"), ("logical_not", "not"), ("bitwise_not", "not")]},
    "aten.gelu": _aten_gelu,
    **{f"aten.{n}": _aten_alias for n in ["clone", "contiguous", "alias", "detach", "detach_",
                                          "lift_fresh_copy", "lift_fresh"]},
    "aten.dropout": _aten_dropout, "aten.feature_dropout": _aten_dropout,
    "aten._to_copy": _aten_to, "aten.to": _aten_to, "aten.type_as": _aten_type_as,
    **{f"aten.{n}": _aten_reshape for n in ["view", "_unsafe_view", "reshape", "flatten",
                                            "unflatten", "squeeze", "unsqueeze", "view_as"]},
    "aten.expand": _aten_expand, "aten.expand_as": _aten_expand, "aten.broadcast_to": _aten_expand,
    "aten.permute": _aten_permute, "aten.transpose": _aten_transpose, "aten.t": _aten_t,
    "aten.slice": _aten_slice, "aten.select": _aten_select, "aten.narrow": _aten_narrow,
    "aten.split": _aten_split, "aten.split_with_sizes": _aten_split, "aten.chunk": _aten_split,
    "aten.unbind": _aten_unbind, "aten.cat": _aten_cat, "aten.concat": _aten_cat,
    "aten.stack": _aten_stack,
    "aten.where": _aten_where, "aten.masked_fill": _aten_masked_fill,
    "aten.linear": _aten_linear, "aten.addmm": _aten_addmm, "aten.baddbmm": _aten_baddbmm,
    "aten.mm": _aten_matmul, "aten.bmm": _aten_matmul, "aten.matmul": _aten_matmul,
    "aten.conv1d": _aten_conv, "aten.conv2d": _aten_conv, "aten.convolution": _aten_conv,
    "aten.batch_norm": _aten_batch_norm, "aten._native_batch_norm_legit_no_training":
        _aten_batch_norm,
    "aten.layer_norm": _aten_layer_norm, "aten.native_layer_norm": _aten_layer_norm,
    "aten.max_pool2d": _aten_max_pool, "aten.max_pool2d_with_indices": _aten_max_pool,
    "aten.max_pool1d": _aten_max_pool, "aten.avg_pool2d": _aten_avg_pool,
    "aten.adaptive_avg_pool2d": _aten_adaptive_avg_pool,
    "aten.mean": _aten_reduce("reduce_mean"), "aten.sum": _aten_reduce("reduce_sum"),
    "aten.amax": _aten_reduce("reduce_max"), "aten.amin": _aten_reduce("reduce_min"),
    "aten.softmax": _aten_softmax("softmax"), "aten._softmax": _aten_softmax("softmax"),
    "aten.log_softmax": _aten_softmax("log_softmax"),
    "aten._log_softmax": _aten_softmax("log_softmax"),
    "aten.embedding": _aten_embedding, "aten.index_select": _aten_index_select,
    "aten.constant_pad_nd": _aten_pad, "aten.pad": _aten_pad,
    "aten.clamp": _aten_clamp, "aten.clip": _aten_clamp, "aten.hardtanh": None,
}
ATEN_RULES = {k: v for k, v in ATEN_RULES.items() if v is not None}


# --------------------------------------------------------------------------------------------
# ONNX rules (ai.onnx and com.microsoft)
# --------------------------------------------------------------------------------------------

class OnnxArgs:
    def __init__(self, L: Lowerer):
        self.L = L

    def __call__(self, k):
        ins = self.L.node.inputs
        return self.L.val(ins[k]) if k < len(ins) and ins[k] is not None else None

    def at(self, name, default=None):
        return self.L.node.attrs.get(name, default)

    def ints(self, k):
        cid = self(k)
        return None if cid is None else [int(x) for x in np.asarray(self.L.const_data(cid)).ravel()]


def _onnx_dtype(enum):
    import onnx
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(enum)).name


def _ox_binary(op):
    def rule(L, O):
        return {0: L.binary(op, O(0), O(1), None if op in COMPARE else L.out_dtype())}
    return rule


def _ox_nary(op):
    def rule(L, O):
        y = O(0)
        for k in range(1, len(L.node.inputs)):
            y = L.binary(op, y, O(k), L.out_dtype())
        return {0: y}
    return rule


def _ox_unary(op):
    def rule(L, O):
        return {0: L.unary(op, O(0))}
    return rule


def _ox_alias(L, O):
    return {0: O(0)}


def _ox_reshape(L, O):
    return {0: L.reshape(O(0), L.out_shape())}


def _ox_expand(L, O):
    return {0: L.broadcast(O(0), L.out_shape())}


def _ox_cast(L, O):
    return {0: L.cast(O(0), _onnx_dtype(O.at("to")))}


def _ox_transpose(L, O):
    x = O(0)
    perm = O.at("perm") or list(reversed(range(len(L.shape(x)))))
    return {0: L.permute(x, perm)}


def _ox_matmul(L, O):
    return {0: L.matmul(O(0), O(1))}


def _ox_gemm(L, O):
    a, b, c = O(0), O(1), O(2)
    if O.at("transA", 0):
        a = L.permute(a, (1, 0))
    if O.at("transB", 0):
        b = L.permute(b, (1, 0))
    y = L.matmul(a, b)
    alpha, beta = O.at("alpha", 1.0), O.at("beta", 1.0)
    if alpha != 1.0:
        y = L.binary("mul", y, alpha)
    if c is not None:
        if beta != 1.0:
            c = L.binary("mul", c, beta)
        y = L.binary("add", y, c)
    return {0: y}


def _ox_conv_core(L, O, x, w, b):
    if O.at("auto_pad", "NOTSET") not in ("NOTSET", "VALID"):
        raise LoweringError(f"auto_pad={O.at('auto_pad')}")
    nsp = len(L.shape(x)) - 2
    pads = O.at("pads") or [0] * (2 * nsp)
    return L.conv(x, w, b, O.at("strides") or [1] * nsp, pads, O.at("dilations") or [1] * nsp,
                  O.at("group", 1), L.out_shape())


def _ox_conv(L, O):
    return {0: _ox_conv_core(L, O, O(0), O(1), O(2))}


def _activation(L, y, name, params):
    if not name:
        return y
    name = name.decode() if isinstance(name, bytes) else name
    if name == "Relu":
        return L.unary("relu", y)
    if name == "Sigmoid":
        return L.unary("sigmoid", y)
    if name == "Tanh":
        return L.unary("tanh", y)
    if name == "Clip" and params and len(params) == 2:
        y = L.binary("max", y, params[0])
        return L.binary("min", y, params[1])
    if name == "LeakyRelu":
        raise LoweringError("LeakyRelu activation")
    raise LoweringError(f"activation {name}")


def _ox_fused_conv(L, O):
    y = _ox_conv_core(L, O, O(0), O(1), O(2))
    if O(3) is not None:
        y = L.binary("add", y, O(3))
    return {0: _activation(L, y, O.at("activation"), O.at("activation_params"))}


def _ox_fused_gemm(L, O):
    y = _ox_gemm(L, O)[0]
    return {0: _activation(L, y, O.at("activation"), O.at("activation_params"))}


def _ox_fused_matmul(L, O):
    a, b = O(0), O(1)
    if O.at("transBatchA", 0) or O.at("transBatchB", 0):
        raise LoweringError("FusedMatMul batch transposes")
    if O.at("transA", 0):
        a = L.transpose2(a, -1, -2)
    if O.at("transB", 0):
        b = L.transpose2(b, -1, -2)
    y = L.matmul(a, b)
    alpha = O.at("alpha", 1.0)
    return {0: L.binary("mul", y, alpha) if alpha != 1.0 else y}


def _ox_batch_norm(L, O):
    x = O(0)
    return {0: L.emit("batch_norm", [x, O(1), O(2), O(3), O(4)], L.shape(x), L.dtype(x),
                      {"eps": _f32(O.at("epsilon", 1e-5))})}


def _ox_layer_norm(L, O):
    x = O(0)
    r = len(L.shape(x))
    axis = O.at("axis", -1) % r
    if L.has_slot(1) or L.has_slot(2):
        raise LoweringError("LayerNormalization mean/inv_std outputs")
    return {0: L.layer_norm(x, O(1), O(2), range(axis, r), O.at("epsilon", 1e-5))}


def _ox_skip_layer_norm(L, O):
    x, skip, gamma, beta, bias = O(0), O(1), O(2), O(3), O(4)
    s = L.binary("add", x, skip)
    if bias is not None:
        s = L.binary("add", s, bias)
    if L.has_slot(1) or L.has_slot(2):
        raise LoweringError("SkipLayerNormalization mean/inv_std outputs")
    r = len(L.shape(x))
    out = {0: L.layer_norm(s, gamma, beta, [r - 1], O.at("epsilon", 1e-12))}
    if L.has_slot(3):
        out[3] = s
    return out


def _ox_simplified_ln(L, O):
    x = O(0)
    r = len(L.shape(x))
    axis = O.at("axis", -1) % r
    return {0: L.emit("rms_norm", [x, O(1)], L.shape(x), L.dtype(x),
                      {"axes": tuple(range(axis, r)), "eps": _f32(O.at("epsilon", 1e-5))})}


def _ox_skip_simplified_ln(L, O):
    x, skip, gamma, bias = O(0), O(1), O(2), O(3)
    s = L.binary("add", x, skip)
    if bias is not None:
        s = L.binary("add", s, bias)
    r = len(L.shape(x))
    out = {0: L.emit("rms_norm", [s, gamma], L.shape(x), L.dtype(x),
                     {"axes": (r - 1,), "eps": _f32(O.at("epsilon", 1e-5))})}
    if L.has_slot(3):
        out[3] = s
    return out


def _ox_gelu(approximate):
    def rule(L, O):
        x = O(0)
        if O(1) is not None:
            x = L.binary("add", x, O(1))
        approx = O.at("approximate", approximate)
        return {0: L.unary("gelu", x, {"approximate": approx})}
    return rule


def _ox_quick_gelu(L, O):
    x = O(0)
    s = L.unary("sigmoid", L.binary("mul", x, O.at("alpha", 1.702)))
    return {0: L.binary("mul", x, s)}


def _ox_bias_add(L, O):
    y = L.binary("add", O(0), O(1))
    return {0: L.binary("add", y, O(2)) if O(2) is not None else y}


def _ox_pool(op):
    def rule(L, O):
        x = O(0)
        nsp = len(L.shape(x)) - 2
        if O.at("auto_pad", "NOTSET") not in ("NOTSET", "VALID"):
            raise LoweringError(f"auto_pad={O.at('auto_pad')}")
        k = O.at("kernel_shape")
        attrs = {"kernel": tuple(k), "stride": tuple(O.at("strides") or [1] * nsp),
                 "pads": tuple(O.at("pads") or [0] * (2 * nsp)),
                 "ceil_mode": bool(O.at("ceil_mode", 0))}
        if op == "max_pool":
            attrs["dilation"] = tuple(O.at("dilations") or [1] * nsp)
        else:
            attrs["count_include_pad"] = bool(O.at("count_include_pad", 0))
        out = {0: L.emit(op, [x], L.out_shape(), L.dtype(x), attrs)}
        if L.has_slot(1):
            raise LoweringError("MaxPool indices output")
        return out
    return rule


def _ox_global_pool(op):
    def rule(L, O):
        x = O(0)
        return {0: L.reduce(op, x, range(2, len(L.shape(x))), True, L.out_shape())}
    return rule


def _ox_reduce(op):
    def rule(L, O):
        x = O(0)
        axes = O.ints(1) if len(L.node.inputs) > 1 and L.node.inputs[1] else O.at("axes")
        if not axes and O.at("noop_with_empty_axes", 0):
            return {0: x}
        return {0: L.reduce(op, x, axes or range(len(L.shape(x))), O.at("keepdims", 1),
                            L.out_shape())}
    return rule


def _ox_softmax(op):
    def rule(L, O):
        x = O(0)
        opset = L.src.meta.get("opset", {}).get("ai.onnx", 13)
        axis = O.at("axis", -1 if opset >= 13 else 1) % len(L.shape(x))
        if opset < 13 and axis != len(L.shape(x)) - 1:
            raise LoweringError("pre-13 softmax over flattened axes")
        return {0: L.emit(op, [x], L.shape(x), L.dtype(x), {"axis": axis})}
    return rule


def _ox_gather(L, O):
    x, idx = O(0), O(1)
    axis = O.at("axis", 0) % len(L.shape(x))
    iv = L.V(idx)
    if iv.kind == "const" and iv.numel == 1:
        # Gather with one constant index is a slice (+ reshape when the index is a scalar)
        i = int(np.asarray(iv.data).reshape(())) % L.shape(x)[axis]
        return {0: L.reshape(L.slice(x, [(axis, i, i + 1, 1)]), L.out_shape())}
    return {0: L.emit("gather", [x, idx], L.out_shape(), L.dtype(x), {"axis": axis})}


def _ox_slice(L, O):
    x = O(0)
    starts, ends = O.ints(1), O.ints(2)
    axes = O.ints(3) or list(range(len(starts)))
    steps = O.ints(4) or [1] * len(starts)
    big = 2 ** 62
    specs = [(a, s, min(e, big), st) for a, s, e, st in zip(axes, starts, ends, steps)]
    return {0: L.slice(x, specs)}


def _ox_split(L, O):
    x = O(0)
    axis = O.at("axis", 0) % len(L.shape(x))
    out, start = {}, 0
    for slot in range(len(L.node.meta["onnx_outputs"])):
        if not L.has_slot(slot):
            raise LoweringError("Split output missing")
        size = L.out_shape(slot)[axis]
        out[slot] = L.slice(x, [(axis, start, start + size, 1)])
        start += size
    return out


def _ox_concat(L, O):
    xs = [O(k) for k in range(len(L.node.inputs))]
    return {0: L.emit("concat", xs, L.out_shape(), L.out_dtype(),
                      {"axis": O.at("axis") % len(L.out_shape())})}


def _ox_where(L, O):
    return {0: L.emit("where", [O(0), O(1), O(2)], L.out_shape(), L.out_dtype())}


def _ox_pad(L, O):
    if O.at("mode", "constant") != "constant":
        raise LoweringError(f"pad mode {O.at('mode')}")
    x = O(0)
    pads = O.ints(1) if len(L.node.inputs) > 1 else O.at("pads")
    if len(L.node.inputs) > 3 and L.node.inputs[3]:
        raise LoweringError("Pad with axes input")
    v = O(2) if len(L.node.inputs) > 2 and L.node.inputs[2] else L.scalar(0.0, L.dtype(x))
    return {0: L.emit("pad", [x, v], L.out_shape(), L.dtype(x), {"pads": tuple(pads)})}


def _ox_clip(L, O):
    y = O(0)
    if O(1) is not None:
        y = L.binary("max", y, O(1))
    if O(2) is not None:
        y = L.binary("min", y, O(2))
    return {0: y}


def _ox_dropout(L, O):
    if L.has_slot(1):
        raise LoweringError("Dropout mask output")
    return {0: O(0)}


ONNX_RULES: Dict[str, Callable] = {
    **{f"ai.onnx::{n}": _ox_binary(op) for n, op in [
        ("Add", "add"), ("Sub", "sub"), ("Mul", "mul"), ("Div", "div"), ("Pow", "pow"),
        ("Equal", "eq"), ("Less", "lt"), ("LessOrEqual", "le"), ("Greater", "gt"),
        ("GreaterOrEqual", "ge"), ("And", "and"), ("Or", "or")]},
    "ai.onnx::Max": _ox_nary("max"), "ai.onnx::Min": _ox_nary("min"),
    "ai.onnx::Sum": _ox_nary("add"),
    **{f"ai.onnx::{n}": _ox_unary(op) for n, op in [
        ("Relu", "relu"), ("Tanh", "tanh"), ("Exp", "exp"), ("Log", "log"), ("Sqrt", "sqrt"),
        ("Neg", "neg"), ("Sigmoid", "sigmoid"), ("Erf", "erf"), ("Abs", "abs"),
        ("Reciprocal", "reciprocal"), ("Sin", "sin"), ("Cos", "cos"), ("Not", "not")]},
    "ai.onnx::Gelu": _ox_gelu("none"),
    "ai.onnx::Identity": _ox_alias, "ai.onnx::Dropout": _ox_dropout,
    "ai.onnx::Cast": _ox_cast,
    **{f"ai.onnx::{n}": _ox_reshape for n in ["Reshape", "Flatten", "Squeeze", "Unsqueeze"]},
    "ai.onnx::Expand": _ox_expand, "ai.onnx::Transpose": _ox_transpose,
    "ai.onnx::MatMul": _ox_matmul, "ai.onnx::Gemm": _ox_gemm, "ai.onnx::Conv": _ox_conv,
    "ai.onnx::BatchNormalization": _ox_batch_norm,
    "ai.onnx::LayerNormalization": _ox_layer_norm,
    "ai.onnx::MaxPool": _ox_pool("max_pool"), "ai.onnx::AveragePool": _ox_pool("avg_pool"),
    "ai.onnx::GlobalAveragePool": _ox_global_pool("reduce_mean"),
    "ai.onnx::GlobalMaxPool": _ox_global_pool("reduce_max"),
    "ai.onnx::ReduceMean": _ox_reduce("reduce_mean"), "ai.onnx::ReduceSum": _ox_reduce("reduce_sum"),
    "ai.onnx::ReduceMax": _ox_reduce("reduce_max"), "ai.onnx::ReduceMin": _ox_reduce("reduce_min"),
    "ai.onnx::Softmax": _ox_softmax("softmax"), "ai.onnx::LogSoftmax": _ox_softmax("log_softmax"),
    "ai.onnx::Gather": _ox_gather, "ai.onnx::Slice": _ox_slice, "ai.onnx::Split": _ox_split,
    "ai.onnx::Concat": _ox_concat, "ai.onnx::Where": _ox_where, "ai.onnx::Pad": _ox_pad,
    "ai.onnx::Clip": _ox_clip,
    "com.microsoft::FusedConv": _ox_fused_conv, "com.microsoft::FusedGemm": _ox_fused_gemm,
    "com.microsoft::FusedMatMul": _ox_fused_matmul,
    "com.microsoft::BiasGelu": _ox_gelu("none"), "com.microsoft::Gelu": _ox_gelu("none"),
    "com.microsoft::FastGelu": _ox_gelu("tanh"), "com.microsoft::QuickGelu": _ox_quick_gelu,
    "com.microsoft::BiasAdd": _ox_bias_add,
    "com.microsoft::SkipLayerNormalization": _ox_skip_layer_norm,
    "com.microsoft::LayerNormalization": _ox_layer_norm,
    "com.microsoft::SimplifiedLayerNormalization": _ox_simplified_ln,
    "ai.onnx::RMSNormalization": _ox_simplified_ln,
    "com.microsoft::SkipSimplifiedLayerNormalization": _ox_skip_simplified_ln,
}


CUSTOM_RULES: Dict[str, Callable] = {}


def register_lowering(op: str, rule: Callable):
    """Declare the logical computation of a custom (e.g. fused) kernel.

    `op` is a torch op name ("mylib.fused_add_relu", overload optional) or an ONNX op
    ("my.domain::FusedAddRelu"). `rule(L, args)` builds canonical ops with the `Lowerer` `L` and
    returns {output_slot: canonical value}. For torch ops `args(name)` reads an argument by its
    schema name; for ONNX ops `args(k)` reads input k and `args.at(name)` an attribute.
    """
    CUSTOM_RULES[op] = rule


def lowering_rule(node: Node):
    is_onnx = "::" in node.op
    Args = OnnxArgs if is_onnx else AtenArgs
    for key in (node.op, ".".join(node.op.split(".")[:2])):
        if key in CUSTOM_RULES:
            return CUSTOM_RULES[key], Args
    if node.op.startswith("aten."):
        return ATEN_RULES.get(aten_name(node.op)), AtenArgs
    return ONNX_RULES.get(node.op), Args


# --------------------------------------------------------------------------------------------
# Canonical rewrites
# --------------------------------------------------------------------------------------------

def _replace_uses(k: TensorGraph, old: str, new: str):
    for n in k.nodes.values():
        n.inputs = [new if i == old else i for i in n.inputs]
    k.outputs = [new if o == old else o for o in k.outputs]
    k._consumers = None


def _single_use(k: TensorGraph, vid: str) -> bool:
    return vid not in k.outputs and len(k.consumers(vid)) == 1 and \
        sum(i == vid for i in k.consumers(vid)[0].inputs) == 1


def _scalar_const(k, vid) -> Optional[Fraction]:
    return exact_scalar(k.values[vid]) if vid is not None else None


def _fold_constants(k: TensorGraph) -> bool:
    changed = False
    for n in list(k.topo_order()):
        if n.op not in EXACT_FOLD or len(n.outputs) != 1:
            continue
        if not all(i is None or k.is_const(i) for i in n.inputs):
            continue
        xs = [k.values[i].data if i is not None else None for i in n.inputs]
        try:
            y = _eval_exact(n, xs, k.values[n.outputs[0]])
        except Exception:
            continue
        out = n.outputs[0]
        k.remove_node(n.id)
        old = k.values.pop(out)
        exact = None
        if n.op == "neg" and xs[0].size == 1:
            ex = exact_scalar(k.values[n.inputs[0]])
            exact = -ex if ex is not None else None
        k.add_value(Value(out, old.shape, old.dtype, "const", data=np.asarray(y, old.dtype),
                          origin=old.origin, exact=exact))
        changed = True
    return changed


def _eval_exact(n, xs, out: Value):
    a = n.attrs
    if n.op == "reshape":
        return xs[0].reshape(a["shape"])
    if n.op == "permute":
        return np.transpose(xs[0], a["perm"])
    if n.op == "broadcast_to":
        return np.broadcast_to(xs[0], a["shape"]).copy()
    if n.op == "slice":
        idx = [slice(None)] * xs[0].ndim
        for axis, s, e, st in a["spec"]:
            idx[axis] = slice(s, e, st)
        return xs[0][tuple(idx)]
    if n.op == "concat":
        return np.concatenate(xs, axis=a["axis"])
    if n.op == "cast":
        return xs[0].astype(a["to"])
    if n.op == "gather":
        return np.take(xs[0], xs[1], axis=a["axis"])
    if n.op == "neg":
        return -xs[0]
    if n.op == "where":
        return np.where(xs[0], xs[1], xs[2])
    if n.op == "not":
        return np.logical_not(xs[0])
    ops = {"eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal,
           "gt": np.greater, "ge": np.greater_equal, "and": np.logical_and, "or": np.logical_or}
    return ops[n.op](xs[0], xs[1])


def _simplify_layout(k: TensorGraph) -> bool:
    changed = False
    for n in list(k.topo_order()):
        if n.id not in k.nodes:
            continue
        x = n.inputs[0] if n.inputs else None
        out = n.outputs[0] if n.outputs else None
        if n.op in ("reshape", "broadcast_to", "cast", "permute", "slice") and x is not None:
            vx, vo = k.values[x], k.values[out]
            if n.op in ("reshape", "broadcast_to") and vx.shape == vo.shape \
                    or n.op == "cast" and vx.dtype == vo.dtype \
                    or n.op == "permute" and n.attrs["perm"] == tuple(range(len(vx.shape))) \
                    or n.op == "slice" and not n.attrs["spec"]:
                _alias_out(k, n, x)
                changed = True
                continue
        p = k.producer(x) if x is not None else None
        if p is None or p.op != n.op or not _single_use(k, x):
            continue
        if n.op == "reshape":
            n.inputs = [p.inputs[0]]
        elif n.op == "permute":
            inner, outer = p.attrs["perm"], n.attrs["perm"]
            n.attrs = {"perm": tuple(inner[i] for i in outer)}
            n.inputs = [p.inputs[0]]
        elif n.op == "broadcast_to":
            n.inputs = [p.inputs[0]]
        elif n.op == "slice":
            axes_p = {s[0] for s in p.attrs["spec"]}
            axes_n = {s[0] for s in n.attrs["spec"]}
            if axes_p & axes_n:
                continue
            n.attrs = {"spec": tuple(sorted(p.attrs["spec"] + n.attrs["spec"]))}
            n.inputs = [p.inputs[0]]
        else:
            continue
        _merge_origins(n, p)
        k.remove_node(p.id)
        k._consumers = None
        changed = True
    return changed


def _alias_out(k, n, x):
    out = n.outputs[0]
    k.remove_node(n.id)
    _replace_uses(k, out, x)
    k.meta.setdefault("elided_origins", set()).update(n.meta.get("origins", ()))


def _merge_origins(n, p):
    n.meta["origins"] = set(n.meta.get("origins", ())) | set(p.meta.get("origins", ()))
    if p.meta.get("approximate_constants"):
        n.meta["approximate_constants"] = True


def _elide_broadcasts(k: TensorGraph) -> bool:
    changed = False
    for n in list(k.topo_order()):
        if n.op != "broadcast_to" or n.outputs[0] in k.outputs:
            continue
        out = n.outputs[0]
        cons = k.consumers(out)
        if cons and all(c.op in ELEMENTWISE and c.op != "where" and
                        k.values[c.outputs[0]].shape == k.values[out].shape for c in cons):
            _alias_out(k, n, n.inputs[0])
            changed = True
    return changed


def _normalize_arith(k: TensorGraph) -> bool:
    """sub -> add(neg), div by scalar const -> mul by exact reciprocal."""
    changed = False
    for n in list(k.topo_order()):
        if n.op == "sub":
            b = n.inputs[1]
            ex = _scalar_const(k, b)
            if ex is not None:
                vb = k.values[b]
                neg = k.add_const(-vb.data, exact=-ex)
            else:
                neg = k.fresh("t")
                k.add_value(Value(neg, k.values[b].shape, k.values[b].dtype))
                k.add_node(Node(k.fresh("k"), "neg", [b], [neg], {}, source=n.source,
                                meta={"origins": set(n.meta["origins"])}))
            n.op, n.inputs = "add", [n.inputs[0], neg]
            changed = True
        elif n.op == "div":
            ex = _scalar_const(k, n.inputs[1])
            vb = k.values[n.inputs[1]]
            if ex is not None and ex != 0 and np.dtype(vb.dtype).kind == "f":
                r = 1 / ex
                c = k.add_const(np.asarray(float(r), dtype=vb.dtype), exact=r)
                n.op, n.inputs = "mul", [n.inputs[0], c]
                changed = True
    if changed:
        k.reorder()
    return changed


def _flatten_nary(k: TensorGraph) -> bool:
    changed = False
    for n in list(k.topo_order()):
        if n.op not in NARY or n.id not in k.nodes:
            continue
        new_inputs, absorbed = [], []
        for i in n.inputs:
            p = k.producer(i)
            if p is not None and p.op == n.op and _single_use(k, i) and \
                    k.values[i].dtype == k.values[n.outputs[0]].dtype:
                new_inputs.extend(p.inputs)
                absorbed.append(p)
            else:
                new_inputs.append(i)
        if n.op == "mul":
            new_inputs = _combine_scalars(k, new_inputs, n)
        if absorbed or new_inputs != n.inputs:
            for p in absorbed:
                _merge_origins(n, p)
                k.remove_node(p.id)
            n.inputs = new_inputs
            k._consumers = None
            changed = True
        if len(n.inputs) == 1 and k.values[n.inputs[0]].shape == k.values[n.outputs[0]].shape \
                and k.values[n.inputs[0]].dtype == k.values[n.outputs[0]].dtype:
            _alias_out(k, n, n.inputs[0])
            changed = True
    if changed:
        k.reorder()
    return changed


def _combine_scalars(k, inputs, n):
    scal = [(i, _scalar_const(k, i)) for i in inputs]
    exs = [e for _, e in scal if e is not None]
    if len(exs) < 2 and not (len(exs) == 1 and exs[0] == 1 and len(inputs) > 1):
        return inputs
    rest = [i for i, e in scal if e is None]
    prod = Fraction(1)
    for e in exs:
        prod *= e
    if prod == 1 and rest:
        return rest
    dt = k.values[n.outputs[0]].dtype
    return rest + [k.add_const(np.asarray(float(prod), dtype=dt), exact=prod)]


def _hoist_scalars_from_matmul(k: TensorGraph) -> bool:
    changed = False
    for n in list(k.topo_order()):
        if n.op != "matmul":
            continue
        for pos in (0, 1):
            x = n.inputs[pos]
            p = k.producer(x)
            if p is None or p.op != "mul" or not _single_use(k, x):
                continue
            scal = [i for i in p.inputs if _scalar_const(k, i) is not None]
            rest = [i for i in p.inputs if _scalar_const(k, i) is None]
            if len(scal) != 1 or len(rest) != 1:
                continue
            if k.values[rest[0]].shape != k.values[x].shape:
                continue
            n.inputs[pos] = rest[0]
            out = n.outputs[0]
            new = k.fresh("t")
            k.add_value(Value(new, k.values[out].shape, k.values[out].dtype))
            _replace_uses(k, out, new)
            k.remove_node(p.id)
            k.add_node(Node(k.fresh("k"), "mul", [out, scal[0]], [new], {}, source=p.source,
                            meta={"origins": set(p.meta["origins"])}))
            changed = True
    if changed:
        k.reorder()
    return changed


SQRT1_2 = 1 / math.sqrt(2)
SQRT_2_PI = math.sqrt(2 / math.pi)


def _close(ex: Optional[Fraction], target: float, rel=1e-6) -> bool:
    return ex is not None and abs(float(ex) - target) <= rel * abs(target)


def _mul_parts(k, vid):
    """(scalar Fraction or None, non-scalar inputs, node) for a mul node producing vid."""
    p = k.producer(vid)
    if p is None or p.op != "mul":
        return None
    scal = [_scalar_const(k, i) for i in p.inputs]
    exs = [e for e in scal if e is not None]
    rest = [i for i, e in zip(p.inputs, scal) if e is None]
    return (exs[0] if len(exs) == 1 else (None if not exs else "many")), rest, p


def _add_one(k, vid):
    """If vid = add(1, y) return (y, node)."""
    p = k.producer(vid)
    if p is None or p.op != "add" or len(p.inputs) != 2:
        return None
    for a, b in (p.inputs, p.inputs[::-1]):
        if _scalar_const(k, a) == 1:
            return b, p
    return None


def _match_gelu(k, n):
    """n: mul node. Returns (x, approximate, pattern nodes, constants) or None. `constants` are
    the exact values the expression uses (`scale`, and `cubic` for the tanh form); the
    interpreter bounds their distance to the reference GELU constants."""
    parts = _mul_parts(k, n.outputs[0])
    if parts is None or parts[0] is None or parts[0] == "many" or parts[0] != Fraction(1, 2):
        return None
    _, rest, _ = parts
    if len(rest) != 2:
        return None
    for x, other in (rest, rest[::-1]):
        a1 = _add_one(k, other)
        if a1 is None:
            continue
        inner, add_node = a1
        f = k.producer(inner)
        if f is None or f.op not in ("erf", "tanh"):
            continue
        z = f.inputs[0]
        zp = _mul_parts(k, z)
        if zp is None or zp[0] in (None, "many") or len(zp[1]) != 1:
            continue
        c, (zin,), zmul = zp
        nodes = [n, add_node, f, zmul]
        if f.op == "erf":
            if zin == x and _close(c, SQRT1_2):
                return x, "none", nodes, {"scale": c}
            continue
        if not _close(c, SQRT_2_PI):
            continue
        s = k.producer(zin)
        if s is None or s.op != "add" or len(s.inputs) != 2 or x not in s.inputs:
            continue
        cube = s.inputs[1] if s.inputs[0] == x else s.inputs[0]
        cp = _mul_parts(k, cube)
        if cp is None or not _close(cp[0] if cp[0] != "many" else None, 0.044715):
            continue
        q = cp[1]
        qn = k.producer(q[0]) if len(q) == 1 else None
        consts = {"scale": c, "cubic": cp[0]}
        if len(q) == 3 and all(v == x for v in q):
            return x, "tanh", nodes + [s, cp[2]], consts
        if qn is not None and qn.op == "pow" and qn.inputs[0] == x and \
                _scalar_const(k, qn.inputs[1]) == 3:
            return x, "tanh", nodes + [s, cp[2], qn], consts
    return None


def _recognize_gelu(k: TensorGraph) -> bool:
    changed = False
    for n in list(k.topo_order()):
        if n.id not in k.nodes or n.op != "mul":
            continue
        m = _match_gelu(k, n)
        if m is None:
            continue
        x, approx, nodes, consts = m
        inside = {p.id for p in nodes}
        internal = [v for p in nodes[1:] for v in p.outputs]
        if any(v in k.outputs or any(c.id not in inside for c in k.consumers(v))
               for v in internal):
            continue      # an inner value is used elsewhere: keep the expression
        origins = set()
        for p in nodes:
            origins |= set(p.meta.get("origins", ()))
        out = n.outputs[0]
        for p in nodes:
            if p.id in k.nodes:
                k.remove_node(p.id)
        k.add_node(Node(k.fresh("k"), "gelu", [x], [out], {"approximate": approx},
                        source=n.source, meta={"origins": origins,
                                               "approximate_constants": True,
                                               "idiom_constants": consts}))
        changed = True
    if changed:
        k.reorder()
        k.dead_code_elimination()
    return changed


def canonicalize(src: TensorGraph, fold_constants=True, normalize_arith=True,
                 flatten=True, recognize_idioms=True) -> TensorGraph:
    """Lower `src` to a canonical graph and apply the exact rewrites."""
    L = Lowerer(src)
    for n in src.nodes.values():
        L.node = n
        rule, Args = lowering_rule(n)
        try:
            if rule is None:
                res = L.opaque()
            else:
                res = rule(L, Args(L))
        except LoweringError as e:
            res = L.opaque(str(e))
        slots = n.meta.get("output_slots", list(range(len(n.outputs))))
        for vid, slot in zip(n.outputs, slots):
            cid = res.get(slot)
            if cid is None:
                if not src.consumers(vid) and vid not in src.outputs:
                    continue          # unused extra output, e.g. batch-norm statistics
                raise LoweringError(f"rule for {n.op} produced no output for slot {slot}")
            sv, cv = src.values[vid], L.k.values[cid]
            if tuple(sv.shape) != tuple(cv.shape):
                raise LoweringError(f"{n.op} ({n.id}): lowered shape {cv.shape} != {sv.shape}")
            if cid not in L.k.inputs and L.k.producer(cid) is None and cv.kind != "const":
                raise LoweringError(f"{n.op}: dangling output")
            if cv.origin is None:
                cv.origin = vid
            L.vmap[vid] = cid
    k = L.k
    k.outputs = [L.val(o) for o in src.outputs]
    for _ in range(20):
        changed = False
        if fold_constants:
            changed |= _fold_constants(k)
        changed |= _simplify_layout(k)
        changed |= _elide_broadcasts(k)
        if normalize_arith:
            changed |= _normalize_arith(k)
        if flatten:
            changed |= _flatten_nary(k)
        if normalize_arith:
            changed |= _hoist_scalars_from_matmul(k)
        if recognize_idioms:
            changed |= _recognize_gelu(k)
        k.dead_code_elimination()
        if not changed:
            break
    k.reorder()
    live = set()
    for n in k.nodes.values():
        live |= set(n.meta.get("origins", ()))
    k.meta["elided_origins"] = {nid for nid in src.nodes if nid not in live}
    k.validate()
    return k
