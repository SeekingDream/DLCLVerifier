"""Reconstruction of a TVM program from its lowered representation.

`compilers.compile_tvm` keeps the module TVM generates code from: a Relax `main` whose bindings
are `call_tir` calls, and one scheduled TIR function per call. A TIR function is an explicit
loop nest over buffers: every addition, multiplication, division, comparison and index is
written out, and so is the order in which a reduction visits its terms. Nothing in it calls a
library. The reconstruction therefore does not look at operator names at all:

* `build_tvm_graph` turns the Relax `main` into an ordered graph with one node per `call_tir`;
* `TirLowering` replaces each node by `run_primfunc`, an interpreter that executes the TIR
  statement by statement with `dlclv.primitives` operations only.

What the TIR does not say, and is therefore a stated assumption that is checked on the device:

1. **FMA contraction.** The CUDA compiler contracts `a * b + c` into one fused multiply-add.
   The interpreter applies exactly this rule: an addition or subtraction with exactly one
   multiplication as a direct operand is one `fma32`. Both operands being products is
   ambiguous and is refused.
2. **`max` / `min` ties.** What is returned when both operands are equal, which matters for
   +0 against -0: the first operand, the second, or +0 above -0 whatever the order.

For every kernel the candidates (contracted or not, either tie convention, as far as the
function contains such an expression) are run against the compiled kernel itself on
adversarial inputs, and one must reproduce it bit for bit. Evidence level "known": the order
and every operation are read from the lowered IR, the two compiler conventions are corroborated
on the device.

Supported TIR (what `dlight.gpu.Fallback` produces): loops bound to GPU threads around blocks
and serial loops, float32 buffers, integer index arithmetic, `+ - * /`, `sqrt`, `max`, `min`,
comparisons, `select` / `if_then_else`, casts between integers. GPU-thread loops run as one
tensor operation over all their iterations; that is the same as running them one by one only
if no iteration reads what another wrote, which is checked: within one thread region a block
writes distinct elements, and a read of a buffer written in the region is the block's own
element. Anything else (tiled schedules with shared-memory buffers, vectorized loops,
transcendental intrinsics, other dtypes) raises `LoweringFailed`.
"""
from __future__ import annotations

import operator
from typing import Callable, Dict, List, Optional

from . import primitives as P
from .lowering import KINDS, LoweredOp, Lowering, LoweringFailed, adversarial_tensor

NAMESPACE = "dlclv.tir_lowering"


def _tir():
    try:
        import tvm
    except ImportError:
        import tilelang  # noqa: F401  (bundles TVM)
        import tvm
    return tvm, (tvm.tirx if hasattr(tvm, "tirx") else tvm.tir)


# What `max(a, b)` / `min(a, b)` return when a == b, i.e. for +0 against -0: the first operand,
# the second, or by the sign of zero (+0 above -0, as IEEE 754-2019 maximum/minimum and CUDA's
# fmaxf/fminf do).
TIES = ("zero-sign", "first", "second")


class Model:
    """The two conventions the TIR leaves to the CUDA compiler."""

    def __init__(self, contract: bool = True, tie: str = "zero-sign"):
        self.contract, self.tie = contract, tie

    def __str__(self):
        return (("a*b+c contracted to one FMA" if self.contract else "no FMA contraction")
                + ("; max/min order +0 above -0" if self.tie == "zero-sign" else
                   f"; max/min return their {self.tie} operand on a tie"))


# --------------------------------------------------------------------------------------------
# Static facts about one PrimFunc
# --------------------------------------------------------------------------------------------

def _children(node):
    """Sub-expressions and sub-statements of a TIR node (generic walk)."""
    out = []
    for k in ("a", "b", "value", "condition", "true_value", "false_value", "body", "block",
              "predicate", "then_case", "else_case", "min", "extent"):
        v = getattr(node, k, None)
        if v is not None and not isinstance(v, (int, float, str, bool)):
            out.append(v)
    for k in ("args", "indices", "seq", "iter_values"):
        v = getattr(node, k, None)
        if v is not None:
            out += list(v)
    return out


def analyse(func) -> dict:
    """Which model knobs matter for this function, and a short summary of its loops."""
    _, T = _tir()
    facts = {"mul_add": False, "max_min": False, "blocks": [], "serial_loops": []}
    stack = [func.body]
    while stack:
        n = stack.pop()
        name = type(n).__name__
        if name in ("Add", "Sub") and "float" in str(n.dtype):
            if type(n.a).__name__ == "Mul" or type(n.b).__name__ == "Mul":
                facts["mul_add"] = True
        if name in ("Max", "Min") and "float" in str(n.dtype):
            facts["max_min"] = True
        if name == "SBlock" and n.name_hint != "root":
            facts["blocks"].append(str(n.name_hint))
        if name == "For" and n.kind == T.ForKind.SERIAL:
            facts["serial_loops"].append((str(n.loop_var.name), int(n.extent)))
        stack += _children(n)
    return facts


# --------------------------------------------------------------------------------------------
# The interpreter
# --------------------------------------------------------------------------------------------

class _Run:
    """One execution of a PrimFunc on torch tensors. `check=True` (used when the lowering is
    built, outside the primitive guard) also verifies that running the GPU-thread loops as
    tensor operations is the same as running their iterations one by one."""

    def __init__(self, func, tensors, model: Model, check: bool):
        import torch
        self.tvm, self.T = _tir()
        self.torch, self.model, self.check = torch, model, check
        self.func = func
        self.device = tensors[0].device
        self.buf = {}                      # buffer key -> tensor
        self.shape = {}
        self.buffers = []                  # (Buffer, key): two buffers may share a name
        if len(tensors) != len(func.params):
            raise LoweringFailed(f"{len(func.params)} buffers expected, {len(tensors)} given")
        for p, t in zip(func.params, tensors):
            b = func.buffer_map[p]
            self._declare(b, t)
        self.env = {}                      # var name -> python int or int64 tensor
        self.rank = 0                      # number of open GPU-thread loops
        self.region_writes = None          # buffers written in the current thread region
        self.block_store = None
        self.cache = {}

    def key(self, b, declare=False):
        for known, k in self.buffers:
            if known.same_as(b):
                return k
        if not declare:
            raise LoweringFailed(f"use of undeclared buffer {b.name}")
        k = f"{b.name}#{len(self.buffers)}"
        self.buffers.append((b, k))
        return k

    def _declare(self, b, t=None):
        torch = self.torch
        name, shape = self.key(b, declare=True), tuple(int(s) for s in b.shape)
        if str(b.dtype) != "float32":
            raise LoweringFailed(f"buffer {name} has dtype {b.dtype}: only float32 is modelled")
        if t is None:
            t = torch.full(shape, float("nan"), dtype=torch.float32, device=self.device)
        if tuple(t.shape) != shape or t.dtype != torch.float32:
            raise LoweringFailed(f"buffer {name}: expected float32{list(shape)}, got "
                                 f"{t.dtype}{list(t.shape)}")
        self.buf[name], self.shape[name] = t, shape

    # ---- statements ----

    def stmt(self, s, active):
        T, name = self.T, type(s).__name__
        if name == "SeqStmt":
            for q in s.seq:
                self.stmt(q, active)
        elif name == "For":
            self.loop(s, active)
        elif name == "SBlockRealize":
            self.block(s, active)
        elif name == "BufferStore":
            self.store(s, active)
        elif name == "IfThenElse" and s.else_case is None:
            c = self.expr(s.condition, active)
            self.stmt(s.then_case, c if active is None else self._and(active, c))
        elif name in ("DeclBuffer",):
            self._declare(s.buffer)
            if getattr(s, "body", None) is not None:
                self.stmt(s.body, active)
        else:
            raise LoweringFailed(f"TIR statement {name} is not modelled")

    def loop(self, s, active):
        T, torch = self.T, self.torch
        var, lo, n = str(s.loop_var.name), self._int(s.min), self._int(s.extent)
        step = getattr(s, "step", None)
        if step is not None and self._int(step) != 1:
            raise LoweringFailed(f"loop {var} has a step")
        if s.kind == T.ForKind.THREAD_BINDING:
            outer = self.rank == 0
            if outer:
                self.region_writes = self._writes(s)
                self.cache = {}
            idx = torch.arange(lo, lo + n, device=self.device)
            self.env[var] = idx.reshape((n,) + (1,) * 0) if self.rank == 0 else \
                idx.reshape((n,) + (1,) * self.rank)
            # earlier thread variables keep their axes: a new axis is added in front
            self.rank += 1
            try:
                self.stmt(s.body, active)
            finally:
                self.rank -= 1
                del self.env[var]
                if outer:
                    self.region_writes = None
                    self.cache = {}
        elif s.kind == T.ForKind.SERIAL:
            for i in range(lo, lo + n):
                self.env[var] = i
                self.stmt(s.body, active)
            del self.env[var]
        else:
            raise LoweringFailed(f"loop {var} of kind {s.kind} is not modelled (only GPU-thread "
                                 "and serial loops)")

    def _writes(self, s):
        out, stack = set(), [s]
        while stack:
            n = stack.pop()
            if type(n).__name__ == "BufferStore":
                out.add(self.key(n.buffer))
            stack += _children(n)
        return out

    def block(self, s, active):
        b = s.block
        if b.init is not None or len(b.match_buffers):
            raise LoweringFailed(f"block {b.name_hint}: init / match_buffer are not modelled")
        for ab in b.alloc_buffers:
            self._declare(ab)
        saved = dict(self.env)
        values = [self.expr(v, active) for v in s.iter_values]
        for iv, v in zip(b.iter_vars, values):
            self.env[str(iv.var.name)] = v
        pred = self.expr(s.predicate, active)
        if pred is not True:
            active = pred if active is None else self._and(active, pred)
        self.stmt(b.body, active)
        self.env = saved

    def _and(self, a, b):
        if a is True:
            return b
        if b is True:
            return a
        if a is False or b is False:
            return False
        return a & b

    def _flat(self, name, indices, active):
        """Row-major flat index (python int or int64 tensor) of buffer `name`."""
        shape = self.shape[name]
        if len(indices) != len(shape):
            raise LoweringFailed(f"{name}: {len(indices)} indices for rank {len(shape)}")
        flat, stride = 0, 1
        for i, d in zip(reversed(indices), reversed(shape)):
            flat = flat + i * stride
            stride *= d
        if self.check:
            torch = self.torch
            numel = stride
            f = flat if torch.is_tensor(flat) else torch.tensor(flat)
            ok = (f >= 0) & (f < numel)
            for i, d in zip(indices, shape):
                it = i if torch.is_tensor(i) else torch.tensor(i, device=self.device)
                ok = ok & (it >= 0) & (it < d)
            if active is not None and active is not True:
                ok = ok | ~active
            if active is not False and not bool(ok.all()):
                raise LoweringFailed(f"{name}: index out of bounds outside its guard")
        return flat

    def store(self, s, active):
        torch = self.torch
        name = self.key(s.buffer)
        if active is False:
            return
        saved = self.block_store
        self.block_store = s
        try:
            idx = [self.expr(i, active) for i in s.indices]
            val = self.expr(s.value, active)
        finally:
            self.block_store = saved
        flat = self._flat(name, idx, active)
        buf = self.buf[name]
        if not torch.is_tensor(val):
            val = torch.full((), float(val), dtype=buf.dtype, device=self.device)
        if val.dtype != buf.dtype:
            raise LoweringFailed(f"{name}: stored value has dtype {val.dtype}")
        tensors = [t for t in (flat, val, active) if torch.is_tensor(t)]
        grid = torch.broadcast_shapes(*[t.shape for t in tensors]) if tensors else ()
        if not torch.is_tensor(flat):
            flat = torch.full((), flat, dtype=torch.int64, device=self.device)
        flat, val = flat.expand(grid).reshape(-1), val.expand(grid).reshape(-1)
        if torch.is_tensor(active):
            keep = active.expand(grid).reshape(-1)
            flat, val = flat[keep], val[keep]
        if self.check and self.rank and flat.numel() != torch.unique(flat).numel():
            raise LoweringFailed(f"{name}: two GPU threads write the same element")
        self.buf[name] = buf.reshape(-1).index_put((flat,), val).reshape(buf.shape)

    # ---- expressions ----

    def _int(self, e):
        v = self.expr(e, None)
        if not isinstance(v, int):
            raise LoweringFailed("loop bounds must be constants")
        return v

    def expr(self, e, active):
        torch, name = self.torch, type(e).__name__
        if isinstance(e, bool):
            return e
        if isinstance(e, int):
            return int(e)
        if name == "IntImm":
            return bool(e.value) if str(e.dtype) == "bool" else int(e.value)
        if name == "FloatImm":
            if str(e.dtype) != "float32":
                raise LoweringFailed(f"float constant of dtype {e.dtype}")
            return torch.full((), float(e.value), dtype=torch.float32, device=self.device)
        if name in ("Var", "SizeVar"):
            if str(e.name) not in self.env:
                raise LoweringFailed(f"unbound variable {e.name}")
            return self.env[str(e.name)]
        if name == "BufferLoad":
            return self.load(e, active)
        if name in ("Select",) or (name == "Call" and str(e.op.name).split(".")[-1]
                                   == "if_then_else"):
            c, a, b = (e.condition, e.true_value, e.false_value) if name == "Select" else e.args
            cond = self.expr(c, active)
            if isinstance(cond, bool):
                return self.expr(a if cond else b, active)
            x = self.expr(a, cond if active is None else self._and(active, cond))
            y = self.expr(b, ~cond if active is None else self._and(active, ~cond))
            x, y = self._float(x), self._float(y)
            return torch.where(cond, x, y)
        if name == "Call":
            op = str(e.op.name).split(".")[-1]
            if op == "sqrt" and str(e.dtype) == "float32":
                return torch.sqrt(self._float(self.expr(e.args[0], active)))
            raise LoweringFailed(f"TIR intrinsic {e.op.name} is not an explicit primitive")
        if name == "Cast":
            v = self.expr(e.value, active)
            if "int" in str(e.dtype) and (isinstance(v, int) or not v.is_floating_point()):
                return v
            raise LoweringFailed(f"cast to {e.dtype} is not modelled")
        if name == "Not":
            v = self.expr(e.a, active)
            return (not v) if isinstance(v, bool) else ~v
        if name in ("And", "Or"):
            a, b = self.expr(e.a, active), self.expr(e.b, active)
            if name == "And":
                return self._and(a, b)
            if a is True or b is True:
                return True
            return b if a is False else (a if b is False else a | b)
        if not hasattr(e, "a") or not hasattr(e, "b"):
            raise LoweringFailed(f"TIR expression {name} is not modelled")
        is_float = "float" in str(e.a.dtype)
        if not is_float:
            return self.integer(name, self.expr(e.a, active), self.expr(e.b, active))
        if str(e.a.dtype) != "float32" or str(e.b.dtype) != "float32":
            raise LoweringFailed(f"{name} on {e.a.dtype}: only float32 arithmetic is modelled")
        if name in ("Add", "Sub") and self.model.contract:
            ma, mb = type(e.a).__name__ == "Mul", type(e.b).__name__ == "Mul"
            if ma and mb:
                raise LoweringFailed("a*b + c*d: which product the compiler fuses is not "
                                     "determined by the IR")
            if ma or mb:
                m, other = (e.a, e.b) if ma else (e.b, e.a)
                x = self._float(self.expr(m.a, active))
                y = self._float(self.expr(m.b, active))
                z = self._float(self.expr(other, active))
                if name == "Add":
                    return P.fma32(x, y, z)                    # x*y + z
                return P.fma32(x, y, -z) if ma else P.fma32(-x, y, z)   # x*y - z ; z - x*y
        a, b = self._float(self.expr(e.a, active)), self._float(self.expr(e.b, active))
        if name == "Add":
            return a + b
        if name == "Sub":
            return a - b
        if name == "Mul":
            return a * b
        if name == "Div":
            return a / b
        if name in ("Max", "Min"):
            tie = self.model.tie
            if tie == "zero-sign":
                # IEEE 754-2019 maximum/minimum: +0 is larger than -0, whatever the order.
                shape = torch.broadcast_shapes(a.shape, b.shape)
                a, b = a.expand(shape), b.expand(shape)
                negative = a.contiguous().view(torch.int32) < 0
                if name == "Max":
                    return torch.where(b > a, b, torch.where(a > b, a,
                                                             torch.where(negative, b, a)))
                return torch.where(b < a, b, torch.where(a < b, a,
                                                         torch.where(negative, a, b)))
            first = tie == "first"
            if name == "Max":
                return torch.where(b > a, b, a) if first else torch.where(a > b, a, b)
            return torch.where(b < a, b, a) if first else torch.where(a < b, a, b)
        cmp = {"LT": operator.lt, "LE": operator.le, "GT": operator.gt, "GE": operator.ge,
               "EQ": operator.eq, "NE": operator.ne}.get(name)
        if cmp is None:
            raise LoweringFailed(f"float {name} is not modelled")
        return cmp(a, b)

    def _float(self, v):
        torch = self.torch
        if torch.is_tensor(v) and v.dtype == torch.float32:
            return v
        raise LoweringFailed("expected a float32 value")

    def integer(self, name, a, b):
        torch = self.torch
        ints = isinstance(a, int) and isinstance(b, int)
        if name == "Add":
            return a + b
        if name == "Sub":
            return a - b
        if name == "Mul":
            return a * b
        if name in ("FloorDiv", "FloorMod", "Div", "Mod"):
            if name in ("Div", "Mod") and self.check:        # truncating: same for a >= 0, b > 0
                for v in (a, b):
                    neg = v < 0
                    if (neg if isinstance(neg, bool) else bool(neg.any())):
                        raise LoweringFailed("truncating division of a negative index")
            return a // b if name in ("FloorDiv", "Div") else a % b
        if name in ("Max", "Min"):
            if ints:
                return max(a, b) if name == "Max" else min(a, b)
            a = a if torch.is_tensor(a) else torch.full((), a, dtype=torch.int64,
                                                        device=self.device)
            return torch.where((b > a) if name == "Max" else (b < a), b + a * 0, a + b * 0)
        cmp = {"LT": operator.lt, "LE": operator.le, "GT": operator.gt, "GE": operator.ge,
               "EQ": operator.eq, "NE": operator.ne}.get(name)
        if cmp is None:
            raise LoweringFailed(f"integer {name} is not modelled")
        if torch.is_tensor(a) or not torch.is_tensor(b):
            return cmp(a, b)
        swapped = {"LT": operator.gt, "LE": operator.ge, "GT": operator.lt, "GE": operator.le,
                   "EQ": operator.eq, "NE": operator.ne}[name]
        return swapped(b, a)

    def load(self, e, active):
        torch = self.torch
        name = self.key(e.buffer)
        if self.check and self.region_writes is not None and name in self.region_writes:
            st = self.block_store
            own = st is not None and st.buffer.same_as(e.buffer) and \
                len(st.indices) == len(e.indices) and all(
                    self.tvm.ir.structural_equal(i, j) for i, j in zip(st.indices, e.indices))
            if not own:
                raise LoweringFailed(
                    f"{name} is written in this GPU-thread region and read at another element: "
                    "iterations are not independent")
        idx = [self.expr(i, active) for i in e.indices]
        flat = self._flat(name, idx, active)
        buf = self.buf[name].reshape(-1)
        if not torch.is_tensor(flat):
            if active is False:
                return torch.zeros((), dtype=buf.dtype, device=self.device)
            return buf[flat]
        if torch.is_tensor(active):                       # guarded: read element 0 instead
            flat = torch.where(active, flat, torch.zeros_like(flat))
        elif active is False:
            flat = torch.zeros_like(flat)
        return buf[flat]


def run_primfunc(func, inputs, n_out: int, model: Model, check: bool = False):
    """Execute a scheduled TIR function: `inputs` are its leading buffers, the last `n_out`
    are outputs (allocated here). Returns the output tensor(s)."""
    import torch
    params = list(func.params)
    if len(inputs) + n_out != len(params):
        raise LoweringFailed(f"{len(params)} buffers, {len(inputs)} inputs and {n_out} outputs")
    device = inputs[0].device if inputs else None
    outs = []
    for p in params[len(inputs):]:
        b = func.buffer_map[p]
        outs.append(torch.full(tuple(int(s) for s in b.shape), float("nan"),
                               dtype=torch.float32, device=device or _DEVICE[0]))
    run = _Run(func, list(inputs) + outs, model, check)
    body = func.body
    if type(body).__name__ == "SBlockRealize" and str(body.block.name_hint) == "root":
        for ab in body.block.alloc_buffers:
            run._declare(ab)
        body = body.block.body
    run.stmt(body, None)
    res = [run.buf[run.key(func.buffer_map[p])] for p in params[len(inputs):]]
    return res[0] if n_out == 1 else tuple(res)


_DEVICE: List[object] = [None]


# --------------------------------------------------------------------------------------------
# Relax main -> ordered graph of call_tir nodes
# --------------------------------------------------------------------------------------------

def build_tvm_graph(compiled, example_inputs):
    """FX graph (the SSA container the frontend uses) of the lowered Relax `main`: one
    placeholder per input, one buffer per constant, one call per `call_tir`. The call targets
    run the compiled TIR kernel; the reconstruction replaces every one of them."""
    import torch
    tvm, T = _tir()
    from tvm import relax
    meta = compiled.meta or {}
    lowered, tir_call = meta.get("tvm_lowered"), meta.get("tir_call")
    if lowered is None or tir_call is None:
        raise LoweringFailed("compiled tvm program has no lowered module / kernel entry point")
    device = example_inputs[0].device
    _DEVICE[0] = device
    main = lowered["main"]
    graph, root = torch.fx.Graph(), torch.nn.Module()
    env: Dict[object, object] = {}
    if len(main.params) != len(example_inputs):
        raise LoweringFailed("Relax main and the example disagree on the number of inputs")
    for p in main.params:
        env[p] = graph.placeholder(str(p.name_hint))
    consts = []

    def value(e):
        if isinstance(e, relax.Var):
            if e not in env:
                raise LoweringFailed(f"Relax variable {e.name_hint} used before its binding")
            return env[e]
        if isinstance(e, relax.Constant):
            t = torch.from_numpy(e.data.numpy().copy())
            name = f"const_{len(consts)}"
            root.register_buffer(name, t.to(device))
            consts.append(name)
            return graph.get_attr(name)
        if isinstance(e, relax.Tuple):
            return tuple(value(f) for f in e.fields)
        if isinstance(e, relax.TupleGetItem):
            v = value(e.tuple_value)
            if isinstance(v, tuple):
                return v[int(e.index)]
            return graph.call_function(operator.getitem, (v, int(e.index)))
        if isinstance(e, relax.Call) and getattr(e.op, "name", None) == "relax.call_tir":
            if len(e.args) != 2:
                raise LoweringFailed("call_tir with packed shape arguments is not modelled")
            gv = e.args[0]
            name, func = str(gv.name_hint), lowered[gv]
            args = value(e.args[1])
            sinfo = e.sinfo_args[0]
            n_out = len(sinfo.fields) if isinstance(sinfo, relax.TupleStructInfo) else 1
            if any(isinstance(a, tuple) for a in args):
                raise LoweringFailed(f"{name}: tuple argument")
            return graph.call_function(tir_target(name, func, n_out, tir_call, device), args)
        raise LoweringFailed(f"Relax expression {type(e).__name__} "
                             f"{getattr(getattr(e, 'op', None), 'name', '')} is not a call_tir")

    body = main.body
    for block in body.blocks:
        for b in block.bindings:
            if type(b).__name__ != "VarBinding":
                raise LoweringFailed(f"Relax binding {type(b).__name__} is not modelled")
            env[b.var] = value(b.value)
    out = value(body.body)
    if isinstance(out, tuple) and len(out) == 1:
        out = out[0]
    graph.output(out)
    return torch.fx.GraphModule(root, graph)


def tir_target(name, func, n_out, tir_call, device):
    """Call target of one `call_tir`: runs the compiled kernel (never used by a finished
    reconstruction, which lowers every such node; it is what the lowering validates against)."""
    import torch

    def kernel(*inputs):
        outs = [torch.full(tuple(int(s) for s in func.buffer_map[p].shape), float("nan"),
                           dtype=torch.float32, device=device)
                for p in list(func.params)[len(inputs):]]
        tir_call(name, [x.contiguous() for x in inputs] + outs)
        return outs[0] if n_out == 1 else tuple(outs)

    kernel.__name__ = kernel.__qualname__ = f"tir_{name}"
    kernel.__module__ = NAMESPACE
    kernel.tir = (name, func, n_out)
    return kernel


# --------------------------------------------------------------------------------------------
# The lowering
# --------------------------------------------------------------------------------------------

def same_bits(a, b) -> bool:
    """Bitwise equal where neither is NaN, and NaN at the same places (NaN sign and payload
    are outside every claim)."""
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    na, nb = torch.isnan(a), torch.isnan(b)
    if bool((na != nb).any()):
        return False
    diff = ((a != b) | (torch.signbit(a) != torch.signbit(b))) & ~na
    return not bool(diff.any())


class TirLowering(Lowering):
    """Every `call_tir` of a `compilers.compile_tvm` program, executed from its scheduled TIR."""

    name = "tvm-tir"

    def __init__(self, compiled, trials: int = 5, seed: int = 0):
        self.compiled, self.trials, self.seed = compiled, trials, seed
        self.failures: Dict[str, str] = {}
        self._cache: Dict[tuple, tuple] = {}
        self.options = dict(compiled.options)
        self._tie = None                   # tie convention a kernel of this program fixed

    def _select(self, name, func, n_out, kernel, args):
        """The model under which the interpreter reproduces the compiled kernel."""
        import torch
        facts = analyse(func)
        models = [Model(c, t) for c in ((True, False) if facts["mul_add"] else (True,))
                  for t in (TIES if facts["max_min"] else TIES[:1])]
        gen = torch.Generator().manual_seed(self.seed)
        cases = [list(args)]
        for t in range(self.trials if args else 0):
            cases.append([adversarial_tensor(tuple(a.shape), KINDS[t % len(KINDS)], gen,
                                             a.dtype, a.device) for a in args])
        if args and facts["max_min"]:
            # Ties between +0 and -0 in every position: zeros of random sign, all -0, all +0.
            mixed = [torch.where(torch.rand(a.shape, generator=gen) < 0.5, -0.0, 0.0)
                     .to(a.dtype).to(a.device) for a in args]
            cases += [mixed, [torch.full_like(a, -0.0) for a in args],
                      [torch.zeros_like(a) for a in args]]
        alive, first_error = list(models), None
        for i, case in enumerate(cases):
            with torch.no_grad():
                want = kernel(*case)
            want = want if isinstance(want, tuple) else (want,)
            keep = []
            for m in alive:
                try:
                    with torch.no_grad():
                        got = run_primfunc(func, case, n_out, m, check=(i == 0))
                except LoweringFailed as e:
                    first_error = first_error or str(e)
                    continue
                got = got if isinstance(got, tuple) else (got,)
                if all(same_bits(g, w) for g, w in zip(got, want)):
                    keep.append(m)
                elif first_error is None and len(alive) == 1:
                    bad = sum(int(((g != w) | (torch.signbit(g) != torch.signbit(w))).sum())
                              for g, w in zip(got, want))
                    first_error = (f"the interpreted TIR ({m}) differs from the compiled "
                                   f"kernel in {bad} elements on validation input {i}")
            alive = keep
            if not alive:
                raise LoweringFailed(f"TIR function {name}: "
                                     + (first_error or "no convention reproduces the kernel"))
        return alive, facts, len(cases)

    def lower(self, node_name, target, args, kwargs):
        info = getattr(target, "tir", None)
        if info is None:
            return None
        name, func, n_out = info
        key = (name,)
        if key not in self._cache:
            self._cache[key] = self._select(name, func, n_out, target, list(args))
        alive, facts, n_inputs = self._cache[key]
        ties = {m.tie for m in alive}
        if facts["max_min"] and len(ties) == 1:
            self._tie = self._tie or next(iter(ties))
        # A kernel whose inputs never tie takes the convention the program's other kernels
        # showed (one compiler): it is still one of its own surviving candidates.
        model = next((m for m in alive if m.tie == self._tie), alive[0])
        alive = [model] + [m for m in alive if m is not model]
        loops = ", ".join(f"{v} < {n}" for v, n in facts["serial_loops"])
        desc = (f"blocks {facts['blocks']}"
                + (f"; serial reduction loops in this order: {loops}, one float32 accumulator "
                   "per output element" if loops else "; no reduction")
                + f"; {model}")
        ident = self.compiled.meta["tvm_kernel_source_identical"]()

        def fn(*inputs):
            return run_primfunc(func, list(inputs), n_out, model)

        return LoweredOp(
            fn, f"TVM TIR function {name} ({self.options})", desc, "known",
            {"static": "scheduled TIR of the lowered module (compiled.meta['tvm_lowered']"
                       f"['{name}'].script())",
             "tir_blocks": facts["blocks"], "serial_loops": facts["serial_loops"],
             "model": str(model), "indistinguishable_models": [str(m) for m in alive[1:]],
             "validation_inputs": n_inputs,
             "kernel_source_identical_to_program": ident},
            ["TVM's code generator and nvcc implement each scheduled TIR function as written: "
             "the statements of one GPU thread in program order, float32 operations as IEEE "
             "operations, serial loops in increasing order.",
             f"Compiler conventions the TIR does not fix ({model}) were selected by bitwise "
             f"validation against the compiled kernel on {n_inputs} inputs per function "
             "(dynamic evidence for these two conventions only).",
             "The kernel validated against is the same TIR function built on its own with the "
             "same code generator and nvcc"
             + (" (its CUDA source is verbatim the program's)." if ident else
                "; its CUDA source could not be matched to the program's text.")])
