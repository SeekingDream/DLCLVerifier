"""Framework-neutral dataflow graph for tensor programs.

The same `TensorGraph` class holds two levels:

* a *source graph*, built by a frontend (`frontends.py`), whose nodes are the operations the
  program actually runs: ATen ops, ONNX ops, or fused kernels such as `com.microsoft::FusedConv`;
* a *canonical graph*, built by `canonical.py`, whose nodes are logical operations. Each
  canonical node records the source nodes it came from in `meta["origins"]`.

Values are tensors with a static shape and dtype. Constants (weights and anything computed only
from shapes and constants) carry their data.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class GraphError(ValueError):
    pass


@dataclass
class Value:
    id: str
    shape: tuple
    dtype: str                       # numpy dtype name, e.g. "float32", "int64", "bool"
    kind: str = "intermediate"       # input | const | intermediate
    data: Optional[np.ndarray] = None
    input_index: Optional[int] = None
    producer: Optional[str] = None
    name: str = ""                   # name in the source program
    origin: Optional[str] = None     # canonical graphs: the source value this one stands for
    exact: Any = None                # canonical scalar consts: exact rational value (Fraction)

    @property
    def numel(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def describe(self):
        return f"{self.id}:{self.dtype}{list(self.shape)}"


@dataclass
class Node:
    id: str
    op: str
    inputs: List[Optional[str]]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)
    source: str = ""                 # operation name in the source program
    meta: Dict[str, Any] = field(default_factory=dict)


class TensorGraph:
    def __init__(self, name: str = "", frontend: str = ""):
        self.name = name
        self.frontend = frontend
        self.values: Dict[str, Value] = {}
        self.nodes: Dict[str, Node] = {}       # insertion order is a topological order
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.meta: Dict[str, Any] = {}
        self._consumers = None
        self._counter = collections.Counter()

    # ---- construction ----

    def fresh(self, prefix: str) -> str:
        while True:
            self._counter[prefix] += 1
            vid = f"{prefix}{self._counter[prefix]}"
            if vid not in self.values and vid not in self.nodes:
                return vid

    def add_value(self, v: Value) -> Value:
        if v.id in self.values:
            raise GraphError(f"duplicate value {v.id}")
        v.shape = tuple(int(d) for d in v.shape)
        self.values[v.id] = v
        if v.kind == "input":
            if v.input_index is None:
                v.input_index = len(self.inputs)
            self.inputs.append(v.id)
        self._consumers = None
        return v

    def add_const(self, data, name: str = "", vid: Optional[str] = None, origin=None,
                  exact=None) -> str:
        data = np.asarray(data)
        vid = vid or self.fresh("c")
        self.add_value(Value(vid, data.shape, data.dtype.name, "const", data=data, name=name,
                             origin=origin, exact=exact))
        return vid

    def add_node(self, n: Node) -> Node:
        if n.id in self.nodes:
            raise GraphError(f"duplicate node {n.id}")
        for i in n.inputs:
            if i is not None and i not in self.values:
                raise GraphError(f"node {n.id} ({n.op}) uses undefined value {i}")
        for o in n.outputs:
            v = self.values.get(o)
            if v is None:
                raise GraphError(f"node {n.id} output {o} has no Value")
            if v.producer is not None:
                raise GraphError(f"value {o} produced twice")
            v.producer = n.id
        self.nodes[n.id] = n
        self._consumers = None
        return n

    def remove_node(self, nid: str):
        n = self.nodes.pop(nid)
        for o in n.outputs:
            self.values[o].producer = None
        self._consumers = None
        return n

    # ---- queries ----

    def producer(self, vid: str) -> Optional[Node]:
        p = self.values[vid].producer
        return self.nodes[p] if p is not None else None

    def consumers(self, vid: str) -> List[Node]:
        if self._consumers is None:
            cons = collections.defaultdict(list)
            for n in self.nodes.values():
                for i in dict.fromkeys(x for x in n.inputs if x is not None):
                    cons[i].append(n)
            self._consumers = cons
        return self._consumers.get(vid, [])

    def is_const(self, vid: str) -> bool:
        return self.values[vid].kind == "const"

    def op_histogram(self) -> Dict[str, int]:
        return dict(collections.Counter(n.op for n in self.nodes.values()))

    def topo_order(self) -> List[Node]:
        order, state = [], {}

        def visit(n):
            if state.get(n.id) == 2:
                return
            if state.get(n.id) == 1:
                raise GraphError(f"cycle at node {n.id}")
            state[n.id] = 1
            for i in n.inputs:
                p = self.producer(i) if i is not None else None
                if p is not None:
                    visit(p)
            state[n.id] = 2
            order.append(n)

        for n in list(self.nodes.values()):
            visit(n)
        return order

    def reorder(self):
        """Rebuild insertion order as a topological order after rewrites."""
        self.nodes = {n.id: n for n in self.topo_order()}

    def dead_code_elimination(self) -> List[Node]:
        live = set(self.outputs)
        keep = []
        for n in reversed(self.topo_order()):
            if any(o in live for o in n.outputs):
                keep.append(n.id)
                live.update(i for i in n.inputs if i is not None)
        removed = [self.remove_node(nid) for nid in list(self.nodes) if nid not in set(keep)]
        for vid in [v for v, val in self.values.items()
                    if v not in live and val.producer is None and val.kind != "input"]:
            del self.values[vid]
        return removed

    def validate(self):
        seen = set(self.inputs) | {v for v, val in self.values.items() if val.kind == "const"}
        for n in self.nodes.values():
            for i in n.inputs:
                if i is not None and i not in seen:
                    raise GraphError(f"node {n.id} ({n.op}) reads {i} before it is defined")
            seen.update(n.outputs)
        for o in self.outputs:
            if o not in seen:
                raise GraphError(f"graph output {o} is never defined")
        for v in self.values.values():
            if v.kind == "const" and v.data is None:
                raise GraphError(f"const {v.id} has no data")
            if v.kind == "const" and tuple(v.data.shape) != v.shape:
                raise GraphError(f"const {v.id} data shape {v.data.shape} != {v.shape}")

    def summary(self) -> str:
        ins = ", ".join(self.values[i].describe() for i in self.inputs)
        outs = ", ".join(self.values[o].describe() for o in self.outputs)
        nconst = sum(v.kind == "const" for v in self.values.values())
        hist = sorted(self.op_histogram().items(), key=lambda kv: -kv[1])
        ops = ", ".join(f"{k}x{v}" for k, v in hist[:12]) + (" ..." if len(hist) > 12 else "")
        return (f"TensorGraph {self.name!r} [{self.frontend}]: {len(self.nodes)} nodes, "
                f"{nconst} consts\n  inputs:  {ins}\n  outputs: {outs}\n  ops: {ops}")

    def __repr__(self):
        return f"<TensorGraph {self.name!r} {self.frontend} nodes={len(self.nodes)}>"
