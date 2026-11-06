"""Match corresponding computational units of two tensor programs.

Both source graphs are lowered to canonical graphs (`canonical.py`). Every canonical value gets
a structural signature: graph inputs by position, constants by exact content, node outputs by
(op, attributes, shape, dtype, input signatures). Signatures ignore floating-point rounding but
include every attribute and shape that changes the logical computation. Operands of commutative
ops are sorted, and `add` / `mul` chains are n-ary, so a reassociated sum has the same signature.

Two canonical nodes match when their output signatures are equal. A *unit* is a connected group
of source operations that only match together: the source ops of one program that share
canonical nodes with source ops of the other, closed transitively. An add followed by a ReLU in
one program and a fused add-ReLU kernel in the other become one unit with two source ops on one
side and one on the other. A unit is valid only if every canonical node in it is matched and its
boundary inputs and outputs map one-to-one.

Anything else is reported. An unmatched region (for example a Conv whose weights absorbed a
BatchNorm) is paired with the other program's region when both start from corresponding values
and their outputs feed the same kind of consumer (earliest candidates first, recorded as a
`cut_position` ambiguity when several exist), and a cut is kept only if a consumer then matches.
The outputs are used as cut points so that matching continues downstream. A cut is never a
match: a result with any unmatched region is incomplete and cannot be certified.
Identical computations repeated in one program and computed once in the other (CSE) map to the
single counterpart; unequal repeat counts on both sides are reported as ambiguous.
"""
from __future__ import annotations

import collections
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .canonical import COMMUTATIVE, NARY, _attr_key, canonicalize, exact_scalar
from .graph import GraphError, TensorGraph


class MatchError(GraphError):
    pass


# --------------------------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------------------------

def _h(obj) -> str:
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:20]


def signatures(k: TensorGraph, cuts: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    cuts = cuts or {}
    sig: Dict[str, str] = {}
    for vid, v in k.values.items():
        if vid in cuts:
            sig[vid] = cuts[vid]
        elif v.kind == "input":
            sig[vid] = _h(("in", v.input_index, v.shape, v.dtype))
        elif v.kind == "const":
            ex = exact_scalar(v)
            if ex is not None:
                sig[vid] = _h(("scalar", np.dtype(v.dtype).kind, str(ex)))
            else:
                data = np.ascontiguousarray(v.data)
                sig[vid] = _h(("const", v.dtype, v.shape, hashlib.sha1(data.tobytes()).hexdigest()))
    for n in k.topo_order():
        ins = [sig[i] if i is not None else "none" for i in n.inputs]
        if n.op in COMMUTATIVE or n.op in NARY:
            ins = sorted(ins)
        attrs = _attr_key(n.attrs)
        for slot, o in enumerate(n.outputs):
            if o in cuts:
                sig[o] = cuts[o]
                continue
            v = k.values[o]
            sig[o] = _h((n.op, attrs, slot, v.shape, v.dtype, tuple(ins)))
    return sig


def node_signature(k, n, sig):
    return tuple(sig[o] for o in n.outputs)


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------

@dataclass
class BoundaryPair:
    value_1: str
    value_2: str
    shape: tuple
    dtype: str
    kind: str                         # input | const | intermediate
    source_1: Optional[str] = None    # name of the value in source program 1, if it has one
    source_2: Optional[str] = None


@dataclass
class MatchedUnit:
    id: str
    kind: str                         # one_to_one | fusion | fission | many_to_many
    source_nodes_1: List[str]
    source_nodes_2: List[str]
    ops_1: List[str]
    ops_2: List[str]
    canonical_nodes_1: List[str]
    canonical_nodes_2: List[str]
    logical_ops: List[str]
    inputs: List[BoundaryPair]
    outputs: List[BoundaryPair]
    impl_1: List[dict] = field(default_factory=list)
    impl_2: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class UnmatchedRegion:
    reason: str
    source_nodes_1: List[str]
    source_nodes_2: List[str]
    ops_1: List[str]
    ops_2: List[str]
    canonical_nodes_1: List[str]
    canonical_nodes_2: List[str]
    inputs_1: List[str]
    inputs_2: List[str]
    outputs_1: List[str]
    outputs_2: List[str]
    output_cuts: List[BoundaryPair] = field(default_factory=list)


@dataclass
class Ambiguity:
    kind: str                         # duplicate_computation | cut_position
    detail: str
    values_1: List[str]
    values_2: List[str]
    resolution: str


@dataclass
class MatchResult:
    graph_1: TensorGraph
    graph_2: TensorGraph
    canonical_1: TensorGraph
    canonical_2: TensorGraph
    units: List[MatchedUnit]
    unmatched: List[UnmatchedRegion]
    ambiguities: List[Ambiguity]
    input_pairs: List[BoundaryPair]
    output_pairs: List[Tuple[BoundaryPair, bool]]
    elided_1: List[str]
    elided_2: List[str]
    duplicate_groups: int = 0

    @property
    def complete(self) -> bool:
        """Every op is in a valid unit and every output corresponds. Says nothing numerical."""
        return not self.unmatched and all(ok for _, ok in self.output_pairs)

    def unit_of_value(self, side: int, value: str) -> Optional[MatchedUnit]:
        for u in self.units:
            for b in u.outputs:
                if (b.value_1 if side == 1 else b.value_2) == value:
                    return u
        return None

    def stats(self) -> dict:
        kinds = collections.Counter(u.kind for u in self.units)
        return {"units": len(self.units), "unit_kinds": dict(kinds),
                "unmatched_regions": len(self.unmatched),
                "unmatched_source_ops_1": sum(len(r.source_nodes_1) for r in self.unmatched),
                "unmatched_source_ops_2": sum(len(r.source_nodes_2) for r in self.unmatched),
                "ambiguities": len(self.ambiguities),
                "duplicate_groups": self.duplicate_groups,
                "outputs_matched": sum(ok for _, ok in self.output_pairs),
                "outputs": len(self.output_pairs),
                "source_ops_1": len(self.graph_1.nodes), "source_ops_2": len(self.graph_2.nodes),
                "elided_1": len(self.elided_1), "elided_2": len(self.elided_2),
                "complete": self.complete}

    def summary(self, max_items: int = 8) -> str:
        s = self.stats()
        lines = [f"match {self.graph_1.name!r} [{self.graph_1.frontend}] vs "
                 f"{self.graph_2.name!r} [{self.graph_2.frontend}]: "
                 f"{'COMPLETE' if self.complete else 'INCOMPLETE'}",
                 f"  units: {s['units']} {s['unit_kinds']}",
                 f"  outputs matched: {s['outputs_matched']}/{s['outputs']}",
                 f"  unmatched regions: {s['unmatched_regions']} "
                 f"({s['unmatched_source_ops_1']} / {s['unmatched_source_ops_2']} source ops)",
                 f"  ambiguities: {s['ambiguities']}; elided no-op source ops: "
                 f"{s['elided_1']} / {s['elided_2']}"]
        fused = [u for u in self.units if u.kind != "one_to_one"]
        for u in fused[:max_items]:
            lines.append(f"  {u.kind:<12} {'+'.join(u.ops_1)}  <->  {'+'.join(u.ops_2)}")
        for r in self.unmatched[:max_items]:
            lines.append(f"  UNMATCHED    {'+'.join(r.ops_1) or '-'}  <->  "
                         f"{'+'.join(r.ops_2) or '-'}: {r.reason}")
        for a in self.ambiguities[:max_items]:
            lines.append(f"  AMBIGUOUS    {a.kind}: {a.detail} ({a.resolution})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"stats": self.stats(),
                "units": [asdict(u) for u in self.units],
                "unmatched": [asdict(r) for r in self.unmatched],
                "ambiguities": [asdict(a) for a in self.ambiguities],
                "input_pairs": [asdict(b) for b in self.input_pairs],
                "output_pairs": [dict(asdict(b), matched=ok) for b, ok in self.output_pairs],
                "elided_1": self.elided_1, "elided_2": self.elided_2}


# --------------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------------

class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _pair(k1, k2, v1, v2) -> BoundaryPair:
    a, b = k1.values[v1], k2.values[v2]
    return BoundaryPair(v1, v2, a.shape, a.dtype, a.kind, a.origin or a.name or None,
                        b.origin or b.name or None)


def _anchors(k, sig, other_sigs, matched_cache):
    """For each unmatched value: the matched signatures its unmatched upstream region reads."""
    memo = {}

    def anchors(v):
        if v in memo:
            return memo[v]
        memo[v] = frozenset()
        p = k.producer(v)
        acc = set()
        if p is not None:
            for i in p.inputs:
                if i is None:
                    continue
                if sig[i] in other_sigs:
                    acc.add(sig[i])
                elif k.values[i].kind == "const":
                    continue
                else:
                    acc |= anchors(i)
        memo[v] = frozenset(acc)
        return memo[v]
    return anchors


def _cut_candidates(k1, k2, s1, s2):
    """Pairs (v1, v2) of unmatched values that could be the two ends of one unmatched region:
    same shape and dtype, same matched upstream anchors, same consumer ops, unique both ways.
    Also returns the ambiguous groups."""
    set1, set2 = set(s1.values()), set(s2.values())
    a1, a2 = _anchors(k1, s1, set2, None), _anchors(k2, s2, set1, None)

    def keys(k, s, other, anchors):
        out = collections.defaultdict(list)
        for n in k.nodes.values():
            for o in n.outputs:
                if s[o] in other or o in k.outputs:
                    continue
                cons = k.consumers(o)
                if not cons:
                    continue
                cops = tuple(sorted((c.op, repr(_attr_key(c.attrs))) for c in cons))
                v = k.values[o]
                anc = anchors(o)
                if anc:
                    out[(v.shape, v.dtype, anc, cops)].append(o)
        return out

    kk1, kk2 = keys(k1, s1, set2, a1), keys(k2, s2, set1, a2)
    pos1 = {o: i for i, n in enumerate(k1.nodes.values()) for o in n.outputs}
    pos2 = {o: i for i, n in enumerate(k2.nodes.values()) for o in n.outputs}
    pairs, ambiguous = [], []
    for key in kk1.keys() & kk2.keys():
        c1 = sorted(kk1[key], key=pos1.get)
        c2 = sorted(kk2[key], key=pos2.get)
        # Several candidates share a key when a long unmatched stretch hangs off one matched
        # value. Cuts are tried upstream first, so the earliest pair is the region boundary;
        # later ones are retried after it is cut.
        pairs.append((c1[0], c2[0], len(c1) > 1 or len(c2) > 1))
        if len(c1) > 1 or len(c2) > 1:
            ambiguous.append((key, c1, c2))
    pairs.sort(key=lambda p: pos1[p[0]])
    return pairs, ambiguous


def _find_cuts(k1, k2, max_trials=10_000):
    """Choose cut pairs one at a time, upstream first, keeping a cut only if a consumer of the
    cut value then matches. Returns (cuts1, cuts2, ambiguous groups)."""
    cuts1, cuts2, tried, n, positional = {}, {}, set(), 0, []
    s1, s2 = signatures(k1), signatures(k2)
    for _ in range(max_trials):
        pairs, _ = _cut_candidates(k1, k2, s1, s2)
        pairs = [p for p in pairs if p[:2] not in tried]
        if not pairs:
            break
        accepted = False
        for v1, v2, by_position in pairs:
            tried.add((v1, v2))
            tag = f"cut:{n + 1}"
            t1, t2 = dict(cuts1, **{v1: tag}), dict(cuts2, **{v2: tag})
            n1, n2 = signatures(k1, t1), signatures(k2, t2)
            other = set(n2.values())
            if any(n1[o] in other for c in k1.consumers(v1) for o in c.outputs):
                cuts1, cuts2, s1, s2, n = t1, t2, n1, n2, n + 1
                if by_position:
                    positional.append((v1, v2))
                accepted = True
                break
        if not accepted:
            break
    return cuts1, cuts2, positional


def match_graphs(g1: TensorGraph, g2: TensorGraph, fold_constants=True, normalize_arith=True,
                 flatten=True, recognize_idioms=True, max_cut_trials=10_000) -> MatchResult:
    opts = dict(fold_constants=fold_constants, normalize_arith=normalize_arith, flatten=flatten,
                recognize_idioms=recognize_idioms)
    k1, k2 = canonicalize(g1, **opts), canonicalize(g2, **opts)
    if len(k1.inputs) != len(k2.inputs):
        raise MatchError(f"programs take {len(k1.inputs)} vs {len(k2.inputs)} inputs")
    input_pairs = []
    for a, b in zip(k1.inputs, k2.inputs):
        va, vb = k1.values[a], k2.values[b]
        if va.shape != vb.shape or va.dtype != vb.dtype:
            raise MatchError(f"input {va.input_index}: {va.describe()} vs {vb.describe()}")
        input_pairs.append(_pair(k1, k2, a, b))
    if len(k1.outputs) != len(k2.outputs):
        raise MatchError(f"programs return {len(k1.outputs)} vs {len(k2.outputs)} outputs")

    ambiguities: List[Ambiguity] = []
    cuts1, cuts2, positional = _find_cuts(k1, k2, max_cut_trials)
    for v1, v2 in positional:
        ambiguities.append(Ambiguity(
            "cut_position", f"unmatched region ending at {k1.values[v1].describe()} was paired "
            "with the earliest of several candidates", [v1], [v2],
            "paired by program order; the region stays unmatched"))
    s1, s2 = signatures(k1, cuts1), signatures(k2, cuts2)

    # ---- node pairs ----
    by1, by2 = collections.defaultdict(list), collections.defaultdict(list)
    for n in k1.topo_order():
        by1[node_signature(k1, n, s1)].append(n.id)
    for n in k2.topo_order():
        by2[node_signature(k2, n, s2)].append(n.id)
    pairs, duplicates = [], 0
    for key in by1.keys() & by2.keys():
        a, b = by1[key], by2[key]
        if any(o.startswith("cut:") for o in key):
            continue
        if len(a) != 1 or len(b) != 1:
            duplicates += 1
        if len(b) == 1 and len(a) > 1:
            # program 2 computes once what program 1 computes several times (CSE): every copy
            # corresponds to the single value, since all copies compute the same thing
            pairs.extend((x, b[0]) for x in a)
            continue
        if len(a) == 1 and len(b) > 1:
            pairs.extend((a[0], y) for y in b)
            continue
        pairs.extend(zip(a, b))
        if len(a) != len(b):
            op = k1.nodes[a[0]].op
            ambiguities.append(Ambiguity(
                "duplicate_computation", f"{len(a)} x {len(b)} {op} nodes compute the same value",
                a, b, "paired in program order; extra nodes left unmatched"))
    m1 = {a: b for a, b in pairs}
    m2 = {b: a for a, b in pairs}
    # Value correspondence used for boundaries: through node pairs (so duplicates stay with
    # their own consumers), graph inputs by position, and cut points.
    vmap, rmap = {}, {}
    for a, b in pairs:
        if len(k1.nodes[a].outputs) != len(k2.nodes[b].outputs):
            continue
        vmap.update(zip(k1.nodes[a].outputs, k2.nodes[b].outputs))
        rmap.update(zip(k2.nodes[b].outputs, k1.nodes[a].outputs))
    vmap.update(zip(k1.inputs, k2.inputs))
    rmap.update(zip(k2.inputs, k1.inputs))
    inv_cut = {t: v for v, t in cuts2.items()}
    vmap.update({v: inv_cut[t] for v, t in cuts1.items() if t in inv_cut})
    rmap.update({inv_cut[t]: v for v, t in cuts1.items() if t in inv_cut})

    # ---- source-level components ----
    uf = _UF()
    for side, k in ((1, k1), (2, k2)):
        for n in k.nodes.values():
            orig = sorted(n.meta.get("origins", ()))
            for o in orig:
                uf.find((side, o))
            for o in orig[1:]:
                uf.union((side, orig[0]), (side, o))
    for a, b in pairs:
        oa = sorted(k1.nodes[a].meta["origins"])
        ob = sorted(k2.nodes[b].meta["origins"])
        uf.union((1, oa[0]), (2, ob[0]))
    comps = collections.defaultdict(lambda: ([], []))
    for side, k in ((1, k1), (2, k2)):
        for n in k.nodes.values():
            for o in n.meta.get("origins", ()):
                comp = comps[uf.find((side, o))][side - 1]
                if o not in comp:
                    comp.append(o)
    canon_of = {1: collections.defaultdict(list), 2: collections.defaultdict(list)}
    for side, k in ((1, k1), (2, k2)):
        for n in k.nodes.values():
            canon_of[side][uf.find((side, sorted(n.meta["origins"])[0]))].append(n.id)

    units, regions = [], []
    order1 = {nid: i for i, nid in enumerate(k1.nodes)}
    order2 = {nid: i for i, nid in enumerate(k2.nodes)}
    sorig1 = {nid: i for i, nid in enumerate(g1.nodes)}
    sorig2 = {nid: i for i, nid in enumerate(g2.nodes)}
    for root, (S1, S2) in comps.items():
        U1 = sorted(canon_of[1][root], key=order1.get)
        U2 = sorted(canon_of[2][root], key=order2.get)
        S1 = sorted(S1, key=sorig1.get)
        S2 = sorted(S2, key=sorig2.get)
        in1, out1 = _boundary(k1, U1)
        in2, out2 = _boundary(k2, U2)
        um1 = [n for n in U1 if n not in m1]
        um2 = [n for n in U2 if n not in m2]
        problems = []
        if um1 or um2 or not U1 or not U2:
            problems.append(_unmatched_reason(k1, k2, U1, U2, um1, um2))
        ins = outs = None
        if not problems:
            ins, bad_in = _map_boundary(k1, k2, in1, in2, s1, s2, vmap, rmap)
            outs, bad_out = _map_boundary(k1, k2, out1, out2, s1, s2, vmap, rmap)
            if bad_in or bad_out:
                problems.append("boundary values do not correspond one-to-one: "
                                + "; ".join(bad_in + bad_out))
        if problems:
            cuts = [_pair(k1, k2, a, b) for a in out1 for b in out2
                    if s1[a].startswith("cut:") and s1[a] == s2[b]]
            regions.append(UnmatchedRegion(
                problems[0], S1, S2, [g1.nodes[x].op for x in S1], [g2.nodes[x].op for x in S2],
                U1, U2, in1, in2, out1, out2, cuts))
            continue
        kind = ("one_to_one" if len(S1) == 1 and len(S2) == 1 else
                "fusion" if len(S2) == 1 else "fission" if len(S1) == 1 else "many_to_many")
        notes = []
        if any(k1.nodes[n].meta.get("approximate_constants") for n in U1) or \
                any(k2.nodes[n].meta.get("approximate_constants") for n in U2):
            notes.append("matched through an idiom whose constants agree only to 1e-6")
        units.append(MatchedUnit(
            "", kind, S1, S2, [g1.nodes[x].op for x in S1], [g2.nodes[x].op for x in S2],
            U1, U2, [k1.nodes[n].op for n in U1], ins, outs,
            [g1.nodes[x].meta.get("impl", g1.meta.get("impl", {})) for x in S1],
            [g2.nodes[x].meta.get("impl", g2.meta.get("impl", {})) for x in S2], notes))

    regions = _pair_regions(k1, k2, s1, s2, regions)
    units = _dependency_order(k1, units)
    for i, u in enumerate(units):
        u.id = f"u{i}"
    output_pairs = [(_pair(k1, k2, a, b), s1[a] == s2[b] and not s1[a].startswith("cut:"))
                    for a, b in zip(k1.outputs, k2.outputs)]
    return MatchResult(g1, g2, k1, k2, units, regions, ambiguities, input_pairs, output_pairs,
                       sorted(k1.meta.get("elided_origins", ()), key=sorig1.get),
                       sorted(k2.meta.get("elided_origins", ()), key=sorig2.get), duplicates)


def _pair_regions(k1, k2, s1, s2, regions):
    """Merge unmatched regions of the two programs that read corresponding values or share a
    cut, so each reported region shows both sides of the disagreement."""
    both = set(s1.values()) & set(s2.values())

    def keys(r):
        ks = {("in", s1[v]) for v in r.inputs_1 if s1[v] in both and not k1.is_const(v)}
        ks |= {("in", s2[v]) for v in r.inputs_2 if s2[v] in both and not k2.is_const(v)}
        ks |= {("cut", s1[v]) for v in r.outputs_1 if s1[v].startswith("cut:")}
        ks |= {("cut", s2[v]) for v in r.outputs_2 if s2[v].startswith("cut:")}
        return ks

    uf = _UF()
    owner = {}
    for i, r in enumerate(regions):
        uf.find(i)
        for key in keys(r):
            if key in owner:
                uf.union(i, owner[key])
            else:
                owner[key] = i
    groups = collections.defaultdict(list)
    for i in range(len(regions)):
        groups[uf.find(i)].append(regions[i])
    out = []
    for rs in groups.values():
        if len(rs) == 1:
            out.append(rs[0])
            continue
        cat = lambda attr: [x for r in rs for x in getattr(r, attr)]  # noqa: E731
        U1, U2 = cat("canonical_nodes_1"), cat("canonical_nodes_2")
        in1, out1 = _boundary(k1, U1)
        in2, out2 = _boundary(k2, U2)
        reason = _unmatched_reason(k1, k2, U1, U2, U1, U2)
        cuts = [_pair(k1, k2, a, b) for a in out1 for b in out2
                if s1[a].startswith("cut:") and s1[a] == s2[b]]
        out.append(UnmatchedRegion(reason, cat("source_nodes_1"), cat("source_nodes_2"),
                                   cat("ops_1"), cat("ops_2"), U1, U2, in1, in2, out1, out2,
                                   cuts))
    return out


def _boundary(k, U):
    inside = set(U)
    produced = {o for n in U for o in k.nodes[n].outputs}
    ins, outs = [], []
    for n in U:
        for i in k.nodes[n].inputs:
            if i is not None and i not in produced and i not in ins:
                ins.append(i)
    for n in U:
        for o in k.nodes[n].outputs:
            if o in k.outputs or any(c.id not in inside for c in k.consumers(o)):
                outs.append(o)
    return ins, outs


def _map_boundary(k1, k2, vs1, vs2, s1, s2, vmap, rmap):
    """Pair boundary values: computed values through the node pairing, constants by content
    (equal signatures mean identical bytes)."""
    allowed, covered = set(vs2), set()
    pairs, bad = [], []
    for v in vs1:
        if v in vmap:
            hit = vmap[v] if vmap[v] in allowed else None
        elif k1.is_const(v):
            hit = next((w for w in vs2 if k2.is_const(w) and s2[w] == s1[v]), None)
        else:
            hit = None
        if hit is None:
            bad.append(f"{k1.values[v].describe()} has no counterpart")
            continue
        covered.add(hit)          # several CSE'd copies may share one counterpart
        pairs.append(_pair(k1, k2, v, hit))
    for w in vs2:
        if w in covered:
            continue
        v = rmap.get(w)
        if v is not None and v in vs1:   # program 2 copies of one program-1 value
            pairs.append(_pair(k1, k2, v, w))
        else:
            bad.append(f"{k2.values[w].describe()} in program 2 has no counterpart")
    return pairs, bad


def _unmatched_reason(k1, k2, U1, U2, um1, um2):
    opaque = [k.nodes[n] for k, U in ((k1, um1), (k2, um2)) for n in U
              if k.nodes[n].op == "opaque"]
    if opaque:
        srcs = sorted({f"{n.attrs['source']} ({n.meta.get('reason', '')})" for n in opaque})
        return "unsupported operation: " + ", ".join(srcs)
    if not U1 or not U2:
        side = 2 if not U1 else 1
        return f"only program {side} has these operations; no counterpart computes their values"
    if (um1 or um2) and (len(um1) < len(U1) or len(um2) < len(U2)):
        return "source operations are only partially matched"
    return ("no counterpart computes the same values (different constants, e.g. folded "
            "weights, or a rewrite that is not exact in real arithmetic)")


def _dependency_order(k1, units):
    prod = {}
    for u in units:
        for b in u.outputs:
            prod[b.value_1] = u
    deps = {id(u): {id(prod[b.value_1]) for b in u.inputs if b.value_1 in prod} - {id(u)}
            for u in units}
    pos = {nid: i for i, nid in enumerate(k1.nodes)}
    ready = sorted([u for u in units if not deps[id(u)]],
                   key=lambda u: pos[u.canonical_nodes_1[0]])
    out, done = [], set()
    while ready:
        u = ready.pop(0)
        out.append(u)
        done.add(id(u))
        for v in units:
            if id(v) not in done and v not in ready and deps[id(v)] <= done:
                ready.append(v)
        ready.sort(key=lambda u: pos[u.canonical_nodes_1[0]])
    if len(out) != len(units):
        raise MatchError("matched units form a cycle")
    return out
