"""v13 · adjustments to the base agent itself: B1 (K informs the scheduler through `cost_multiplier`), B2 (`k_map` in
the python tool), B3/B5 (setup_env switches). B2 constraints (witnessed edges only; hidden level gets no route;
route = shortest known one).
    python tests/test_v13_base_hooks.py
"""
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
N = 150
V12 = dict(advisor=True, h8=None, inject_hint=False, max_runtime_s=600.0, last_resort_s=1.0)   # = the K-only-at-the-end variant


class StubBase(common.Stub):
    """Stub with the two base surfaces that v13 touches: `_progress_pace` (read by the gate at every re-prioritization)
    and `_run_python_tool` + the state serialization (`_k_map()`, called by the B2 edit for every sandbox state)."""

    def __init__(self, seed, mode="random"):
        super().__init__(seed, mode)
        self._progress_pace = "base-pace"
        self.maps: list = []
        self.paces: list = []
        self.routes = 0
        self.hyb = None

    def _run_python_tool(self, state_path, arguments):
        return {"ok": True}

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
        if callable(getattr(self, "_k_map", None)):
            m = self._k_map()
            self.maps.append(m)
            if m and m.get("usable") and m.get("route_to_untried"):
                self.routes += _check_route(self.hyb, m)   # at the moment the map is served
            self._run_python_tool(None, {"code": "m = k_map; print(m and m.get('route_to_untried'))"})
        if hasattr(self._progress_pace, "cost_multiplier"):
            self.paces.append(self._progress_pace.cost_multiplier())
        return super().analyze(state_path, action_num, valid_actions=valid_actions, step_env=step_env, **kw)


def _run(game="ls20", n=N, **opts):
    an = {}

    def make(g, index):
        gid = str(g.env_name).split("-")[0]
        o = dict(common.HOOK_OPTS)
        o.update(V12)
        o.update(opts)
        a = ka.HybridAnalyzer(g, StubBase(gid), **o)
        a.model.hyb = a
        an.setdefault(gid, []).append(a)
        return a
    bm = common.run_games([game], make, max_actions=n, conc=1)
    return bm.game_runs[0], an[game][0]


def _as_tuple(obj):
    if isinstance(obj, dict):
        return (6, int(obj["col"]), int(obj["row"]))
    return ({v: k for k, v in ka._MODEL_NAME.items()}[obj],)


def _check_route(a, m) -> int:
    ag = a.policy.agent
    cur = ag.nodes[ag.prev_key]
    for act in m["route_to_untried"]["actions"]:          # every step is an observed edge (CHANGE), not an inherited one
        t = _as_tuple(act)
        assert cur.tried.get(t) == ka._CHANGE and t not in cur.unproven, (t, cur.tried.get(t))
        cur = ag.nodes[cur.edges[t]]
        assert cur.level == m["level"]
    assert ka._untried_legal(cur), "the route does not end in a state with an untried action"
    return 1


def test_kmap_witness_only_and_known_route():
    seen = 0
    for game in ("ls20", "vc33"):
        r, a = _run(game, kmap=True)
        assert r.state != "crashed", r.solver_note
        assert a.k_turns == 0 and a.stats["layer_exc"] == 0, a.stats
        assert all(p == "Step " + p.split()[1] + "\nline2\nrest of the prompt" for p in a.model.prompts)  # nothing in the prompt
        assert a.stats["kmap_used"] > 0 and a.stats["kmap_served"] > 0, a.stats
        maps = [m for m in a.model.maps if m]
        assert maps, "no map served"
        for m in maps:
            json.dumps(m)                                    # goes to the sandbox as JSON
            assert m["usable"] is False or (m["here"] == 0 and all(e[3] in ("NOOP", "CHANGE", "PROGRESS", "DEATH") for e in m["edges"]))
        seen += a.model.routes
    assert seen > 0, "no route checked: vacuous test"
    return f"k_map with observed edges only, JSON, nothing in the prompt; {seen} routes checked step by step when served"


def test_kmap_hidden_level_no_route():
    r, a = _run("ls20", n=60, kmap=True)
    ag = a.policy.agent
    a.hidden_levels.add(ag.nodes[ag.prev_key].level)
    m = a.k_map_payload()
    assert m["usable"] is False and "edges" not in m and "route_to_untried" not in m, m
    assert a.stats["kmap_unusable"] >= 1
    return "level with hidden state: usable=False, no edges and no route"


def test_kmap_failure_does_not_crash():
    orig = ka._untried_legal

    def boom(node):
        raise RuntimeError("injected into _untried_legal")
    try:
        ka._untried_legal = boom
        r, a = _run("ls20", n=60, kmap=True)
    finally:
        ka._untried_legal = orig
    assert r.state != "crashed", r.solver_note
    assert a.stats["layer_exc"] > 0 and all(m is None for m in a.model.maps), a.stats
    assert a.k_turns == 0 and len(r.history) > 0
    return f"k_map failure: payload None (layer_exc={a.stats['layer_exc']}), game goes on with the model"


def test_pace_only_penalizes_and_falls_back_to_1():
    r, a = _run("ls20", n=60, k_pace=True)
    assert isinstance(a.model._progress_pace, ka._KPace)
    assert a.model.paces and all(1.0 <= p <= 2.0 for p in a.model.paces), a.model.paces[:5]
    assert a.model._progress_pace.record_completion(1e6, 10.0) is None
    lvl = int(a.game.current_state.levels_completed)
    a._pace_level, a._pace_acts, a._pace_cert = lvl, 40, 40
    assert a.pace_factor() == 2.0
    a._pace_acts, a._pace_cert = 19, 19                       # < 20 actions in the level: fraction ignored
    base = a.pace_factor()
    ag = a.policy.agent
    a.hidden_levels.add(ag.nodes[ag.prev_key].level)
    assert base == 1.0 and a.pace_factor() == 1.25, (base, a.pace_factor())
    orig = ka.HybridAnalyzer.pace_factor
    try:
        ka.HybridAnalyzer.pace_factor = lambda self: 1 / 0
        assert a.model._progress_pace.cost_multiplier() == 1.0
    finally:
        ka.HybridAnalyzer.pace_factor = orig
    return "K factor between 1 and 2; 2.0 with everything certified; 1.25 from a hidden level alone; failure => 1.0 (the base)"


def test_v12_no_base_hooks():
    r, a = _run("ls20", n=40)
    m = a.model
    assert m._progress_pace == "base-pace" and not hasattr(m, "_k_map") and not m.maps and not m.paces
    assert a.stats["kmap_served"] == a.stats["pace_penalized"] == 0
    return "v12 (kmap=False, k_pace=False): base pace untouched, no k_map"


def test_kmap_synthetic_graph_inherited_and_other_level():
    """M70/M72: in a hand-built graph, the route uses neither an edge inherited in a merge (unproven) nor an edge to
    another level; the only valid route is the one through an observed edge on the same level."""
    r, a = _run("ls20", n=20, kmap=True)
    ag = a.policy.agent
    Node = type(next(iter(ag.nodes.values())))
    L = 7
    def new_node(k, lvl):
        n = Node(k, lvl, b"", (64, 64))
        n.cands = [(1,), (2,), (3,), (4,)]
        ag.nodes[k] = n
        return n
    h, x, y, z = new_node(-1, L), new_node(-2, L), new_node(-3, L + 1), new_node(-4, L)
    for n in (h,):
        n.tried = {(1,): ka._CHANGE, (2,): ka._CHANGE, (3,): ka._CHANGE, (4,): ka._NOOP}
    h.edges = {(1,): x.key, (2,): y.key, (3,): z.key, (4,): h.key}
    h.unproven = {(1,)}                                   # UP inherited in a merge: not a witness
    ag.prev_key = h.key
    m = a.k_map_payload()
    assert m["usable"] and m["route_to_untried"] == {"target": 1, "actions": ["LEFT"]}, m["route_to_untried"]
    assert all(e[1] != "UP" for e in m["edges"]), m["edges"]
    assert m["states"] == 2, m["states"]                  # here + z; x (inherited) and y (other level) left out
    return "inherited edge kept out of the map and the route; edge to another level kept out of the route; route = LEFT (observed)"


TESTS = [test_kmap_synthetic_graph_inherited_and_other_level, test_kmap_witness_only_and_known_route, test_kmap_hidden_level_no_route, test_kmap_failure_does_not_crash,
         test_pace_only_penalizes_and_falls_back_to_1, test_v12_no_base_hooks]
if __name__ == "__main__":
    raise SystemExit(common.main(TESTS))
