"""I2 (per-instance state) and I3 (legal action, no data read from the shared Enum member).
    python tests/test_i2_i3_threads.py
"""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
from arcengine import FrameData, GameState  # noqa: E402
N = 80


def _seq(run):
    return [str(h.action) if hasattr(h, "action") else str(h) for h in run.history]


def test_concurrent_isolation():
    """ls20 alone vs. ls20 in two simultaneous passes together with vc33 (n_passes=2, concurrency=4): the action sequence of each ls20 is the same
    as in the solo run (K and the stub are deterministic per game; no time trigger fires within 80 actions)."""
    m0 = (copy.deepcopy(ka._MODEL_NAME), copy.deepcopy(ka._ENGINE_ID))
    a1 = {}
    solo = common.run_games(["ls20"], common.make_factory(ka, a1), max_actions=N, conc=1)
    ref = _seq(solo.game_runs[0])
    a2 = {}
    bm = common.run_games(["ls20", "vc33"], common.make_factory(ka, a2), max_actions=N, conc=4, passes=2)
    outs = [_seq(r) for r in bm.game_runs if common.gid_of(r) == "ls20"]
    assert len(outs) == 2
    for s in outs:
        assert s == ref, f"sequence diverged under concurrency: first difference at {next(i for i, (x, y) in enumerate(zip(s, ref)) if x != y)}"
    x, y = a2["ls20"]
    assert x is not y and x.policy is not y.policy and x.policy.agent is not y.policy.agent
    assert x.certs is not y.certs and x.stats is not y.stats and x.policy.agent.nodes is not y.policy.agent.nodes
    assert type(x.policy.agent) is not type(y.policy.agent), "agent class shared (single graph_explorer module)"
    # no mutable container shared between the two instances (scan of the attributes, one level down)
    def mut(o):
        return {id(v): k for k, v in vars(o).items() if isinstance(v, (dict, list, set))}
    for o1, o2 in ((x, y), (x.policy, y.policy), (x.policy.agent, y.policy.agent)):
        inter = set(mut(o1)) & set(mut(o2))
        assert not inter, f"shared: {[mut(o1)[i] for i in inter]}"
    assert (ka._MODEL_NAME, ka._ENGINE_ID) == m0, "module tables mutated"
    return f"2x ls20 concurrent = solo ({len(ref)} actions); distinct agents, graphs, certificates and classes"


def _frame(cx, cy):
    g = [[0] * 64 for _ in range(64)]
    for y in range(cy - 2, cy + 3):
        for x in range(cx - 2, cx + 3):
            g[y][x] = 3
    return FrameData(game_id="t", frame=[g], state=GameState.NOT_FINISHED, levels_completed=0, win_levels=1,
                     available_actions=[6])


class _G:
    def __init__(self, name):
        self.env_name = name


def test_shared_enum_interleaved():
    """Deterministic interleaving of the thread race: between A's K choosing the click (set_data on the ACTION6 member) and
    A's KPolicy.propose reading the action, B's K chooses another click. A must send A's click."""
    A, B = ka.KPolicy(_G("aa01")), ka.KPolicy(_G("bb02"))
    fa, fb = _frame(10, 10), _frame(50, 50)
    for p, f in ((A, fa), (B, fb)):
        p.propose(f)                                   # first frame: K sees the state
    origA = A.agent.choose_action

    def interleaved(frames, latest):
        act = origA(frames, latest)
        B.agent.choose_action(B.frames, fb)            # the other thread calls set_data on the same Enum member
        return act
    A.agent.choose_action = interleaved
    errors = 0
    for _ in range(20):
        args, _d = A.propose(fa)
        pa = A.agent.prev_act
        if args.get("action") != "MOUSE":
            continue
        if (args["col"], args["row"]) != (pa[1], pa[2]):
            errors += 1
    assert errors == 0, f"{errors}/20 clicks of A went out with B's coordinates"
    return "20 interleaved proposals, 0 clicks swapped"


def test_legal_actions():
    """Every action requested from the harness step_env (by K, by the model and by the fallback) is a dict, has a name mapped
    by the harness and, if executed, was in available_actions; MOUSE with integer row/col in 0..63."""
    reg = []
    orig = common.solver_mod._HarnessGameSession.step_env

    def spy(self, arguments):
        avail = [int(a) for a in self.game.current_state.available_actions]
        p = orig(self, arguments)
        reg.append((common.gid_of(self.game.game_run), arguments, avail, p))
        return p
    common.solver_mod._HarnessGameSession.step_env = spy
    an = {}
    try:
        common.run_games(["ls20", "vc33", "sp80"], common.make_factory(ka, an, mode="mix"), max_actions=N, conc=3)
    finally:
        common.solver_mod._HarnessGameSession.step_env = orig
    eid = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "MOUSE": 6, "RESET": 0}
    bad, invalid, k_undo = [], 0, 0
    for gid, args, avail, p in reg:
        if not isinstance(args, dict):
            bad.append(f"{gid}: not a dict {args!r}")
            continue
        items = args["actions"] if "actions" in args else [args]
        for it in items:
            nm = str(it.get("action", "")).upper()
            if nm == "UNDO":
                k_undo += 1
            if nm == "MOUSE":
                r, c = it.get("row"), it.get("col")
                if not (isinstance(r, int) and isinstance(c, int) and 0 <= r < 64 and 0 <= c < 64):
                    bad.append(f"{gid}: MOUSE outside the grid {it}")
            if p.get("executed") and nm in eid and eid[nm] not in avail:
                bad.append(f"{gid}: executed outside available_actions {it} {avail}")
        if p.get("error"):
            invalid += 1
    assert not bad, "; ".join(bad[:5])
    return f"{len(reg)} requests; 0 illegal executed; refused by the harness without spending an action: {invalid} (UNDO from K: {k_undo})"


def test_no_set_data_in_layer():
    src = common.KA_FILE.read_text(encoding="utf-8")
    assert ".set_data(" not in src, "hybrid_analyzer calls set_data"
    mod_fn = getattr(ka, "_import_graph_explorer", getattr(ka, "_import_graph_explorer", None))
    ak = Path(mod_fn().__file__).read_text(encoding="utf-8")   # RT2-C1: the second embedded file as well
    assert ".set_data(" not in ak, "graph_explorer calls set_data on the shared Enum member"
    return "hybrid_analyzer.py and graph_explorer.py: 0 set_data calls; sending by dict / instance"


TESTS = [test_concurrent_isolation, test_shared_enum_interleaved, test_legal_actions, test_no_set_data_in_layer]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
