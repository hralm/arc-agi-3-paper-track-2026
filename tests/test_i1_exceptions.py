"""I1: no exception from the layer reaches the harness. Injects RuntimeError into each method of HybridAnalyzer/KPolicy
(and into K, the model, the factory) and checks, in the real HarnessSolver, that the game does not become "crashed",
reaches the action cap and that the failure was counted (stats layer_exc/k_exc/model_exc or FACTORY_ERRORS). A method
not exercised by the hook configuration is reported as such (it does not count as evidence).
    python tests/test_i1_exceptions.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
N = 40
GAMES = ["ls20"]          # keys + death by budget: exercises H8 list, K, model

TARGETS = [("HybridAnalyzer", m) for m in ("_analyze", "_k_turn", "_time_pressure", "_filter_valid", "_masked_here", "_cert_sync",
                                           "_record", "h8_lines", "_wrap_step_env", "_drop_level", "_yield_says_k")] + \
          [("KPolicy", m) for m in ("propose", "absorb", "observe", "mark_invalid", "current_frame", "frame_from_state",
                                    "hint_lines", "cross_level_lines")]


def _inject(cls, name, hits):
    orig = getattr(cls, name)

    def boom(*a, **k):
        hits[0] += 1
        raise RuntimeError(f"injected into {cls.__name__}.{name}")
    setattr(cls, name, boom)
    return orig


def _one(cls_name, name, mode="mix"):
    cls = getattr(ka, cls_name)
    hits = [0]
    orig = _inject(cls, name, hits)
    an = {}
    try:
        bm = common.run_games(GAMES, common.make_factory(ka, an, mode=mode), max_actions=N, conc=1)
    finally:
        setattr(cls, name, orig)
    r = bm.game_runs[0]
    a = an["ls20"][0]
    counted = a.stats.get("layer_exc", a.stats.get("layer_exc", 0)) + a.stats.get("k_exc", 0) + a.stats.get("model_exc", 0)
    return r, a, hits[0], counted


def test_method_injection():
    lines, bad = [], []
    for cls_name, name in TARGETS:
        r, a, hits, counted = _one(cls_name, name)
        ok = r.state != "crashed" and len(r.history) >= N
        if hits == 0:
            lines.append(f"{cls_name}.{name}: not exercised")
            continue
        if not ok or counted == 0:
            bad.append(f"{cls_name}.{name}: state={r.state} actions={len(r.history)} hits={hits} counted={counted} note={r.solver_note}")
        lines.append(f"{cls_name}.{name}: hits={hits} counted={counted} actions={len(r.history)}")
    assert not bad, "; ".join(bad)
    return " | ".join(lines)


def test_k_choose_action_raises():
    """Exception inside the K agent (graph_explorer.MyAgent.choose_action) with the model stalled: K is called every 3 turns."""
    orig = ka.KPolicy.__init__

    def init(self, *aa, **kk):
        orig(self, *aa, **kk)
        def boom(*a, **k):
            raise RuntimeError("K raised")
        self.agent.choose_action = boom
    ka.KPolicy.__init__ = init
    an = {}
    try:
        bm = common.run_games(GAMES, common.make_factory(ka, an, mode="stall"), max_actions=N, conc=1)
    finally:
        ka.KPolicy.__init__ = orig
    r, a = bm.game_runs[0], an["ls20"][0]
    assert r.state != "crashed" and len(r.history) >= N, f"state={r.state} actions={len(r.history)} note={r.solver_note}"
    assert a.stats["k_exc"] > 0 and a.stats["fallback_act"] > 0, a.stats
    return f"k_exc={a.stats['k_exc']} fallback_act={a.stats['fallback_act']} actions={len(r.history)}"


def test_model_raises():
    an = {}
    bm = common.run_games(GAMES, common.make_factory(ka, an, mode="raise"), max_actions=N, conc=1)
    r, a = bm.game_runs[0], an["ls20"][0]
    assert r.state != "crashed" and len(r.history) >= N, f"state={r.state} actions={len(r.history)} note={r.solver_note}"
    assert a.stats["model_exc"] > 0, a.stats
    return f"model_exc={a.stats['model_exc']} k_failure={a.stats['k_failure']}"


def test_factory_import_fails():
    """KPolicy.__init__ (import of graph_explorer) raises: the game continues with the model alone and the error is kept in FACTORY_ERRORS."""
    orig = ka.KPolicy.__init__

    def boom(self, *a, **k):
        raise RuntimeError("import of graph_explorer failed")
    ka.KPolicy.__init__ = boom
    ka.FACTORY_ERRORS.clear() if hasattr(ka, "FACTORY_ERRORS") else None
    models = []

    def mk(game, index):
        m = common.Stub("ls20", "random")
        models.append(m)
        return m
    try:
        bm = common.run_games(GAMES, ka.hybrid_factory(mk, **common.HOOK_OPTS), max_actions=N, conc=1)
    finally:
        ka.KPolicy.__init__ = orig
    r = bm.game_runs[0]
    assert r.state != "crashed" and len(r.history) >= N, f"state={r.state} actions={len(r.history)} note={r.solver_note}"
    assert getattr(ka, "FACTORY_ERRORS", {}).get("ls20"), "factory error not recorded"
    return f"model alone played {len(r.history)} actions; FACTORY_ERRORS={ka.FACTORY_ERRORS}"


def test_properties_raise():
    """The model's total_tokens raises: the harness reads tokens inside _execute_action; if it propagates, every action becomes an error."""
    class Bad(common.Stub):
        @property
        def total_tokens(self):
            raise RuntimeError("tokens")
        @total_tokens.setter
        def total_tokens(self, v):
            pass
        @property
        def generated_tokens(self):
            raise RuntimeError("tokens")
        @generated_tokens.setter
        def generated_tokens(self, v):
            pass
    an = {}

    def make(game, index):
        a = ka.HybridAnalyzer(game, Bad("ls20", "random"), **common.HOOK_OPTS)
        an["ls20"] = a
        return a
    bm = common.run_games(GAMES, make, max_actions=N, conc=1)
    r = bm.game_runs[0]
    assert r.state != "crashed" and len(r.history) >= N, f"state={r.state} actions={len(r.history)} note={r.solver_note}"
    return f"actions={len(r.history)}"


def test_no_double_action():
    """The model executes an action, the layer code fails in post-processing and the model raises: the turn counts as
    executed and K does not act on top of it (no double action in the same turn)."""
    class ActsThenRaises(common.Stub):
        def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
            step_env({"actions": [{"action": "UP"}]})
            raise RuntimeError("raised after acting")
    orig = ka.HybridAnalyzer._record

    def boom(*a, **k):
        raise RuntimeError("_record")
    ka.HybridAnalyzer._record = boom
    an = {}

    def make(game, index):
        a = ka.HybridAnalyzer(game, ActsThenRaises("ls20", "random"), **common.HOOK_OPTS)
        an["ls20"] = a
        return a
    try:
        bm = common.run_games(GAMES, make, max_actions=N, conc=1)
    finally:
        ka.HybridAnalyzer._record = orig
    r, a = bm.game_runs[0], an["ls20"]
    assert r.state != "crashed" and len(r.history) >= N, f"state={r.state} actions={len(r.history)}"
    assert a.k_turns == 0 and a.stats["fallback_act"] == 0, f"K/fallback acted on top: k_turns={a.k_turns} {a.stats}"
    return f"actions={len(r.history)} model turns={a.model_turns} K=0 model_exc={a.stats['model_exc']}"


TESTS = [test_no_double_action, test_method_injection, test_k_choose_action_raises, test_model_raises, test_factory_import_fails, test_properties_raise]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
