"""v14 · cross-game memory (family prior and schema) in K's H5, used only as a tie-break after the evidence.
    python tests/test_v14_cross_game.py
"""
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
V12 = dict(advisor=True, h8=None, inject_hint=False, max_runtime_s=600.0, last_resort_s=1.0)


def _new_cross():
    ka.CROSS = ka._CrossGame()
    return ka.CROSS


def _run(games, n=150, **opts):
    an = {}
    bm = common.run_games(list(games), common.make_factory(ka, an, **opts), max_actions=n, conc=1)
    return bm, an


def test_shared_store_and_records_win():
    """Two real games share the store; an unambiguous win in the 1st one is recorded and the 2nd one sees it.
    (A real win with a true instance is rare: the 49/66 failure reported in the paper; hence the win here is synthetic.)"""
    cross = _new_cross()
    bm, an = _run(["vc33", "tu93"], n=120, mode="stall", advisor=False, cross_game=True)
    h1, h2 = an["vc33"][0].policy.agent.h5, an["tu93"][0].policy.agent.h5
    assert h1.cross is cross and h2.cross is cross
    assert all(r.state != "crashed" for r in bm.game_runs)
    H = ka._import_graph_explorer().Hyp
    h1.hyps = {("O3", 7): H("O3", 7, 2), ("G2", (7, 3)): H("G2", (7, 3), 2)}
    h1.progress = lambda h, g, m: 0 if h.fam == "O3" else 5            # only O3 holds on the win frame
    before = cross.records
    h1.credit([[0]], None)
    del h1.progress
    fams, schemas = h2.cross.snapshot()
    assert cross.records == before + 1 and fams.get("O3", 0) >= 1, (cross.records, fams)
    assert h1.ag.stats["h5_cross_registro"] >= 1 and h1.ag.stats["h5_cross_exc"] == 0
    return f"single store across games; win of game 1 visible in game 2: families={fams}"


def test_tiebreak_never_beats_evidence():
    H = ka._import_graph_explorer().Hyp
    a, b = H("O3", 5, 2), H("G2", (1, 2), 2)
    assert min([a, b], key=lambda h: h.order(None)) is a                 # tie: fixed family order
    assert min([a, b], key=lambda h: h.order(None, prior=0 if h is a else 9)) is b   # the family prior breaks the tie
    assert min([a, b], key=lambda h: h.order(None, schema=1 if h is a else 0, prior=0 if h is a else 9)) is a  # schema > prior
    c = H("G4", "v", 3)                                                  # stronger evidence
    assert min([a, b, c], key=lambda h: h.order(None, schema=0 if h is c else 99, prior=0 if h is c else 99)) is c
    return "evidence from the game > schema > family prior > fixed order"


def test_role_without_color():
    bm, an = _run(["vc33"], n=60, mode="stall", advisor=False, cross_game=True)
    h5 = an["vc33"][0].policy.agent.h5
    h5.counts0 = Counter({0: 4000, 3: 50, 7: 10, 9: 30})
    h5.mobile_colors = {7}
    r1 = (h5.role("O3", 7), h5.role("G2", (7, 3)))
    h5.counts0 = Counter({1: 4000, 5: 50, 2: 10, 8: 30})              # same classes, permuted colors
    h5.mobile_colors = {2}
    r2 = (h5.role("O3", 2), h5.role("G2", (2, 5)))
    assert r1 == r2 == (("O3", (0, True)), ("G2", (0, True), (2, False))), (r1, r2)
    return f"role invariant under color permutation: {r1}"


def test_ambiguous_not_recorded_and_failure_does_not_crash():
    cross = _new_cross()
    bm, an = _run(["vc33"], n=60, mode="stall", advisor=False, cross_game=True)
    h5 = an["vc33"][0].policy.agent.h5
    before = cross.records
    H = ka._import_graph_explorer().Hyp
    h5.hyps = {("O3", 7): H("O3", 7, 2), ("O3", 9): H("O3", 9, 2)}
    h5.progress = lambda h, g, m: 0                                      # both hold on the win frame
    h5.credit([[0]], None)
    assert cross.records == before and h5.ag.stats["h5_win_ambiguous"] >= 1
    del h5.progress

    class Faulty:
        def snapshot(self):
            raise RuntimeError("injected")
    h5.cross = Faulty()
    h5.hyps = {("O3", 7): H("O3", 7, 2)}
    exc0 = h5.ag.stats["h5_cross_exc"]
    assert h5.top() is not None and h5.ag.stats["h5_cross_exc"] == exc0 + 1
    return "ambiguous win is not recorded; a store that raises does not bring down H5 (h5_cross_exc counts it)"


def test_v12_no_store():
    bm, an = _run(["ls20"], n=40, mode="random", **V12)
    h5 = an["ls20"][0].policy.agent.h5
    assert h5 is None or h5.cross is None
    return "v12 (cross_game=False): H5 without a store"


def test_k_cross_decisao_counts_only_k_turns():
    """Review caveat: the guide's mark only counts when K plays (k_endgame); the discarded proposal of a model
    turn (absorb -> propose -> guide) does not count."""
    _new_cross()
    orig = ka.KPolicy.propose

    def mark(self, *a, **k):
        r = orig(self, *a, **k)
        h5 = self.agent.h5
        if h5 is not None:
            h5.guide_cross_changed = True               # as if the store had changed the 1st place
        return r
    try:
        ka.KPolicy.propose = mark
        an = {}
        bm = common.run_games(["ls20"], common.make_factory(ka, an, mode="random", advisor=True, h8=None, inject_hint=False,
                                                 max_runtime_s=600.0, last_resort_s=1e9, k_level_budget=10,
                                                 k_game_budget=25, cross_game=True), max_actions=200, conc=1,
                         max_runtime=600.0)
    finally:
        ka.KPolicy.propose = orig
    a = an["ls20"][0]
    assert a.model_turns > 0 and a.stats["k_endgame"] > 0, (a.model_turns, a.stats["k_endgame"])
    assert a.stats["k_cross_decisao"] == a.stats["k_endgame"], (a.stats["k_cross_decisao"], a.stats["k_endgame"])
    an2 = {}
    common.run_games(["ls20"], common.make_factory(ka, an2, mode="random", advisor=True, h8=None, inject_hint=False,
                                        max_runtime_s=600.0, last_resort_s=1e9, k_level_budget=10, k_game_budget=25,
                                        cross_game=True), max_actions=200, conc=1, max_runtime=600.0)
    b = an2["ls20"][0]
    assert b.stats["k_cross_decisao"] == 0, b.stats["k_cross_decisao"]   # empty store: nothing to change
    return (f"k_cross_decisao = k_endgame turns ({a.stats['k_endgame']}) with the mark; {a.model_turns} model turns "
            "do not count; empty store => 0")


def test_k_cross_decisao_clears_stale_mark():
    """M84: if K's proposal does not go through the guide (no live instance, guide disabled), the mark left by a
    model turn must not become a K decision. _k_turn clears the mark before propose."""
    _new_cross()
    orig_abs = ka.KPolicy.absorb

    def absorb_mark(self, frame):
        r = orig_abs(self, frame)
        if self.agent.h5 is not None:
            self.agent.h5.guide_cross_changed = True      # stale mark left by the model path
        return r
    an = {}

    def make(g, index):
        o = dict(common.HOOK_OPTS)
        o.update(advisor=True, h8=None, inject_hint=False, max_runtime_s=600.0, last_resort_s=1e9, k_level_budget=10,
                 k_game_budget=25, cross_game=True)
        a = ka.HybridAnalyzer(g, common.Stub("ls20", "random"), **o)
        if a.policy.agent.h5 is not None:
            a.policy.agent.h5.guide = lambda *x, **k: None   # K's proposal does not go through the guide
        an.setdefault("ls20", []).append(a)
        return a
    try:
        ka.KPolicy.absorb = absorb_mark
        common.run_games(["ls20"], make, max_actions=200, conc=1, max_runtime=600.0)
    finally:
        ka.KPolicy.absorb = orig_abs
    a = an["ls20"][0]
    assert a.stats["k_endgame"] > 0 and a.model_turns > 0, (a.stats["k_endgame"], a.model_turns)
    assert a.stats["k_cross_decisao"] == 0, a.stats["k_cross_decisao"]
    return f"stale mark from the model path cleared: k_cross_decisao=0 over {a.stats['k_endgame']} K turns"


TESTS = [test_k_cross_decisao_clears_stale_mark, test_k_cross_decisao_counts_only_k_turns, test_tiebreak_never_beats_evidence, test_role_without_color, test_ambiguous_not_recorded_and_failure_does_not_crash,
         test_v12_no_store, test_shared_store_and_records_win]
if __name__ == "__main__":
    raise SystemExit(common.main(TESTS))
