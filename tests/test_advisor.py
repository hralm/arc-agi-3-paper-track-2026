"""Advising mode (advisor=True): K observes, prunes the H8 list and annotates the prompt, but never plays.
Under RHAE, (h/a)^2 per level, every K action counts towards the model's level.
    python tests/test_advisor.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
N = 60
K_KEYS = ("k_failure", "k_stall", "k_repeat", "k_time", "k_yield", "fallback_act")


def _run(mode, advisor, max_runtime=600.0, games=("ls20",)):
    an = {}
    bm = common.run_games(list(games), common.make_factory(ka, an, mode=mode, advisor=advisor), max_actions=N, conc=1,
                          max_runtime=max_runtime)
    return bm.game_runs[0], an[games[0]][0]


def test_advisor_never_plays_in_mix():
    r, a = _run("mix", True)
    assert r.state != "crashed", r.solver_note
    assert a.k_turns == 0 and all(a.stats[k] == 0 for k in K_KEYS), {k: a.stats[k] for k in K_KEYS}
    assert a.model_turns > 0 and len(r.history) > 0
    assert a.stats["model_act_total"] == len(r.history), (a.stats["model_act_total"], len(r.history))
    return f"mix (stall/random/fail): {len(r.history)} actions, all from the model; k_turns=0; cert_emitted={a.stats['cert_emitted']}"


def test_control_without_advisor_plays():
    r, a = _run("mix", False)
    assert a.k_turns > 0, a.stats
    return f"control advisor=False: k_turns={a.k_turns} (the test above tells the modes apart)"


def test_model_failure_returns_to_harness():
    r, a = _run("fail", True, max_runtime=15.0)
    assert r.state != "crashed", r.solver_note
    assert a.k_turns == 0 and a.stats["fallback_act"] == 0 and len(r.history) == 0, (a.stats, len(r.history))
    return "model always failing: no K action and no fallback action; the harness retries the turn"


def test_model_raising_does_not_crash():
    r, a = _run("raise", True, max_runtime=15.0)
    assert r.state != "crashed", r.solver_note
    assert a.stats["model_exc"] > 0 and a.k_turns == 0 and len(r.history) == 0, (a.stats, len(r.history))
    return f"model raises: model_exc={a.stats['model_exc']}, no K action"


def test_layer_failure_returns_turn_to_model():
    cls = ka.HybridAnalyzer
    orig = cls._filter_valid

    def boom(self, *x, **k):
        raise RuntimeError("injected into _filter_valid")
    cls._filter_valid = boom
    try:
        r, a = _run("random", True)
    finally:
        cls._filter_valid = orig
    assert r.state != "crashed", r.solver_note
    assert a.stats["layer_exc"] > 0 and a.stats["fallback_act"] == 0 and a.k_turns == 0, a.stats
    assert len(r.history) > 0 and a.model.n > 0
    return f"layer_exc={a.stats['layer_exc']}: the turn goes to the plain model ({len(r.history)} actions), no fallback"


def test_no_h8_neither_prunes_nor_annotates():
    r, a = _run("random", True)
    an = {}
    bm = common.run_games(["ls20"], common.make_factory(ka, an, mode="random", advisor=True, h8=None), max_actions=N, conc=1)
    b = an["ls20"][0]
    assert b.h8 is None and b.stats["h8_masked_keys"] == 0 and b.k_turns == 0, b.stats
    assert all(not any("verified to change nothing" in ln for ln in p.splitlines()) for p in b.model.prompts)
    assert len(bm.game_runs[0].history) > 0
    return f"advisor + h8=None: 0 keys pruned, no H8 note in the prompt; with h8=list: {a.stats['h8_masked_keys']} pruned"


def _run_endgame(mode, max_runtime, last_resort_s, lvl_b, game_b, n=N):
    an = {}
    bm = common.run_games(["ls20"], common.make_factory(ka, an, mode=mode, advisor=True, max_runtime_s=max_runtime,
                                                        last_resort_s=last_resort_s, k_level_budget=lvl_b, k_game_budget=game_b),
                          max_actions=n, conc=1, max_runtime=max_runtime)
    return bm.game_runs[0], an["ls20"][0]


def test_last_resort_only_at_end_and_within_budget():
    r, a = _run_endgame("random", 600.0, 1.0, 10, 25)         # final window of 1 s within a 600 s cap: K does not enter
    assert a.stats["k_endgame"] == 0 and a.k_turns == 0 and len(r.history) > 0, a.stats
    r, a = _run_endgame("stall", 20.0, 1e9, 10, 25, n=200)    # window = whole game; model stalled; budget 10/25
    assert r.state != "crashed", r.solver_note
    assert 0 < a.k_acts_game <= 25 and a.k_acts_level <= 10, (a.k_acts_game, a.k_acts_level, a.stats)
    assert a.stats["k_stall"] == a.stats["k_repeat"] == a.stats["k_time"] == a.stats["k_failure"] == 0, a.stats
    return f"outside the window: 0 K turns; inside the window: {a.k_acts_game} K actions (<= 25 per game, <= 10 per level), only k_endgame"


def test_endgame_only_prompt_and_list_intact():
    an = {}
    bm = common.run_games(["ls20"], common.make_factory(ka, an, mode="random", advisor=True, h8=None, inject_hint=False,
                                                        max_runtime_s=600.0, last_resort_s=1.0), max_actions=N, conc=1)
    a = an["ls20"][0]
    assert all(p == "Step " + p.split()[1] + "\nline2\nrest of the prompt" for p in a.model.prompts), a.model.prompts[:2]
    assert a.stats["h8_masked_keys"] == 0 and a.k_turns == 0 and len(bm.game_runs[0].history) > 0, a.stats
    return "endgame only: model prompt identical to the original, list not pruned, K takes no turns outside the window"


TESTS = [test_endgame_only_prompt_and_list_intact, test_last_resort_only_at_end_and_within_budget, test_no_h8_neither_prunes_nor_annotates, test_advisor_never_plays_in_mix, test_control_without_advisor_plays, test_model_failure_returns_to_harness,
         test_model_raising_does_not_crash, test_layer_failure_returns_turn_to_model]
if __name__ == "__main__":
    raise SystemExit(common.main(TESTS))
