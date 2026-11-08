"""RT-1: a level advance starts with life zero; the budget must not inherit the previous life.
    python tests/test_rt1_life_transition.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
from arcengine import GameState  # noqa: E402


class FakeGame:
    def __init__(self):
        self.env_name = "vc33-x"
        grid = [[0] * 16 for _ in range(16)]
        grid[8][8] = 5
        raw = SimpleNamespace(frame=[grid], state=GameState.NOT_FINISHED, levels_completed=0, win_levels=7,
                              available_actions=[1, 2, 3, 4], game_id=self.env_name)
        self.current_state = SimpleNamespace(raw=raw, levels_completed=0, available_actions=[0, 1, 2, 3, 4])


def test_absorb_model_advances_level():
    o = dict(common.HOOK_OPTS)
    h = ka.HybridAnalyzer(FakeGame(), common.Stub("x", "random"), **o)
    pol, ag = h.policy, h.policy.agent
    pol.propose(pol.current_frame())
    ag.prev_act = None
    ag.life, ag.life_budget = 7, None
    frame = pol.current_frame()
    frame.levels_completed = 1
    pol.absorb(frame)
    assert ag.prev_levels == 1, "scenario did not advance a level"
    assert ag.life == 0, f"life inherited on the model's turn: {ag.life}"
    return "absorb on the model's turn leaves life=0 on the new level"


def _run_k(game, max_actions):
    transitions, resets = [], []
    actions_since_transition = 0
    original = ka.HybridAnalyzer._k_turn

    def spy(self, *args, **kwargs):
        nonlocal actions_since_transition
        ag = self.policy.agent
        level_before, life_before = int(ag.prev_levels), int(ag.life)
        resets_before = int(ag.stats.get("reset_life_budget_exhausted", 0))
        result = original(self, *args, **kwargs)
        level_after, life_after = int(ag.prev_levels), int(ag.life)
        real_reset = (int(ag.stats.get("reset_life_budget_exhausted", 0)) > resets_before
                      and result.step_executed and "'k': 'reset'" in result.reasoning)
        if real_reset:
            resets.append({"level": level_after, "actions": actions_since_transition,
                           "life_before": life_before, "budget": ag.life_budget})
        elif level_after == level_before and result.step_executed:
            actions_since_transition += 1
        if level_after > level_before:
            actions_since_transition = 0
            transitions.append({"before": level_before, "after": level_after,
                                "life": life_after, "budget": ag.life_budget})
        return result

    ka.HybridAnalyzer._k_turn = spy
    analyzers = {}
    try:
        bm = common.run_games([game], common.make_factory(ka, analyzers, "fail"), max_actions=max_actions,
                              conc=1, max_runtime=120.0)
    finally:
        ka.HybridAnalyzer._k_turn = original
    return bm, transitions, resets


def test_k_transition_zeroes_life():
    bm, transitions, resets = _run_k("vc33", 600)
    assert transitions, "vacuous pass: K did not open any level"
    wrong = [t for t in transitions if t["life"] != 0]
    assert not wrong, f"life inherited by K: {wrong}"
    return f"{len(transitions)} transitions with life=0: {transitions}"


def test_budget_does_not_reset_new_level_early():
    bm, transitions, resets = _run_k("tu93", 500)
    assert transitions, "vacuous pass: K did not open any level"
    assert all(t["life"] == 0 for t in transitions), f"life inherited by K: {transitions}"
    assert any(t["budget"] == 50 for t in transitions), f"scenario without a budget of 50: {transitions}"
    early = [r for r in resets if r["level"] > 0 and r["actions"] < 49]
    assert not early, f"RESET before 49 actions on the new level: {early}; transitions={transitions}"
    return f"no early RESET; transitions={transitions}; resets={resets}"


TESTS = [test_absorb_model_advances_level, test_k_transition_zeroes_life, test_budget_does_not_reset_new_level_early]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
