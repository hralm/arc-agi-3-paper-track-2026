"""B4-3 (second exhaustion round): click requested outside 0..63. The harness clamps it (solver.py:530) and
executes (63, c); the wrapper read the raw request and K created keys and certificates for coordinates that do not exist.
    python tests/test_b4_3_click_outside.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from inference.agent.tool_agent import AnalyzerTurnResult  # noqa: E402

ka = common.load_ka()


class StubOutside(common.Stub):
    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
        self.n += 1
        r, c = (70, self.rng.randrange(0, 64)) if self.n % 2 else (self.rng.randrange(0, 64), -3)
        p = step_env({"actions": [{"action": "MOUSE", "row": r, "col": c}]})
        return AnalyzerTurnResult(step_executed=bool(p and p.get("executed_count")), reasoning="stub")


def test_click_outside_grid_clamped():
    an = {}

    def make(game, index):
        a = ka.HybridAnalyzer(game, StubOutside(1, "random"), **dict(common.HOOK_OPTS))
        an[index] = a
        return a

    bm = common.run_games(["vc33"], make, max_actions=40, conc=1)
    r, a = bm.game_runs[0], an[0]
    assert len(r.history) > 5, f"the stub barely played: {len(r.history)} actions, state {r.state}"
    execd = {(6, int(h.action.data.get("x", 0)), int(h.action.data.get("y", 0)))
             for h in r.history if int(h.action.id.value) == 6}
    tried = {c for n in a.policy.agent.nodes.values() for c in n.tried if c[0] == 6}
    outside = sorted(c for c in tried if not (0 <= c[1] <= 63 and 0 <= c[2] <= 63))
    assert not outside, f"K recorded clicks outside the grid: {outside[:4]}"
    never = sorted(c for c in tried if c not in execd)
    assert not never, f"K recorded clicks the harness did not execute: {never[:4]}"


if __name__ == "__main__":
    sys.exit(common.main([test_click_outside_grid_clamped]))
