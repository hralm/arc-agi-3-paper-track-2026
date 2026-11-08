"""B3-1 (exhaustion round): batch [MOUSE(r,c), UP] in a click-only game. The harness executes 1 action
(the UP is invalid, stop_reason invalid_action). The wrapper read the coordinates from actions[-1] (the UP -> 0,0) and K learned
edges and emitted H8 certificates for a click that never happened. It now reads the n_exec-th request.
    python tests/test_b3_1_batch_click.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from inference.agent.tool_agent import AnalyzerTurnResult  # noqa: E402

ka = common.load_ka()
GAME = "vc33"


class StubBatch(common.Stub):
    """Always requests a batch of 2: one valid click and one invalid UP (vc33 accepts only clicks)."""

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kw):
        self.n += 1
        self.prompts.append(self._build_user_prompt(action_num, valid_actions=valid_actions))
        r, c = self.rng.randrange(8, 64), self.rng.randrange(8, 64)
        p = step_env({"actions": [{"action": "MOUSE", "row": r, "col": c}, {"action": "UP"}]})
        self.last = p
        return AnalyzerTurnResult(step_executed=bool(p and p.get("executed_count")), reasoning="stub")


def _run():
    an = {}

    def make(game, index):
        a = ka.HybridAnalyzer(game, StubBatch(1, "random"), **dict(common.HOOK_OPTS))
        an[index] = a
        return a

    bm = common.run_games([GAME], make, max_actions=60, conc=1)
    r = bm.game_runs[0]
    a = an[0]
    execd = {(6, int(h.action.data.get("x", 0)), int(h.action.data.get("y", 0)))
             for h in r.history if int(h.action.id.value) == 6}
    tried = {c for n in a.policy.agent.nodes.values() for c in n.tried if c[0] == 6}
    return r, a, execd, tried


def test_truncated_batch_click_executed():
    r, a, execd, tried = _run()
    assert len(r.history) > 5, f"the stub barely played: {len(r.history)} actions, state {r.state}"
    false_clicks = sorted(c for c in tried if c not in execd)
    assert not false_clicks, f"K recorded clicks that were never executed (read actions[-1]): {false_clicks[:5]}"
    false_certs = [c for c in a.certs if c[1][0] == 6 and c[1] not in execd]
    assert not false_certs, f"H8 certificates for a click never executed: {false_certs[:5]}"


def test_truncated_batch_no_false_repeat():
    r, a, execd, tried = _run()
    # k_repeat fired 16x in the same scenario because the "click" (0,0) repeated; with the correct, random click, it does not repeat
    assert a.stats.get("k_repeat", 0) <= 2, f"k_repeat={a.stats.get('k_repeat')} (false click repeated)"


if __name__ == "__main__":
    sys.exit(common.main([test_truncated_batch_click_executed, test_truncated_batch_no_false_repeat]))
