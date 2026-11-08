"""RT2/v3: Enum without shared mutation, certificate after DEATH, and plan across levels.
    python tests/test_rt2_v3.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from test_rt1_life_transition import FakeGame  # noqa: E402

ka = common.load_ka()
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState  # noqa: E402


class Env:
    observation_space = None

    def __init__(self):
        self.calls = []

    def step(self, action, data=None, reasoning=None):
        ActionInput(id=action, data=data or {}, reasoning=reasoning)
        self.calls.append((action, dict(data or {}), reasoning))
        return FrameDataRaw(game_id="teste", frame=[np.zeros((2, 2), dtype=int)], state=GameState.NOT_FINISHED,
                            levels_completed=0, win_levels=7, available_actions=[1, 2, 3, 4, 6])


def test_enum_without_set_data_and_independent_clicks():
    mod = getattr(ka, "_import_graph_explorer", getattr(ka, "_import_graph_explorer", None))()
    assert ".set_data(" not in Path(ka.__file__).read_text(encoding="utf-8")
    assert ".set_data(" not in Path(mod.__file__).read_text(encoding="utf-8")
    member = GameAction.from_id(6)
    before = member.action_data.model_dump()
    calls = []
    original = GameAction.set_data

    def spy(self, value):
        calls.append(dict(value))
        return original(self, value)

    GameAction.set_data = spy
    try:
        env_a, env_b = Env(), Env()
        a = mod.MyAgent(card_id="a", game_id="vc33", agent_name="a", ROOT_URL="http://localhost",
                        record=False, arc_env=env_a)
        b = mod.MyAgent(card_id="b", game_id="tu93", agent_name="b", ROOT_URL="http://localhost",
                        record=False, arc_env=env_b)
        act_a = a._emit(SimpleNamespace(key=100, level=0, legal=frozenset()), (6, 2, 3), "plan")   # v4: Node has `legal`
        act_b = b._emit(SimpleNamespace(key=200, level=0, legal=frozenset()), (6, 30, 40), "plan")
        a.do_action_request(act_a)
        b.do_action_request(act_b)
    finally:
        GameAction.set_data = original
    assert not calls, f"set_data called on the shared Enum: {calls}"
    assert member.action_data.model_dump() == before, "action_data of the shared member changed"
    assert env_a.calls[0][1]["x"] == 2 and env_a.calls[0][1]["y"] == 3, env_a.calls
    assert env_b.calls[0][1]["x"] == 30 and env_b.calls[0][1]["y"] == 40, env_b.calls
    return "0 set_data; standalone clicks keep per-instance coordinates"


def _hyb():
    return ka.HybridAnalyzer(FakeGame(), common.Stub("x", "random"), **dict(common.HOOK_OPTS))


def test_certificate_is_falsified_by_death():
    h = _hyb()
    ag = h.policy.agent
    node = ag._node(100, [[0] * 16 for _ in range(16)], 0, [1, 2])
    node.tried[(1,)] = ka._NOOP
    h._cert_sync(100, (1,), 1)
    assert (100, (1,)) in h.certs and h.stats["cert_emitted"] == 1, "vacuous pass: no certificate"
    node.tried[(1,)] = ka._DEATH
    h._cert_sync(100, (1,), 1)
    ag.prev_key = 100
    filtered = h._filter_valid(["ACTION1", "ACTION2"])
    text = "\n".join(h.h8_lines())
    assert (100, (1,)) not in h.certs, "certificate survived DEATH"
    assert h.stats["cert_falsified"] == 1 and 0 in h.hidden_levels, (h.stats, h.hidden_levels)
    assert "ACTION1" in filtered, f"lethal action is still masked: {filtered}"
    assert "verified to change nothing" not in text, text
    return "DEATH falsifies, revokes and unmasks the certificate"


def test_plan_does_not_cross_level():
    h = _hyb()
    ag = h.policy.agent
    ag.plan.append(((1,), 200))
    ag.plan.append(((2,), 300))

    def fake_step(args):
        return {"executed": True, "executed_actions": ["ACTION1", "ACTION1"],
                "executed_count": 2, "level_completed": True}

    grid = [[0] * 16 for _ in range(16)]
    frame = FrameData(game_id="t", frame=[grid], state=GameState.NOT_FINISHED,
                      levels_completed=1, win_levels=7, available_actions=[1, 2, 3, 4])
    h.policy.current_frame = lambda: frame
    h._wrap_step_env(fake_step)({"actions": [{"action": "ACTION1"}, {"action": "ACTION1"}]})
    assert not ag.plan, f"plan from the previous level survived: {list(ag.plan)}"
    args, diag = h.policy.propose()
    assert diag.get("k", {}).get("how") != "plan", f"action inherited from the previous plan: {args} {diag}"
    return "a batch that opens a level clears the previous plan"


def test_real_model_death_revokes_certificate():
    """Real flow (not synthetic): K does not mark DEATH on an edge already seen as NOOP (`hidden_death`, hidden
    death), so `tried` stays NOOP; the certificate only falls if the analyzer looks at the step's resulting frame
    (GAME OVER)."""
    h = _hyb()
    ag = h.policy.agent
    grid = [[0] * 16 for _ in range(16)]
    node = ag._node(100, grid, 0, [1, 2])
    node.tried[(1,)] = ka._NOOP
    h._cert_sync(100, (1,), 1)
    assert (100, (1,)) in h.certs, "vacuous pass: no certificate"
    ag.prev_key, ag.prev_act = 100, None

    def fake_step(args):
        return {"executed": True, "executed_actions": ["ACTION1"], "executed_count": 1}

    dead = FrameData(game_id="t", frame=[grid], state=GameState.GAME_OVER, levels_completed=0, win_levels=7,
                     available_actions=[1, 2, 3, 4])
    h.policy.current_frame = lambda: dead
    h._wrap_step_env(fake_step)({"actions": [{"action": "ACTION1"}]})
    assert node.tried.get((1,)) == ka._NOOP, f"premise: K marked {node.tried.get((1,))}, this is not the real-flow case"
    assert ag.stats["hidden_death"] == 1, dict(ag.stats)
    assert (100, (1,)) not in h.certs, "certificate survived the real GAME OVER"
    assert h.stats["cert_falsified"] == 1 and 0 in h.hidden_levels, (h.stats, h.hidden_levels)
    ag.prev_key = 100
    assert "ACTION1" in h._filter_valid(["ACTION1", "ACTION2"]), "lethal action is still masked"
    return "model GAME OVER on a certified action (tried=NOOP) falsifies and revokes the level"


TESTS = [test_enum_without_set_data_and_independent_clicks, test_certificate_is_falsified_by_death,
         test_plan_does_not_cross_level, test_real_model_death_revokes_certificate]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
