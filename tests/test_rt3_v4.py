"""RT3/v4: legal action when `available_actions` changes without the pixels changing, and no certificate emitted
or kept over a GAME OVER without a prior certificate, after reopen/reindex.
    python tests/test_rt3_v4.py
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from test_rt1_life_transition import FakeGame  # noqa: E402

ka = common.load_ka()
from arcengine import FrameData, GameState  # noqa: E402

GRID = [[0] * 64 for _ in range(64)]
GRID[20][20] = 3


def _agent(mod, gid="vc33"):
    return mod.MyAgent(card_id="t", game_id=gid, agent_name="t", ROOT_URL="http://localhost", record=False, arc_env=None)


def _frame(avail, state=GameState.NOT_FINISHED, grid=GRID, levels=0):
    return FrameData(game_id="t", frame=[grid], state=state, levels_completed=levels, win_levels=7, available_actions=avail)


def test_revisit_with_new_availability_proposes_legal():
    """Red-team proof (dynamic actions), plus 30 random availability switches."""
    mod = getattr(ka, "_import_graph_explorer", getattr(ka, "_import_graph_explorer", None))()
    ag = _agent(mod)
    first = _frame([1])
    a1 = ag.choose_action([], first)
    a2 = ag.choose_action([first], _frame([2]))
    assert a1.value == 1, a1
    assert a2.value in (0, 2), f"K proposed {a2.name} with available [2]"
    assert ag.stats.get("illegal_action_avoided", 0) + ag.stats.get("local", 0) >= 1, dict(ag.stats)
    import random
    rng = random.Random(7)
    illegal = []
    for i in range(30):
        avail = sorted(rng.sample([1, 2, 3, 4, 5, 7], rng.randint(1, 4)))
        act = ag.choose_action([], _frame(avail))
        if act.value != 0 and act.value not in avail:
            illegal.append((avail, act.name))
    assert not illegal, f"{len(illegal)} illegal proposals: {illegal[:5]}"
    return f"[1]→[2] legal; 30 visits with varying availability, 0 illegal; avoided={ag.stats.get('illegal_action_avoided', 0)}"


def test_plan_with_illegal_step_is_discarded():
    mod = getattr(ka, "_import_graph_explorer", getattr(ka, "_import_graph_explorer", None))()
    ag = _agent(mod)
    ag.choose_action([], _frame([1, 2]))
    ag.prev_key = ag.prev_act = None                    # model batch (RT2-2): step 1 does not see the plan divergence
    ag.plan = deque([((1,), 999), ((2,), 998)])
    act = ag.choose_action([], _frame([2]))             # same frame, 1 (UP) left the list: only the _emit guard holds
    assert act.value in (0, 2), f"illegal plan step was emitted: {act.name}"
    assert not ag.plan, f"plan with an illegal step survived: {list(ag.plan)}"
    assert ag.stats.get("illegal_action_avoided", 0) == 1, dict(ag.stats)
    return "a plan whose 1st step became illegal is discarded and the emitted action is legal"


def test_candidates_gain_new_keys():
    mod = getattr(ka, "_import_graph_explorer", getattr(ka, "_import_graph_explorer", None))()
    ag = _agent(mod)
    ag.choose_action([], _frame([1]))
    node = ag.nodes[ag.prev_key]
    assert (2,) not in node.tier
    ag.choose_action([], _frame([1, 2]))
    assert (2,) in node.tier and node.legal == frozenset({1, 2}), (node.tier, node.legal)
    return "a new key becomes a candidate on revisit; legal = current availability"


def _hyb():
    return ka.HybridAnalyzer(FakeGame(), common.Stub("x", "random"), **dict(common.HOOK_OPTS))


def test_death_without_prior_certificate_does_not_emit():
    """Red-team cases: reopen (the level's certs erased) and reindex (certs.clear()); tried stays NOOP (hidden_death)."""
    for case, key, level, act in (("reopen", 100, 0, (1,)), ("reindex", 200, 1, (2,))):
        h = _hyb()
        ag = h.policy.agent
        node = ag._node(key, [[0] * 16 for _ in range(16)], level, [1, 2])
        node.tried[act] = ka._NOOP
        if case == "reopen":
            ag.reopened[level] = 1
        else:
            ag.stats["reindex"] = 1
        h._cert_sync(key, act, 1, died=True)
        assert (key, act) not in h.certs, f"{case}: certificate emitted over GAME OVER"
        assert h.stats["cert_emitted"] == 0, (case, h.stats)
        assert level in h.hidden_levels and h.stats["cert_falsified"] == 1, (case, h.stats, h.hidden_levels)
        ag.prev_key = key
        assert "ACTION1" in h._filter_valid(["ACTION1", "ACTION2"]), case
    return "reopen and reindex: death on NOOP without a prior certificate does not emit, hides the level and falsifies"


def test_real_death_after_reopen_in_wrapper():
    h = _hyb()
    ag = h.policy.agent
    grid = [[0] * 16 for _ in range(16)]
    node = ag._node(100, grid, 0, [1, 2])
    node.tried[(1,)] = ka._NOOP
    h._cert_sync(100, (1,), 1)
    assert (100, (1,)) in h.certs
    ag.reopened[0] = 1                                   # K reopened: the next _cert_sync erases the level's certificates
    ag.prev_key, ag.prev_act = 100, None
    dead = FrameData(game_id="t", frame=[grid], state=GameState.GAME_OVER, levels_completed=0, win_levels=7,
                     available_actions=[1, 2, 3, 4])
    h.policy.current_frame = lambda: dead
    h._wrap_step_env(lambda a: {"executed": True, "executed_actions": ["ACTION1"], "executed_count": 1})(
        {"actions": [{"action": "ACTION1"}]})
    assert node.tried.get((1,)) == ka._NOOP and ag.stats["hidden_death"] == 1, (node.tried, dict(ag.stats))
    assert (100, (1,)) not in h.certs and h.stats["cert_emitted"] == 1, (h.certs, h.stats)
    assert 0 in h.hidden_levels and h.stats["cert_falsified"] == 1, (h.stats, h.hidden_levels)
    return "wrapper: reopen erases the certificate and the following GAME OVER does not re-emit it; level hidden"


TESTS = [test_revisit_with_new_availability_proposes_legal, test_plan_with_illegal_step_is_discarded,
         test_candidates_gain_new_keys, test_death_without_prior_certificate_does_not_emit, test_real_death_after_reopen_in_wrapper]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
