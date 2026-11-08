"""RT4/v5: only RESET available → RESET; h8_lines consistent with I10; the hint does not
call a NOOP on a hidden level proven; hidden_death on a CHANGE edge or by timer hides the level and
revokes the certificates.
    python tests/test_rt4_v5.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from test_rt1_life_transition import FakeGame  # noqa: E402

ka = common.load_ka()
from arcengine import FrameData, GameState  # noqa: E402

GRID = [[0] * 64 for _ in range(64)]
GRID[20][20] = 3


def _frame(avail, state=GameState.NOT_FINISHED, grid=GRID, levels=0):
    return FrameData(game_id="t", frame=[grid], state=state, levels_completed=levels, win_levels=7, available_actions=avail)


def _hyb():
    return ka.HybridAnalyzer(FakeGame(), common.Stub("x", "random"), **dict(common.HOOK_OPTS))


def test_only_reset_available_proposes_reset():
    mod = getattr(ka, "_import_graph_explorer", getattr(ka, "_import_graph_explorer", None))()
    for avail in ([0], []):
        ag = mod.MyAgent(card_id="t", game_id="vc33", agent_name="t", ROOT_URL="http://localhost", record=False, arc_env=None)
        ag.choose_action([], _frame([1, 2]))                   # 1st visit with keys: the node has candidates
        act = ag.choose_action([], _frame(avail))
        assert act.value == 0, f"avail={avail}: K proposed {act.name}"
        assert ag.stats.get("reset_only_reset_available", 0) == 1, dict(ag.stats)
    h = _hyb()
    h.policy.propose(_frame([1, 2]))
    args, diag = h.policy.propose(_frame([0]))
    assert args == {"action": "RESET"}, (args, diag)
    return "available_actions [0] and []: K and KPolicy propose RESET, never an invented [1..6]"


def test_h8_lines_does_not_announce_removal_undone_by_i10():
    h = _hyb()
    ag = h.policy.agent
    node = ag._node(123, GRID, 0, [1, 2])
    node.tried[(1,)] = ka._NOOP
    h._cert_sync(123, (1,), 1)
    ag.prev_key = 123
    h._last_valid = ["UP"]
    assert h._filter_valid(["UP"]) == ["UP"], "I10: a list with one certified key comes back whole"
    text = "\n".join(h.h8_lines())
    assert "removed from your valid list" not in text, text
    assert "list kept whole" in text and "UP" in text, text
    h._last_valid = ["UP", "DOWN"]
    assert h._filter_valid(["UP", "DOWN"]) == ["DOWN"]
    text = "\n".join(h.h8_lines())
    assert "removed from your valid list (verified to change nothing from this exact state): UP" in text, text
    assert "DOWN" not in text.split("removed from your valid list")[1].splitlines()[0], text
    return "I10 returns the whole list → H8 does not claim a removal; with a real removal, it announces only what was removed"


def test_hint_does_not_call_noop_proven_on_hidden_level():
    h = _hyb()
    ag = h.policy.agent
    node = ag._node(123, GRID, 0, [1, 2])
    node.tried[(1,)] = ka._NOOP
    ag.prev_key = 123
    before = "\n".join(h.policy.hint_lines())
    assert "proven to change NOTHING" in before and "UP" in before, before
    h._cert_sync(123, (1,), 1, died=True)                    # RT3-1: a NOOP that kills without a certificate → hidden level
    assert 0 in h.hidden_levels and h.policy.hidden_levels is h.hidden_levels
    after = "\n".join(h.policy.hint_lines())
    assert "proven to change NOTHING" not in after, after
    assert "hidden state" in after and "UP" in after, after
    return "hidden level: K's hint stops calling the NOOP proven and warns about the hidden state"


def _death(h, grid, key, act_name, avail):
    dead = FrameData(game_id="t", frame=[grid], state=GameState.GAME_OVER, levels_completed=0, win_levels=7, available_actions=avail)
    h.policy.current_frame = lambda: dead
    h._wrap_step_env(lambda a: {"executed": True, "executed_actions": [act_name], "executed_count": 1})({"actions": [{"action": act_name}]})


def test_hidden_death_change_and_timer_hide_level():
    """Red-team cases through the real wrapper: live certificate on (100, UP); death at another node on a CHANGE edge, and by timer."""
    grid = [[0] * 16 for _ in range(16)]
    for case in ("change", "timer"):
        h = _hyb()
        ag = h.policy.agent
        n100 = ag._node(100, grid, 0, [1, 2, 3])
        n100.tried[(1,)] = ka._NOOP
        h._cert_sync(100, (1,), 1)
        assert (100, (1,)) in h.certs
        if case == "change":
            n = ag._node(200, grid, 0, [1, 2, 3]); n.tried[(2,)] = ka._CHANGE
            ag.prev_key, ag.prev_act = 200, None
            _death(h, grid, 200, "ACTION2", [1, 2, 3])
        else:
            ag._node(300, grid, 0, [1, 2, 3])
            ag.life_budget, ag.life = 10, 10
            ag.prev_key, ag.prev_act = 300, None
            _death(h, grid, 300, "ACTION3", [1, 2, 3])
        assert ag.stats["hidden_death"] == 1, (case, dict(ag.stats))
        assert 0 in h.hidden_levels, (case, h.hidden_levels)
        assert (100, (1,)) not in h.certs, (case, h.certs)
        assert h.stats["cert_hidden_death"] == 1, (case, h.stats)
        ag.prev_key = 100
        assert "ACTION1" in h._filter_valid(["ACTION1", "ACTION2"]), case
    return "death on a CHANGE edge and by timer: level hidden, the level's certificates revoked, nothing masked"


TESTS = [test_only_reset_available_proposes_reset, test_h8_lines_does_not_announce_removal_undone_by_i10,
         test_hint_does_not_call_noop_proven_on_hidden_level, test_hidden_death_change_and_timer_hide_level]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
