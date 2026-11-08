"""RT5/v6: edge learned before the RESET caused by only-RESET, and no loop; BFS does not plan through an inherited
illegal edge; the hint does not call a NOOP inherited in the reindex merge proven; h8_lines with an
empty list does not announce a removal.
    python tests/test_rt5_v6.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from test_rt1_life_transition import FakeGame  # noqa: E402

ka = common.load_ka()
from arcengine import FrameData, GameState  # noqa: E402


def _ag(mod):
    return mod.MyAgent(card_id="t", game_id="vc33", agent_name="t", ROOT_URL="http://localhost", record=False, arc_env=None)


def _frame(grid, avail, levels=0):
    return FrameData(game_id="t", frame=[grid], state=GameState.NOT_FINISHED, levels_completed=levels, win_levels=7, available_actions=avail)


def _hyb():
    return ka.HybridAnalyzer(FakeGame(), common.Stub("x", "random"), **dict(common.HOOK_OPTS))


def test_only_reset_learns_edge_and_does_not_repeat():
    mod = ka._import_graph_explorer()
    ga = [[0] * 16 for _ in range(16)]; ga[5][5] = 1
    gb = [[0] * 16 for _ in range(16)]; gb[5][5] = 2
    ag = _ag(mod)
    a1 = ag.choose_action([], _frame(ga, [1, 2]))
    key_a, act_a = ag.prev_key, ag.prev_act
    node_a = ag.nodes[key_a]
    a2 = ag.choose_action([], _frame(gb, [0]))
    assert a2.value == 0, a2
    assert node_a.tried.get(act_a) == mod.CHANGE, f"edge not learned: {node_a.tried}"
    assert act_a not in node_a.untried(mod.MAX_TIER, ag.reactive)
    dead = ag.nodes[ag._key_of(gb, 0)]
    assert dead.legal == frozenset({0}) and not dead.untried(mod.MAX_TIER, ag.reactive), (dead.legal, dead.cands)
    a3 = ag.choose_action([], _frame(ga, [1, 2]))
    assert a3.value != a1.value and a3.value in (1, 2), f"loop: repeated {a3.name}"
    # same grid, but the actions disappeared: that is also a change (not NOOP, not certifiable)
    ag2 = _ag(mod)
    ag2.choose_action([], _frame(ga, [1, 2])); k, a = ag2.prev_key, ag2.prev_act
    ag2.choose_action([], _frame(ga, [0]))
    assert ag2.nodes[k].tried.get(a) == mod.CHANGE, ag2.nodes[k].tried
    return "an action leading to only-RESET becomes CHANGE at the origin; dead node legal={0}; K does not repeat; same grid without actions ≠ NOOP"


def test_bfs_ignores_inherited_illegal_edge():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)
    g_a = [[0] * 16 for _ in range(16)]
    g_b = [r[:] for r in g_a]; g_b[0][0] = 1
    g_e = [[0] * 16 for _ in range(16)]; g_e[5][5] = 4
    k_a, k_b, k_e = ag._key_of(g_a, 0), ag._key_of(g_b, 0), ag._key_of(g_e, 0)
    ag._node(k_a, g_a, 0, [1, 2]); node_b = ag._node(k_b, g_b, 0, [3, 4]); node_e = ag._node(k_e, g_e, 0, [1, 2])
    node_b.edges[(3,)] = k_e; node_b.tried[(3,)] = mod.CHANGE
    ag.ui = {(0, 0)}
    ag._reindex()
    merged = ag.nodes[ag._key_of(g_a, 0)]
    assert (3,) in merged.edges and merged.fused
    for c in list(merged.cands):
        merged.tried[c] = mod.NOOP
    assert node_e.untried(mod.MAX_TIER, ag.reactive)
    act = ag.choose_action([], _frame(g_a, [1, 2]))
    assert ag.stats.get("illegal_action_avoided", 0) == 0, dict(ag.stats)
    assert act.value in (0, 1, 2), act
    assert not any(a[0] == 3 for a, _ in ag.plan), list(ag.plan)
    return "BFS does not route through edge (3,) inherited from the merge and illegal in [1, 2]; no plan aborted"


def test_hint_does_not_call_noop_inherited_in_merge_proven():
    h = _hyb(); ag = h.policy.agent
    g1 = [[0] * 16 for _ in range(16)]; g2 = [r[:] for r in g1]; g2[0][0] = 1
    k1, k2 = ag._key_of(g1, 0), ag._key_of(g2, 0)
    n1 = ag._node(k1, g1, 0, [1, 2]); n1.tried[(1,)] = ka._NOOP
    ag._node(k2, g2, 0, [1, 2])
    h._cert_sync(k1, (1,))
    ag.prev_key, ag.prev_levels, ag.ui = k2, 0, {(0, 0)}
    ag._reindex(); h._cert_sync(None, None)
    assert len(ag.nodes) == 1 and not h.certs
    hint = "\n".join(h.policy.hint_lines())
    assert "proven to change NOTHING" not in hint, hint
    assert "merged from two different screens" in hint and "UP" in hint, hint
    return "merged node: the hint downgrades the inherited NOOP to 'NOT a proof'"


def test_h8_lines_with_empty_list_does_not_announce_removal():
    h = _hyb(); ag = h.policy.agent
    node = ag._node(123, [[0] * 16 for _ in range(16)], 0, [1, 2]); node.tried[(1,)] = ka._NOOP
    h._cert_sync(123, (1,), 1); ag.prev_key = 123
    h._last_valid = []
    assert h._filter_valid([]) == []
    text = "\n".join(h.h8_lines())
    assert "removed from your valid list" not in text, text
    assert "advisory" in text and "UP" in text, text
    return "empty list: H8 does not claim a removal; keeps the advisory note"


TESTS = [test_only_reset_learns_edge_and_does_not_repeat, test_bfs_ignores_inherited_illegal_edge,
         test_hint_does_not_call_noop_inherited_in_merge_proven, test_h8_lines_with_empty_list_does_not_announce_removal]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
