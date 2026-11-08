"""RT6/v7: hint and h8_lines do not suggest an action outside `legal`; `_reindex` remaps `start_key`;
BFS does not cross an intermediate node merged through an action that is not legal on every merged screen; the
merge's "not a proof" mark is per action and disappears when K observes it again.
    python tests/test_rt6_v7.py
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


def _g(*cells):
    g = [[0] * 16 for _ in range(16)]
    for y, x, v in cells:
        g[y][x] = v
    return g


def test_hint_does_not_suggest_illegal_untried():
    h = _hyb(); ag = h.policy.agent
    g = _g()
    k = ag._key_of(g, 0)
    node = ag._node(k, g, 0, [1, 2]); node.tried[(1,)] = ka._CHANGE
    node.legal = frozenset({0})                      # only RESET now
    ag.prev_key = k
    hint = "\n".join(h.policy.hint_lines())
    assert "not yet tried" not in hint, hint
    node.legal = frozenset({1})                      # only UP legal: DOWN is not a suggestion
    node.tried.pop((1,))
    hint = "\n".join(h.policy.hint_lines())
    assert "not yet tried from this exact state: UP" in hint and "DOWN" not in hint, hint
    return "hint: with legal={0} no suggestion; with legal={1} only UP"


def test_h8_lines_counts_legal_untried_at_destination():
    h = _hyb(); ag = h.policy.agent
    ga, gb = _g(), _g((5, 5, 3))
    ka_, kb = ag._key_of(ga, 0), ag._key_of(gb, 0)
    na = ag._node(ka_, ga, 0, [1, 2]); nb = ag._node(kb, gb, 0, [1, 2, 3, 4])
    na.tried[(1,)] = ka._CHANGE; na.edges[(1,)] = kb
    nb.legal = frozenset({1})
    ag.prev_key = ka_
    text = "\n".join(h.h8_lines())
    assert "(1 untried actions there)" in text, text
    return "h8_lines: destination with legal={1} counts 1 untried, not 4"


def test_reindex_remaps_start_key():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)
    g0, g1, g2, gm = _g((0, 0, 7)), _g((0, 0, 7), (1, 1, 1)), _g((0, 0, 7), (2, 2, 2)), _g((0, 0, 7), (3, 3, 3))
    ag.choose_action([], _frame(g0, [1, 2]))
    ag._hud_mask = lambda lvl, g: {(0, 0)}
    ag.choose_action([], _frame(g1, [1, 2]))
    assert ag.stats.get("reindex", 0) >= 1, dict(ag.stats)
    assert ag.start_key[0] in ag.nodes and ag.start_key[0] == ag._key_of(g0, 0), "start_key orphaned after reindex"
    ag.life_budget, ag.life = 10, 8
    k1, k2, km = ag._key_of(g1, 0), ag._key_of(g2, 0), ag._key_of(gm, 0)
    n1 = ag.nodes[k1]; ag._node(k2, g2, 0, [1, 2]); nm = ag._node(km, gm, 0, [1, 2])
    n1.edges[(1,)] = km; n1.tried[(1,)] = mod.CHANGE
    nm.edges[(1,)] = k2; nm.tried[(1,)] = mod.CHANGE
    n0 = ag.nodes[ag._key_of(g0, 0)]; n0.edges[(2,)] = k2; n0.tried[(2,)] = mod.CHANGE
    for n in (n1, nm):
        for c in list(n.cands):
            n.tried[c] = mod.CHANGE
    act = ag.choose_action([], _frame(g1, [1, 2]))
    assert act.value == 0 and ag.stats.get("reset_plan_over_budget", 0) == 1, (act, dict(ag.stats))
    return "start_key follows the node after the reindex; a plan that does not fit in the remaining life resets (plan over budget)"


def _merged_m(mod, ag, legal_m1=(1, 2), legal_m2=(3, 4)):
    gs, gm1, gm2, gt = _g((1, 1, 5)), _g(), _g((0, 0, 1)), _g((5, 5, 4))
    ks, km1, km2, kt = (ag._key_of(g, 0) for g in (gs, gm1, gm2, gt))
    ns = ag._node(ks, gs, 0, [1, 2]); ag._node(km1, gm1, 0, list(legal_m1))
    nm2 = ag._node(km2, gm2, 0, list(legal_m2)); ag._node(kt, gt, 0, [1, 2])
    ns.edges[(1,)] = km1; ns.tried[(1,)] = mod.CHANGE
    nm2.edges[(3,)] = kt; nm2.tried[(3,)] = mod.CHANGE
    ag.ui = {(0, 0)}
    ag._reindex()
    m = ag.nodes[ag._key_of(gm1, 0)]
    for c in list(ns.cands):
        ns.tried[c] = mod.CHANGE
    for c in list(m.cands):
        m.tried[c] = mod.CHANGE
    return gs, gm1, m


def test_bfs_does_not_cross_merged_by_non_common_action():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)
    gs, gm1, m = _merged_m(mod, ag)
    assert m.fused and m.core == frozenset() and m.plan_legal() == frozenset(), (m.core, m.legal)
    ag.choose_action([], _frame(gs, [1, 2]))
    assert not any(a[0] == 3 for a, _ in ag.plan), list(ag.plan)
    ag.choose_action([], _frame(gm1, [1, 2]))
    assert ag.stats.get("illegal_action_avoided", 0) == 0, dict(ag.stats)
    # an action common to both screens crosses; a later merge does not lose the empty core
    ag2 = _ag(mod)
    _, _, m2 = _merged_m(mod, ag2, (1, 3), (3, 4))
    assert m2.core == frozenset({3}), m2.core
    empty = mod.Node(1, 0, b"", (0, 0)); empty.fused, empty.core = True, frozenset()
    other = mod.Node(2, 0, b"", (0, 0)); other.legal = frozenset({1, 2})
    empty.merge(other)
    assert empty.core == frozenset(), empty.core
    return "BFS: a merged intermediate node only lets through an action legal on every screen; an empty core survives a new merge"


def test_bfs_from_merged_start_uses_real_legal():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)
    _, gm1, m = _merged_m(mod, ag)
    m.legal = frozenset({3, 4})                      # arrived on screen M2: 3 is legal now (empty core)
    path = ag._bfs(m, mod.MAX_TIER)
    assert path and path[0][0] == (3,), path
    return "merged start node: BFS uses the availability of the screen it is on, not the core"


def test_noop_observed_after_merge_is_proof_again():
    h = _hyb(); ag = h.policy.agent
    g1, g2 = _g(), _g((0, 0, 1))
    k1, k2 = ag._key_of(g1, 0), ag._key_of(g2, 0)
    n1 = ag._node(k1, g1, 0, [1, 2, 3, 4]); n1.tried[(2,)] = ka._NOOP   # 3 and 4 remain: the graph neither exhausts nor reopens
    ag._node(k2, g2, 0, [1, 2, 3, 4]).tried[(1,)] = ka._NOOP     # UP inherited from the other screen
    ag.ui = {(0, 0)}
    ag._reindex()
    kf = ag._key_of(g1, 0)
    m = ag.nodes[kf]
    assert {(1,), (2,)} <= m.unproven, m.unproven
    # K tries UP on the merged node through the real flow: the learned edge clears only UP's mark
    ag.prev_key, ag.prev_act, ag.prev_levels, ag.prev_grid = kf, (1,), 0, g1
    ag.choose_action([], _frame(g1, [1, 2, 3, 4]))
    assert m.tried.get((1,)) == ka._NOOP and (1,) not in m.unproven and (2,) in m.unproven, (m.tried, m.unproven)
    ag.prev_key = m.key                              # the step may have reindexed again
    hint = "\n".join(h.policy.hint_lines())
    assert "proven to change NOTHING from this exact state: UP" in hint, hint
    assert "so that is NOT a proof: DOWN" in hint, hint
    return "merged node: a NOOP observed after the merge is a proof (UP); the inherited one stays 'NOT a proof' (DOWN)"


TESTS = [test_hint_does_not_suggest_illegal_untried, test_h8_lines_counts_legal_untried_at_destination, test_reindex_remaps_start_key,
         test_bfs_does_not_cross_merged_by_non_common_action, test_bfs_from_merged_start_uses_real_legal,
         test_noop_observed_after_merge_is_proof_again]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
