"""Adversarial regressions of v9: grid shape, rotation and BFS frontier."""
from __future__ import annotations

import common
from arcengine import FrameData, GameAction, GameState

ka = common.load_ka()
mod = ka._import_graph_explorer()
DEFAULT_ON = dict(mod.H5_FAMILY_ON)
mod.H5_FAMILY_ON["O1"] = True              # v9.1: O1 is off by default; the tests below exercise O1 when it is on


def agent():
    return mod.MyAgent(card_id="t", game_id="vc33", agent_name="t", ROOT_URL="http://localhost",
                       record=False, arc_env=None)


def test_smaller_grid_does_not_crash_h5():
    ag = agent()
    large = [[0] * 16 for _ in range(16)]
    small = [[0] * 4 for _ in range(4)]
    ag.h5._new_level(0, large)
    ag.h5._add("G1", (5, 5, 3, 3, 9, 9), 3)
    h = ag.h5.top()
    assert ag.h5.progress(h, small, None) is None
    ag.h5.credit(small, None)
    assert ag.h5 is not None and ag.stats["h5_credit_exc"] == 0
    return "G1 not applicable on a smaller frame; credit does not switch H5 off"


def test_credit_isolates_exception_of_one_instance():
    ag = agent()
    grid = [[0, 3], [0, 0]]
    ag.h5._new_level(0, grid)
    ag.h5._add("G1", (0, 0, 1, 1, 1, 1), 3)
    ag.h5._add("O3", 3, 2)
    original = ag.h5.progress

    def fail_one(h, frame, mover):
        if h.fam == "G1":
            raise ValueError("bad instance")
        return original(h, frame, mover)

    ag.h5.progress = fail_one
    ag.h5.credit([[0, 0], [0, 0]], None)
    assert ag.stats["h5_credit_exc"] == 1
    assert ag.stats["h5_win_true_instances"] == 1
    return "an exception in one instance preserves the credit of the others"


def test_prompt_after_top_expires():
    ag = agent()
    ag.h5._new_level(0, [[0, 3], [0, 0]])
    ag.h5._add("O3", 3, 4)
    ag.h5._add("O1", 2, 2)
    assert ag.h5.top().fam == "O3"
    ag.h5.on_exhaust()
    assert ag.h5.top_key == ("O1", 2)
    policy = ka.KPolicy.__new__(ka.KPolicy)
    policy.agent = ag
    assert "value 2" in "\n".join(policy.h5_lines())
    ag.h5.top_key = ("O3", 3)           # stale value coming from another path
    assert "value 2" in "\n".join(policy.h5_lines())
    return "F5 shows the live hypothesis after the leader expires"


def test_bfs_none_on_first_frontier():
    ag = agent()
    g0 = [[0] * 16 for _ in range(16)]
    start = ag._node(ag._key_of(g0, 0), g0, 0, [1, 2])
    g1 = [[0] * 16 for _ in range(16)]
    g1[5][5] = 4
    n1 = ag._node(ag._key_of(g1, 0), g1, 0, [1])
    n1.mover = None
    g2 = [row[:] for row in g1]
    g2[0][0] = 8                          # distinct state: the v8 red-team proof used two identical grids
    n2 = ag._node(ag._key_of(g2, 0), g2, 0, [1])
    n2.mover = (5.0, 5.0)
    assert n1.key != n2.key
    start.edges[(1,)] = n1.key
    start.edges[(2,)] = n2.key
    ag.h5._new_level(0, g0)
    ag.h5._add("O1", 4, 3)
    target, reg, clicks, hs = ag.h5.guide(g0, start, [1, 2])
    assert ag._bfs(start, 0, target, reg, hs)[0][0] == (2,)
    assert ag._bfs(start, 0, target, reg, lambda n: (_ for _ in ()).throw(ValueError("probe")))[0][0] == (1,)
    assert ag.stats["h5_hs_exc"] >= 2
    return "a frontier with score None preserves another optimal node at the same depth"


def test_o1_by_co_response_and_g2_contains_all_a():
    ag = agent()
    grid = [[0] * 9 for _ in range(9)]
    for x in range(2, 7):
        grid[2][x] = grid[6][x] = 4
    for y in range(2, 7):
        grid[y][2] = grid[y][6] = 4
    grid[1][1] = 2
    grid[4][4] = 2
    ag.h5._new_level(0, grid)
    ag.h5.mobile_colors.add(2)
    node = ag._node(ag._key_of(grid, 0), grid, 0, [6])
    ag.h5._instantiate(0, grid, node, [6])
    assert ("O1", (2, 4)) in ag.h5.hyps
    assert ("G2", (2, 4)) in ag.h5.hyps
    g2 = ag.h5.hyps[("G2", (2, 4))]
    assert ag.h5.progress(g2, grid, None) == 1
    filled = [row[:] for row in grid]
    filled[1][1] = 0
    filled[4][5] = 2
    assert ag.h5.progress(g2, filled, None) == 0
    mod.H5_FAMILY_ON["G2"] = False
    try:
        ag.h5.hyps.clear()
        ag.h5._instantiate(0, filled, node, [6])
        assert not any(k[0] == "G2" for k in ag.h5.hyps)
    finally:
        mod.H5_FAMILY_ON["G2"] = True
    return "O1 by mobile class after co-response; G2 requires all of A inside the ring; per-family switch"


def test_cache_invalidated_when_hud_grows():
    ag = agent()
    grid = [[2, 2], [0, 0]]
    assert ag.h5._counts(grid)[2] == 2
    ag.ui = {(0, 0)}
    assert ag.h5._counts(grid)[2] == 1
    return "the cache for the same grid follows the new HUD mask"


def test_hud_does_not_forget_masked_cell():
    ag = agent()
    grid = [[2, 0], [0, 0]]
    ag._hud_mask = lambda level, frame: {(0, 0)}
    ag._update_mask(grid, 0)
    assert ag.ui == {(0, 0)}
    ag._hud_mask = lambda level, frame: set()
    ag._update_mask(grid, 0)
    assert ag.ui == {(0, 0)}
    assert ag.h5._counts(grid)[2] == 0
    return "the learned HUD persists when the probe stops recognizing it in the next frame"


def test_g2_does_not_take_hud_as_object():
    ag = agent()
    grid = [[0] * 7 for _ in range(7)]
    for x in range(1, 6):
        grid[1][x] = grid[5][x] = 4
    for y in range(1, 6):
        grid[y][1] = grid[y][5] = 4
    grid[0][0] = 2
    ag.ui = {(0, 0)}
    assert ag.h5.progress(mod.Hyp("G2", (2, 4), 2), grid, None) is None
    return "A present only in the HUD makes G2 not applicable, never an exact predicate that is true"


def test_exhaustion_on_entry_does_not_reset():
    ag = agent()
    ag.h5 = None
    grid = [[0] * 16 for _ in range(16)]
    node = ag._node(ag._key_of(grid, 0), grid, 0, [1])
    node.tried.update({c: mod.NOOP for c in node.cands})
    ag._bfs = lambda *args, **kwargs: None
    ag.exhaust[0] = 1
    ag.reopened[0] = mod.REOPEN_MAX
    frame = FrameData(game_id="vc33", frame=[grid], state=GameState.NOT_FINISHED,
                      levels_completed=0, win_levels=7, available_actions=[1])
    action = ag.choose_action([], frame)
    assert action == GameAction.from_id(1), (action, ag.stats, ag.life)
    assert ag.stats["reset_avoided"] >= 1 and ag.life == 1, (ag.stats, ag.life)
    return "a graph exhausted on entry picks a legal action instead of a full RESET"


def test_o1_off_by_default_v91():
    assert DEFAULT_ON == {"O1": False, "O3": True, "G1": True, "G2": True, "G4": True}, DEFAULT_ON
    fresh = ka._import_graph_explorer()
    ag = fresh.MyAgent(card_id="t", game_id="vc33", agent_name="t", ROOT_URL="http://localhost",
                       record=False, arc_env=None)
    grid = [[0] * 8 for _ in range(8)]
    ag.h5._new_level(0, grid)
    ag.h5._add("O1", (2, 3), 3)
    assert not any(h.fam == "O1" for h in ag.h5.hyps.values())
    return "v9.1 default: O1 does not instantiate (RT9-A4); the other families stay on"


def test_update_isolates_exception_rt9_a1():
    ag = agent()
    grid = [[0, 3], [0, 0]]
    ag.h5._new_level(0, grid)
    ag.h5._add("G1", (0, 0, 1, 1, 1, 1), 3)
    ag.h5._add("O3", 3, 2)
    original = ag.h5.progress

    def fail_g1(h, frame, mover):
        if h.fam == "G1":
            raise IndexError("bad instance")
        return original(h, frame, mover)
    ag.h5.progress = fail_g1
    node = ag._node(ag._key_of(grid, 0), grid, 0, [1])
    ag.h5._update(grid, node)
    assert ag.stats["h5_update_exc"] == 1 and any(h.fam == "O3" and h.best is not None for h in ag.h5.hyps.values())
    return "an exception in one instance in _update does not prevent the others from being updated"


def test_ragged_grid_not_applicable_rt9_a3():
    ag = agent()
    ragged = [[0, 1, 0], [1, 0], [0, 1, 0]]
    for h in (mod.Hyp("G4", "v", 2), mod.Hyp("O3", 1, 2), mod.Hyp("G2", (1, 0), 2)):
        assert ag.h5.progress(h, ragged, None) is None, h.fam
    return "a ragged grid is not applicable (None), with no IndexError"


def test_ragged_grid_outside_progress_rt91_a1():
    # Guards M64-M66. v9.1 does not have the guards: there the test does not apply.
    if not hasattr(mod.MyAgent, "_ragged"):
        return "not applicable: version without the _ragged guards (v9.1)"
    ag = agent()
    ragged = [[0, 1, 0], [1, 0], [0, 1, 0]]
    assert ag.h5._sym_cells(ragged, "v") == set() and ag.h5._sym_cells(ragged, "h") == set()
    assert ag._click_targets(ragged) == []
    grid = [[0] * 12 for _ in range(11)] + [[0] * 5]   # short last row: the H2 search for R reaches it
    grid[0][0] = grid[1][1] = 3
    ag.changed_ever[0] = {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert ag._h2_regions(0, grid) is None
    return "_sym_cells, _click_targets and _h2_regions treat a ragged grid as empty, with no IndexError"


def test_h2_on_smaller_node_does_not_crash_bfs_rt9_a2():
    ag = agent()
    small = [[0] * 4 for _ in range(4)]
    n = ag._node(ag._key_of(small, 0), small, 0, [1])
    assert ag._h2_mismatch_node(n, (5, 5, 3, 3, 9, 9)) is None
    return "H2 with a region outside the node returns None (BFS skips the node)"


TESTS = [test_o1_off_by_default_v91, test_update_isolates_exception_rt9_a1, test_ragged_grid_not_applicable_rt9_a3,
         test_ragged_grid_outside_progress_rt91_a1,
         test_h2_on_smaller_node_does_not_crash_bfs_rt9_a2, test_smaller_grid_does_not_crash_h5, test_credit_isolates_exception_of_one_instance,
         test_prompt_after_top_expires,
         test_bfs_none_on_first_frontier, test_o1_by_co_response_and_g2_contains_all_a,
         test_cache_invalidated_when_hud_grows, test_hud_does_not_forget_masked_cell,
         test_g2_does_not_take_hud_as_object,
         test_exhaustion_on_entry_does_not_reset]
if __name__ == "__main__":
    raise SystemExit(common.main(TESTS))
