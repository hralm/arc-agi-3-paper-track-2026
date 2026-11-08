"""H5/v8: registry of win-condition hypotheses inside K. Properties: an empty frame is
"not applicable" (A3); eliminations are monotone within the level (A5); an exception goes to "error" and K continues
(A8/A11); recoloring is not removal (O3 requires a cell that becomes background); credit is read on the second-to-last
frame of the action that raises the level (C5); a hypothesis only orders (with H5 off the decision is the v7 one);
elimination only after the triggers have been tried (C33).
    python tests/test_h5_v8.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
from arcengine import FrameData, GameState  # noqa: E402


def _ag(mod):
    return mod.MyAgent(card_id="t", game_id="vc33", agent_name="t", ROOT_URL="http://localhost", record=False, arc_env=None)


def _frame(grids, avail, levels=0):
    if grids and isinstance(grids[0][0], int):
        grids = [grids]
    return FrameData(game_id="t", frame=grids, state=GameState.NOT_FINISHED, levels_completed=levels, win_levels=7,
                     available_actions=avail)


def _g(*cells, n=16):
    g = [[0] * n for _ in range(n)]
    for y, x, v in cells:
        g[y][x] = v
    return g


def test_empty_frame_not_applicable():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)
    h5 = ag.h5
    assert h5 is not None and mod.H5_ON
    for fam, bind in (("O1", 3), ("O3", 3), ("G4", "v"), ("G4", "h")):
        hyp = mod.Hyp(fam, bind, 1)
        assert h5.progress(hyp, [], (1.0, 1.0)) is None, fam
        assert h5.progress(hyp, [[]], (1.0, 1.0)) is None, fam
    assert h5.progress(mod.Hyp("O1", 3, 1), _g((1, 1, 3)), None) is None     # no mover: not applicable
    assert h5.progress(mod.Hyp("O1", 3, 1), _g(), (1.0, 1.0)) is None       # bound class absent: not applicable
    return "progress is None on an empty frame, with no mover and without the bound class"


def test_recolor_does_not_instantiate_o3_removal_does():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)
    g0 = _g((2, 2, 3), (2, 3, 3), (5, 5, 4))
    ag.choose_action([], _frame(g0, [6]))
    g1 = _g((2, 2, 7), (2, 3, 3), (5, 5, 4))                # 3 -> 7: recolor
    ag.choose_action([], _frame(g1, [6]))
    assert not any(k[0] == "O3" for k in ag.h5.hyps), list(ag.h5.hyps)
    g2 = _g((2, 2, 7), (2, 3, 0), (5, 5, 4))                # 3 -> background: removal
    ag.choose_action([], _frame(g2, [6]))
    assert ("O3", 3) in ag.h5.hyps, list(ag.h5.hyps)
    return "O3 only with a cell that becomes background; recoloring does not count"


def test_elimination_only_with_triggers_and_monotone():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)
    h5 = ag.h5
    g = _g((2, 2, 3), (9, 9, 5))
    h5._new_level(0, g)
    h5._add("O3", 3, 2)
    gz = _g((9, 9, 5))                                       # O3(3) holds: no cell with value 3
    node = ag._node(ag._key_of(gz, 0), gz, 0, [1, 2])
    h5._update(gz, node)
    assert ("O3", 3) in h5.hyps, "eliminated without trying the triggers"
    node.tried[(1,)] = mod.NOOP
    h5._update(gz, node)
    assert ("O3", 3) in h5.hyps, "eliminated with only one of the actions tried"
    node.tried[(2,)] = mod.CHANGE
    h5._update(gz, node)
    assert ("O3", 3) not in h5.hyps and ("O3", 3) in h5.elim, (h5.hyps, h5.elim)
    h5._add("O3", 3, 2)                                      # reopening does not recreate it (A5)
    assert ("O3", 3) not in h5.hyps
    assert ag.stats["h5_eliminated_O3"] == 1
    return "sufficiency only after every legal action was tried with no level change; an eliminated one does not come back within the level"


def test_o1_only_expires_never_eliminated():
    mod = ka._import_graph_explorer()
    mod.H5_FAMILY_ON["O1"] = True          # v9.1: O1 is off by default; here its behavior is tested
    ag = _ag(mod)
    h5 = ag.h5
    g = _g((2, 2, 3))
    h5._new_level(0, g)
    h5._add("O1", 3, 2)
    node = ag._node(ag._key_of(g, 0), g, 0, [1, 2]); node.mover = (2.0, 2.0)
    node.tried[(1,)] = mod.NOOP; node.tried[(2,)] = mod.NOOP
    for _ in range(mod.H5_PATIENCE + 2):
        h5._update(g, node)
    h = h5.hyps.get(("O1", 3))
    assert h is not None and h.state == "expired" and ("O1", 3) not in h5.elim, (h and h.state, h5.elim)
    return "O1 has no exact predicate: patience only makes it expire"


def test_credit_on_second_to_last_frame():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)
    g0 = _g((2, 2, 3), (2, 3, 3), (5, 5, 4))
    ag.choose_action([], _frame(g0, [6]))
    g1 = _g((2, 3, 3), (5, 5, 4))
    ag.choose_action([], _frame(g1, [6]))                    # removes one 3: O3(3) instantiated
    assert ("O3", 3) in ag.h5.hyps
    win = _g((5, 5, 4))                                      # second-to-last: no 3 (O3 holds)
    new = _g((1, 1, 6), (5, 5, 4))                           # last: already the next level
    ag.choose_action([], _frame([win, new], [6], levels=1))
    st = ag.stats
    assert st["h5_win_levels"] == 1 and st["h5_win_true_instances"] >= 1 and st["h5_win_no_true_instance"] == 0, dict(st)
    assert ag.h5.compose == "O3" and ag.h5.level == 1, (ag.h5.compose, ag.h5.level)
    ag2 = _ag(mod)                                           # mutant C5: using the last frame gives an anomaly
    ag2.choose_action([], _frame(g0, [6])); ag2.choose_action([], _frame(g1, [6]))
    ag2.choose_action([], _frame([g1, new], [6], levels=1))
    assert ag2.stats["h5_win_no_true_instance"] == 1, dict(ag2.stats)
    return "credit read on the second-to-last frame; the corroborated family carries over to the next level (k->k+1)"


def test_exception_goes_to_error_and_k_continues():
    mod = ka._import_graph_explorer()
    ag = _ag(mod)

    def boom(*a, **k):
        raise RuntimeError("H5 probe")
    ag.h5.observe = boom
    act = ag.choose_action([], _frame(_g((2, 2, 3)), [1, 2]))
    assert ag.h5 is None and ag.stats["h5_exc"] == 1 and act.value in (1, 2), (ag.h5, dict(ag.stats), act)
    act = ag.choose_action([], _frame(_g((2, 3, 3)), [1, 2]))
    assert act.value in (1, 2) and ag.stats["h5_exc"] == 1
    ag2 = _ag(mod)
    ag2.h5.guide = boom
    act = ag2.choose_action([], _frame(_g((2, 2, 3)), [1, 2]))
    assert ag2.h5 is None and ag2.stats["h5_exc"] == 1 and act.value in (1, 2)
    return "exception in observe or in guide: h5_exc=1, module switched off for the game, K keeps playing"


def test_h5_off_decides_like_v7():
    mod = ka._import_graph_explorer()
    seqs = []
    for on in (False, True):
        mod.H5_ON = on
        try:
            ag = _ag(mod)
        finally:
            mod.H5_ON = True
        assert (ag.h5 is None) == (not on)
        g = _g((2, 2, 3), (8, 8, 5))
        acts = []
        for i in range(30):
            gg = [r[:] for r in g]; gg[(i * 3) % 16][(i * 5) % 16] = 1 + i % 4
            a = ag.choose_action([], _frame(gg, [1, 2, 3, 4]))
            acts.append(a.value)
        seqs.append(acts)
        assert ag.stats["h5_calls"] == (30 if on else 0)
    return f"with H5_ON=False no call to the module; sequences: off={seqs[0][:8]}..., on={seqs[1][:8]}..."


def test_determinism_stable_order():
    mod = ka._import_graph_explorer()
    hs = [mod.Hyp("G4", "v", 1), mod.Hyp("O3", 3, 2), mod.Hyp("O1", 5, 2), mod.Hyp("O1", 4, 2), mod.Hyp("G1", (1, 2, 3, 4, 5, 6), 3)]
    ranked = sorted(hs, key=lambda h: h.order(None))
    assert [(h.fam, h.bind) for h in ranked][:2] == [("G1", (1, 2, 3, 4, 5, 6)), ("O1", 4)], [(h.fam, h.bind) for h in ranked]
    ranked2 = sorted(hs, key=lambda h: h.order("O3"))
    assert ranked2[0].fam == "O3", [(h.fam, h.bind) for h in ranked2]
    return "order by integer evidence, family and binding; k->k+1 composition promotes the corroborated family"


def test_f5_hint_only_with_h5_hint():
    from test_rt1_life_transition import FakeGame

    class Model:
        def _build_user_prompt(self, action_num, **kw):
            return "l1\nl2\nrest"
    texts = {}
    for flag in (False, True):
        h = ka.HybridAnalyzer(FakeGame(), common.Stub("x", "random"), **dict(common.HOOK_OPTS, h8="list", h5_hint=flag))
        ag = h.policy.agent
        ag.h5._new_level(0, _g((2, 2, 3)))
        ag.h5._add("O3", 3, 2)
        ag.h5.top()
        h.model = Model()
        h._patch_prompt()
        texts[flag] = h.model._build_user_prompt(1)
    assert "Win-condition hypothesis" not in texts[False], texts[False]
    assert "Win-condition hypothesis" in texts[True] and "make every cell with value 3 disappear" in texts[True], texts[True]
    assert "never forbids" in texts[True]
    return "F5 note only with h5_hint=True; text of the top instance first, marked as a guess that forbids no action"


TESTS = [test_f5_hint_only_with_h5_hint, test_empty_frame_not_applicable, test_recolor_does_not_instantiate_o3_removal_does, test_elimination_only_with_triggers_and_monotone,
         test_o1_only_expires_never_eliminated, test_credit_on_second_to_last_frame, test_exception_goes_to_error_and_k_continues,
         test_h5_off_decides_like_v7, test_determinism_stable_order]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
