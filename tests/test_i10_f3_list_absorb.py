"""I10 (H8 list never empties the list) and B1-F3 (K's ignored proposal leaves no effect of an unexecuted RESET).
    python tests/test_i10_f3_list_absorb.py
"""
import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
from arcengine import GameState  # noqa: E402

ENG = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
MOD = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]


class FakeGame:
    """Minimal game with the view that KPolicy reads (game.current_state.raw)."""
    def __init__(self, name="ls20-x", avail=(1, 2, 3, 4)):
        self.env_name = name
        g = [[0] * 16 for _ in range(16)]
        g[8][8] = 5
        raw = SimpleNamespace(frame=[g], state=GameState.NOT_FINISHED, levels_completed=0, win_levels=3,
                              available_actions=list(avail), game_id=name)
        self.current_state = SimpleNamespace(raw=raw, levels_completed=0, available_actions=[0, *avail])


def _hyb(**kw):
    o = dict(common.HOOK_OPTS)
    o.update(kw)
    return ka.HybridAnalyzer(FakeGame(), common.Stub("x", "random"), **o)


def test_list_never_empty_property():
    h = _hyb()
    rng = random.Random(7)
    key = 12345
    h.policy.agent.prev_key = key
    cases = 0
    for n in range(1, 7):
        for valid in itertools.combinations(ENG + MOD, n):
            valid = list(valid)
            for _ in range(4):
                h.certs = {(key, (a,)): 1 for a in (1, 2, 3, 4, 5, 7) if rng.random() < 0.6}
                out = h._filter_valid(list(valid))
                cases += 1
                assert out, f"empty list: valid={valid} certs={list(h.certs)}"
                assert set(out) <= set(valid), (valid, out)
                names = {ka._MODEL_NAME[a[0]] for (_, a) in h.certs}
                removed = [v for v in valid if v not in out]
                assert all(ka._MODEL_NAME[ka._ENGINE_ID[v]] in names for v in removed), (valid, out)
                if all(ka._MODEL_NAME[ka._ENGINE_ID[v]] in names for v in valid):
                    assert out == valid, "all certified: the list must come back whole"
    # all keys certified and no MOUSE: comes back whole
    h.certs = {(key, (a,)): 1 for a in (1, 2, 3, 4, 5)}
    assert h._filter_valid(["ACTION1", "ACTION2"]) == ["ACTION1", "ACTION2"]
    assert h._filter_valid([]) == []
    return f"{cases} random cases: never empty, removes only certified keys, comes back whole when everything is certified"


def test_list_integration_ls20():
    an = {}
    bm = common.run_games(["ls20"], common.make_factory(ka, an, mode="random"), max_actions=200, conc=1)
    a = an["ls20"][0]
    empty = sum(1 for v in a.model.valid_seen if not v)
    assert empty == 0, f"{empty} model turns with an empty list"
    assert a.stats["h8_masked_keys"] > 0 and a.stats["cert_emitted"] > 0, f"vacuous pass: {a.stats}"
    return (f"{len(a.model.valid_seen)} turns, 0 empty lists; cert_emitted={a.stats['cert_emitted']} "
            f"h8_masked_keys={a.stats['h8_masked_keys']} actions={len(bm.game_runs[0].history)}")


def _payload():
    return {"executed": True, "executed_count": 1, "executed_actions": ["UP"], "board_changed": False,
            "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT"]}


def test_absorb_model_no_phantom_reset():
    """Model turn (E4): K, when processing the frame, would decide RESET for life at the end; the RESET does not happen."""
    h = _hyb()
    pol, ag = h.policy, h.policy.agent
    pol.propose(pol.current_frame())                 # K sees the first frame (current node)
    ag.prev_act = None
    ag.life_budget, ag.life, ag.hud_reset = 5, 4, False
    w = h._wrap_step_env(lambda args: _payload())
    w({"actions": [{"action": "UP"}]})
    assert ag.stats["reset_life_budget_exhausted"] >= 1, "scenario did not trigger the RESET decision (vacuous pass)"
    assert ag.prev_key is not None, "K lost the current node (phantom RESET)"
    assert ag.life == 5, f"life = {ag.life} (should count the model's action: 5)"
    assert ag.hud_reset is False, "hud_reset set without RESET"
    return f"life={ag.life} prev_key kept, hud_reset={ag.hud_reset}"


def test_absorb_k_turn():
    """K turn: after its own action, processing the frame would decide RESET; life must not reset to zero."""
    h = _hyb()
    pol, ag = h.policy, h.policy.agent
    pol.propose(pol.current_frame())
    ag.prev_act = None
    ag.life_budget, ag.life = 3, 1
    h._k_turn(lambda args: _payload(), "k_stall", None, 1)
    assert ag.stats["reset_life_budget_exhausted"] >= 1, "scenario did not trigger the RESET decision (vacuous pass)"
    assert ag.prev_key is not None and ag.life == 2, f"life={ag.life} prev_key={ag.prev_key}"
    return f"life={ag.life} after K's turn"


TESTS = [test_list_never_empty_property, test_list_integration_ls20, test_absorb_model_no_phantom_reset, test_absorb_k_turn]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
