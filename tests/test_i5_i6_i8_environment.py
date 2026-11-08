"""I5 (environment read at the right moment), B1-F4 (attributes the harness reads from the analyzer), I6 (time), I8 (no
network, no game source, no key).
    python tests/test_i5_i6_i8_environment.py
"""
import sysconfig
import tempfile
import json
import os
import re
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ka = common.load_ka()
STDLIB = sysconfig.get_paths()["stdlib"]
SETUP = common.TAAF / "setup_commands.json"


class _G:
    env_name = "ls20-x"


def test_k_agent_path_read_at_construction():
    """EXPLORER_MODULE_PATH / EXPLORER_MODULE_PATH set AFTER the analyzer import must take effect in the next KPolicy."""
    src = common.KA_FILE.read_text(encoding="utf-8")
    var_name = "EXPLORER_MODULE_PATH" if "EXPLORER_MODULE_PATH" in src else "EXPLORER_MODULE_PATH"
    expected_f0 = "graph_explorer.py" if var_name == "EXPLORER_MODULE_PATH" else "graph_explorer.py"
    old = os.environ.pop(var_name, None)
    try:
        p0 = ka.KPolicy(_G())
        alt = Path(tempfile.mkdtemp()) / "alt_explorer.py"
        alt.write_text((common.AGENT / "graph_explorer.py").read_text(encoding="utf-8"), encoding="utf-8")
        os.environ[var_name] = str(alt)
        p1 = ka.KPolicy(_G())
    finally:
        os.environ.pop(var_name, None)
        if old is not None:
            os.environ[var_name] = old
    f0, f1 = (Path(p.cls.choose_action.__code__.co_filename).name for p in (p0, p1))
    assert f0 == expected_f0 and f1 == "alt_explorer.py", (f0, f1)
    return f"without variable: {f0}; with variable set after the import: {f1}"


def test_attributes_passed_to_harness():
    """solver.py reads from the analyzer `_timeout` (cap of the model request) and `generated_tokens`/`total_tokens`."""
    st = common.Stub("x", "random", timeout=900.0)
    h = ka.HybridAnalyzer(_G(), st, **common.HOOK_OPTS)
    st.total_tokens, st.generated_tokens = 1000, 40
    assert getattr(h, "_timeout", None) == 900.0, getattr(h, "_timeout", None)
    assert common.solver_mod._analyzer_reported_tokens(h) == 40, common.solver_mod._analyzer_reported_tokens(h)
    # integration: with 100 000 s of game time, the model request must not wait longer than the ToolAgent _timeout
    seen = []
    orig = common.Stub.analyze

    def spy(self, *a, **k):
        seen.append(k.get("request_timeout_seconds"))
        return orig(self, *a, **k)
    common.Stub.analyze = spy
    try:
        common.run_games(["ls20"], common.make_factory(ka, {}, mode="random"), max_actions=10, conc=1, max_runtime=100000.0)
    finally:
        common.Stub.analyze = orig
    assert seen and max(seen) <= 900.0, f"request timeout = {max(seen):.0f} s"
    return f"_timeout=900 passed through; harness tokens = generated_tokens; model request <= {max(seen):.0f} s"


def test_time_k_time_and_cap():
    """K takes over by time (time_floor) and the game stops at the harness max_runtime_s_per_game, with no action cap."""
    an = {}
    t = time.monotonic()
    bm = common.run_games(["ls20"], common.make_factory(ka, an, mode="random", max_runtime_s=6.0, time_floor_s=4.0),
                          max_actions=None, conc=1, max_runtime=6.0)
    dt = time.monotonic() - t
    a = an["ls20"][0]
    assert a.stats["k_time"] > 0, a.stats
    assert dt < 30, f"game lasted {dt:.1f}s with a 6 s cap"
    return f"k_time={a.stats['k_time']} in {dt:.1f}s (cap 6 s), actions={len(bm.game_runs[0].history)}"


def test_k_latency():
    """Contract of yielding every <= 5 s: worst K call (propose/absorb) over 4 games x 300 actions, model stalled."""
    worst = {}
    orig = ka.KPolicy.propose

    def timed(self, *a, **k):
        t = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            g = self.game.env_name.split("-")[0]
            worst[g] = max(worst.get(g, 0.0), time.perf_counter() - t)
    ka.KPolicy.propose = timed
    try:
        common.run_games(["ls20", "vc33", "sp80", "ft09"], common.make_factory(ka, {}, mode="mix"), max_actions=300, conc=4)
    finally:
        ka.KPolicy.propose = orig
    assert max(worst.values()) < 5.0, worst
    return "worst K call per game (s): " + ", ".join(f"{g}={v:.3f}" for g, v in sorted(worst.items()))


def test_no_network_no_source_no_key():
    """Audit hook: no socket/urllib event and no open() of a game source, of a credential or of anything unexpected
    with a hybrid_analyzer or graph_explorer frame on the stack."""
    ev = []

    def hook(event, args):
        if not (event == "open" or event.startswith("socket.") or event.startswith("urllib") or event == "http.client.connect"):
            return
        f = sys._getframe(1)
        while f is not None and (f.f_code.co_filename.startswith(STDLIB) or f.f_code.co_filename.startswith("<")):
            f = f.f_back                                   # the first frame outside the stdlib is the caller
        fn = f.f_code.co_filename if f is not None else ""
        if fn == str(common.KA_FILE) or fn.endswith("/graph_explorer.py") or fn.endswith("/alt_explorer.py"):
            ev.append((event, str(args[0]) if args else ""))
    sys.addaudithook(hook)
    common.run_games(["ls20", "vc33"], common.make_factory(ka, {}, mode="mix"), max_actions=60, conc=2)
    network = [e for e in ev if e[0] != "open"]
    opened = sorted({p for e, p in ev if e == "open"})
    bad = [p for p in opened if not (p.endswith("graph_explorer.py") or p.endswith(".pyc") or "/transcripts/" in p)]
    assert not network, network[:3]
    assert not bad, bad
    return f"network: 0 events; files opened by the layer: {[Path(p).name for p in opened] or 'none'}"


TESTS = [test_k_agent_path_read_at_construction, test_attributes_passed_to_harness,
         test_time_k_time_and_cap, test_k_latency, test_no_network_no_source_no_key]
if __name__ == "__main__":
    sys.exit(common.main(TESTS))
