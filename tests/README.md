# tests

The regression suite and the mutants of the adversarial verification protocol (paper, Section 4), run against the
modules in [`../agent`](../agent).

- `test_*.py`: 17 files, 92 tests. Each file is a script; it prints one line per test (`PASSA`, `FALHA` or `ERRO`, as
  written during development) and exits non-zero if any test fails. A test name can be passed to run only that test.
  Most tests drive the real TAAF `HarnessSolver` offline, on public games, with a stub in place of the language model.
- `comum.py`: the shared harness setup, the model stub and the hook options of the Duck line (mode 1).
- `mutants.py`: 85 mutants (M1–M85; M33b and M37b are variants). Each re-introduces one fixed defect into a temporary
  copy of `agent/` and requires the protecting tests to fail. M44 and M46 are listed as retired: the code they undid
  was rewritten in a later version.

File names follow the round that produced them: `test_rt<N>_v<M>.py` holds the regressions for the red-team findings
on version M−1, fixed in version M; `test_i*.py` covers the pre-registered invariants (I1–I10). Test names and messages
are in Portuguese, the working language of the project.

## Running

The tests need the official starter clone (public games and `ARC-AGI-3-Agents`) and the TAAF source bundle; see the
main [README](../README.md#reproducing-the-results).

```bash
export ARC_STARTER=/path/to/ARC-AGI-3-Kaggle-Starter
export TAAF_SOURCE=/path/to/taaf-kaggle-source-share
for f in tests/test_*.py; do python "$f" || echo "FAILED: $f"; done   # about 5 minutes
python tests/mutants.py                                              # about 40 minutes
```

`KA_PATH` and `EXPLORER_MODULE_PATH` point the tests at other copies of the two modules (the mutant runner uses them).
Not included: the development tests that assemble Kaggle notebooks from internal build scripts, and the tests of a
later variant (v15) that is not part of the paper.
