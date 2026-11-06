# manifests

- `frozen-artifacts.tsv`: the SHA-256 digests of the frozen artifacts cited in the paper (Sections 5 and 10), one row
  per file: the submitted Duck-line entry v5 (ref. 57852519), the submitted dfranzen-line v14.1 (ref. 57923098), the
  phase-A ablation of 2 November, and the five kernels prepared for the public notebook and the later runs. Files of
  the development tree that are not released (build scripts, internal notebooks) are listed by role; their digests
  commit to the exact bytes without publishing them.
- `SHA256SUMS`: the digest of every file in this repository except itself (`sha256sum -c manifests/SHA256SUMS` from the
  repository root).

The two modules in `agent/` derive from the v14.1 rows marked "source of agent/...": same code, with English comments
and the public identifier names (see [`../agent/README.md`](../agent/README.md)).
