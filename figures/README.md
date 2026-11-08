# figures

- `make_figures.py`: generates Figures 1 and 2 of the paper as vector PDFs in the current folder
  (`python3 figures/make_figures.py`; needs matplotlib, pinned in `requirements.txt`).
- `fig1-modes-phase-a.pdf`: phase-A rounds per arm in two panels, the first batch (2 November) and the second (7 November;
  Table 3), on the same dfranzen base (Section 5, "Three modes of presence").
- `fig2-h5-identification.pdf`: H5 bench wins with and without a true hypothesis instance in the winning frame
  (Table 4; the counts can be recomputed from the bench data with `bench/table4.py`).

The numbers are written in the script, as in the paper. Figure 1's phase-A scores come from the Kaggle kernel runs shipped in
`../bench/results/phase-a/` (the second batch ran the four public notebooks of the main README; `python3 bench/phase_a.py`
recomputes the round means the script holds); Figure 2's counts come from the shipped bench results.
