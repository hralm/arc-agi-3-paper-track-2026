# figures

- `make_figures.py`: generates Figures 1 and 2 of the paper as vector PDFs in the current folder
  (`python3 figures/make_figures.py`; needs matplotlib, pinned in `requirements.txt`).
- `fig1-modes-phase-a.pdf`: phase-A runs per arm (Section 5, "Three modes of presence").
- `fig2-h5-identification.pdf`: H5 bench wins with and without a true hypothesis instance in the winning frame
  (Table 3; the counts can be recomputed from the bench data with `bench/table3.py`).

The numbers are written in the script, as in the paper. Figure 1's phase-A scores come from Kaggle kernel runs whose
raw outputs (logs) are not released; Figure 2's counts come from the shipped bench results.
