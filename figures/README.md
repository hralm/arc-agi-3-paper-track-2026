# figures

- `make_diagrams.py`: generates Figures 1 and 2 of the paper, two diagrams with no data, as vector PDFs in the current folder
  (`python3 figures/make_diagrams.py`; needs matplotlib, pinned in `requirements.txt`).
- `diagram-behavioral-classification.pdf` (Figure 1): the behavioral classification of systems in Gharajedaghi (2011,
  p. 36, Table 2.1), after Ackoff, with the reading of Section 1 beside it.
- `diagram-certificate-and-modes.pdf` (Figure 2): the no-effect certificate and the hidden-state mark (Section 3.2), and
  where K acts in each mode of presence (Sections 3.1 and 5).
- `make_figures.py`: generates Figures 3 and 4 of the paper as vector PDFs in the current folder
  (`python3 figures/make_figures.py`).
- `fig1-modes-phase-a.pdf` (Figure 3): phase-A rounds per arm in two panels, the first batch (2 November) and the second
  (7 November; Table 3), on the same dfranzen base (Section 5, "Three modes of presence").
- `fig2-h5-identification.pdf` (Figure 4): H5 bench wins with and without a true hypothesis instance in the winning frame
  (Table 4; the counts can be recomputed from the bench data with `bench/table4.py`).

The numbers are written in the script, as in the paper. Figure 3's phase-A scores come from the Kaggle kernel runs shipped in
`../bench/results/phase-a/` (the second batch ran the four public notebooks of the main README; `python3 bench/phase_a.py`
recomputes the round means the script holds); Figure 4's counts come from the shipped bench results.
