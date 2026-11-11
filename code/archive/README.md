# code/archive

One-off scripts kept for reference. None is part of the pipeline in `code/master.do`.

- `6_first_stage.py` — traffic redistribution analysis (Lloyd's + AIS), kept for referee requests.
- `diag_threshold_sweep_2009.py` — (2026-11-11) read-only diagnostic: 2009 coefficient by distance-percentile cutoff and by vessel-count threshold; 2011 unweighted estimate on the weighted-table sample. Prints only.
- `diag_sample_groups_2009_2011.py` — (2026-11-11) read-only diagnostic: 2011 specification on the 2009 sample and its complement; mean biomass by year for cells with <2, 2-4 and >=4 vessels/month in Jan-Jun 2009. Prints only.
- `make_anonymous.py` — (2026-11-11) builds `paper/Kelp_anonymous.tex` from `paper/Kelp.tex` (removes authors, affiliations, thanks, PDF author field) for the double-blind ERE submission. Usage, from `paper/`: `python ../code/archive/make_anonymous.py Kelp.tex Kelp_anonymous.tex`.
