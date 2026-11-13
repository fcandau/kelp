# code/archive

One-off scripts kept for reference. None is part of the pipeline in `code/master.do`.

- `6_first_stage.py` — traffic redistribution analysis (Lloyd's + AIS), kept for referee requests.
- `diag_threshold_sweep_2009.py` — (2026-11-11) read-only diagnostic: 2009 coefficient by distance-percentile cutoff and by vessel-count threshold; 2011 unweighted estimate on the weighted-table sample. Prints only.
- `diag_sample_groups_2009_2011.py` — (2026-11-11) read-only diagnostic: 2011 specification on the 2009 sample and its complement; mean biomass by year for cells with <2, 2-4 and >=4 vessels/month in Jan-Jun 2009. Prints only.
- `make_anonymous.py` — (2026-11-11) builds `paper/Kelp_anonymous.tex` from `paper/Kelp.tex` (removes authors, affiliations, thanks, PDF author field) for the double-blind ERE submission. Usage, from `paper/`: `python ../code/archive/make_anonymous.py Kelp.tex Kelp_anonymous.tex`.
- `make_plain_text.py` — (2026-11-13) builds `paper/Kelp_plain.txt` from `paper/Kelp.tex` via pandoc: prose only, one paragraph per line, citations as (Author year), for pasting into a text checker. Usage in its docstring.
- `diag_common_sample_2009_2011.py` — (2026-11-13) read-only diagnostic after the CJE referee report: 2009 and 2011 designs, and a joint regression, on common samples (thresholds on the 2009 count, the 2011 count, both, either). Prints only.
- `event_study_2011_ppml.py` — (archived 2026-11-13) PPML event study of the 2011 boundary modification. Its subsection ("Dynamic effects", Table 3) was removed from the paper; outputs `output/tables/tab_es2011_panels.tex` and `tab_es_2011.tex` are kept. Run it again if a referee asks for the 2011 dynamics. The removed text is in `paper/archive/Kelp_2026-11-13_with_2011_dynamics_and_valuation.tex`.
- `valuation_aggregate.py` — (archived 2026-11-13) aggregate ecosystem-service valuation (63-135 ha, $5-10M per year). The sentence and footnote were removed from the Conclusion; the removed text is in the same archived `.tex`.
