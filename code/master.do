/*==============================================================================
  master.do

  Runs the full estimation pipeline in order.
  Python scripts are called via Stata's -shell- command.

  Run order:
    1_build_data.py        → data/generated/new_model_kelp_exposure_3.csv
    2_a_estim.py           → Main PPML tables, event study figures, placebo test
    2_b_robustness.py      → Robustness event studies (no SWH, cell trends, etc.)
    2_c_anticipation.py    → Anticipation robustness (temporal stability, alt timing)
    2_e_weighting_robustness.py → Weighting diagnostic figure + table
    3_estim2011.do         → 2011 boundary modification PPML tables (Stata)
    2_d_dechaisemartin.do  → de Chaisemartin & D'Haultfoeuille event study (Stata)
    4_balance_table.py     → Balance table by treatment quartile
    5_financial_crisis.py  → Financial crisis robustness table + figure
==============================================================================*/

clear all
set more off

* --- Paths -------------------------------------------------------------------
global root   "c:\Users\fcanda01\Desktop\Recherche\kelp"
global code   "$root\code"
global data   "$root\data\generated"
global output "$root\output"
global python "C:\Users\fcanda01\AppData\Local\anaconda3\envs\r-env\python.exe"


* ==============================================================================
*   Step 1: Data preparation (Python)
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 1: Data preparation (1_build_data.py)"
di as txt "============================================"

shell "$python" "$code\1_build_data.py"

cap confirm file "$data\new_model_kelp_exposure_3.csv"
if _rc {
    di as error "ERROR: new_model_kelp_exposure_3.csv not found. Step 1 failed."
    error 601
}
di as txt "Step 1 complete."


* ==============================================================================
*   Step 2a: Main PPML estimation + event studies + placebo (Python)
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 2a: Main estimation (2_a_estim.py)"
di as txt "============================================"

shell "$python" "$code\2_a_estim.py"

cap confirm file "$output\tables\ship_exposure_kelp_2009_main.tex"
if _rc {
    di as error "ERROR: ship_exposure_kelp_2009_main.tex not found. Step 2a failed."
    error 601
}
cap confirm file "$output\figures\event_study_main.png"
if _rc {
    di as error "WARNING: event_study_main.png not found."
}
di as txt "Step 2a complete."


* ==============================================================================
*   Step 2b: Robustness event studies (Python)
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 2b: Robustness (2_b_robustness.py)"
di as txt "============================================"

shell "$python" "$code\2_b_robustness.py"

di as txt "Step 2b complete."


* ==============================================================================
*   Step 2c: Anticipation robustness (Python)
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 2c: Anticipation (2_c_anticipation.py)"
di as txt "============================================"

shell "$python" "$code\2_c_anticipation.py"

cap confirm file "$output\tables\anticipation_robustness.tex"
if _rc {
    di as error "WARNING: anticipation_robustness.tex not found."
}
cap confirm file "$output\figures\temporal_stability_shipping.png"
if _rc {
    di as error "WARNING: temporal_stability_shipping.png not found."
}
di as txt "Step 2c complete."


* ==============================================================================
*   Step 2e: Weighting robustness (Python)
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 2e: Weighting robustness (2_e_weighting_robustness.py)"
di as txt "============================================"

shell "$python" "$code\2_e_weighting_robustness.py"

cap confirm file "$output\figures\weighting_diagnostic.png"
if _rc {
    di as error "WARNING: weighting_diagnostic.png not found."
}
cap confirm file "$output\tables\weighting_robustness.tex"
if _rc {
    di as error "WARNING: weighting_robustness.tex not found."
}
di as txt "Step 2e complete."


* ==============================================================================
*   Step 3: 2011 boundary modification PPML tables (Stata)
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 3: 2011 PPML tables (3_estim2011.do)"
di as txt "============================================"

do "$code\3_estim2011.do"

cap confirm file "$output\tables\ship_exposure_kelp_2011_b.tex"
if _rc {
    di as error "ERROR: ship_exposure_kelp_2011_b.tex not found. Step 3 failed."
    error 601
}
di as txt "Step 3 complete."


* ==============================================================================
*   Step 4: de Chaisemartin & D'Haultfoeuille event study (Stata)
*   NOTE: This step runs 999 bootstrap replications and takes ~2 hours.
*         It is neutralized by default. Uncomment the "do" line below to
*         regenerate output/figures/event_study_2011_dCdH.png.
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 4: dCdH event study (2_d_dechaisemartin.do)"
di as txt "  *** SKIPPED (slow — uncomment to run) ***"
di as txt "============================================"

* do "$code\2_d_dechaisemartin.do"

cap confirm file "$output\figures\event_study_2011_dCdH.png"
if _rc {
    di as error "WARNING: event_study_2011_dCdH.png not found."
}
di as txt "Step 4 complete (skipped)."


* ==============================================================================
*   Step 5: Balance table (Python)
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 5: Balance table (4_balance_table.py)"
di as txt "============================================"

shell "$python" "$code\4_balance_table.py"

cap confirm file "$output\tables\balance_table.tex"
if _rc {
    di as error "WARNING: balance_table.tex not found."
}
di as txt "Step 5 complete."


* ==============================================================================
*   Step 6: Financial crisis robustness (Python)
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Step 6: Financial crisis (5_financial_crisis.py)"
di as txt "============================================"

shell "$python" "$code\5_financial_crisis.py"

cap confirm file "$output\tables\crisis_robustness.tex"
if _rc {
    di as error "WARNING: crisis_robustness.tex not found."
}
cap confirm file "$output\figures\crisis_decomposition.png"
if _rc {
    di as error "WARNING: crisis_decomposition.png not found."
}
di as txt "Step 6 complete."


* ==============================================================================
*   Done
* ==============================================================================
di as txt ""
di as txt "============================================"
di as txt "  Pipeline complete"
di as txt "============================================"
di as txt ""
di as txt "Main-text outputs:"
di as txt "  $output\tables\ship_exposure_kelp_2009_main.tex"
di as txt "  $output\figures\event_study_main.png"
di as txt "  $output\tables\ship_exposure_kelp_2011_b.tex"
di as txt "  $output\figures\event_study_2011_dCdH.png"
di as txt ""
di as txt "Appendix outputs:"
di as txt "  $output\tables\cdid_weighted.tex"
di as txt "  $output\tables\placebo_test.tex"
di as txt "  $output\tables\anticipation_robustness.tex"
di as txt "  $output\tables\ship_exposure_kelp_2011.tex"
di as txt "  $output\tables\weighting_robustness.tex"
di as txt "  $output\tables\balance_table.tex"
di as txt "  $output\tables\crisis_robustness.tex"
di as txt "  $output\figures\temporal_stability_shipping.png"
di as txt "  $output\figures\treatment_correlation_2009_2011.png"
di as txt "  $output\figures\weighting_diagnostic.png"
di as txt "  $output\figures\crisis_decomposition.png"
di as txt ""
di as txt "Manually created figures (not produced by this pipeline):"
di as txt "  $output\figures\shipping_exposure_2009.png   (AIS traffic map)"
di as txt "  $output\figures\shipping_exposure_2011.png   (AIS traffic map)"
di as txt "  $output\figures\shipping_exposure_png.png    (AIS traffic map, after 2011)"
di as txt "  $output\figures\SC.png                       (co-author: synthetic control)"
di as txt "  $output\figures\map_SC.png                   (co-author: SC donor map)"
di as txt "  $output\figures\weigth_sc.png                (co-author: SC donor weights)"
di as txt "  $output\tables\resultats_sc.tex              (co-author: SC results table)"
