* =================================================================
* 3_estim2011.do
*
* 2011 boundary modification PPML tables:
*   A. Unweighted (ship_exposure_kelp_2011_b.tex) — for main text
*      Restricted to cells with >= 2 vessels/month (count threshold).
*   B. Weighted   (ship_exposure_kelp_2011.tex) — for appendix
*      Same restricted sample, with vessel count weights.
*   C. Robustness: 2010 treatment (ship_exposure_kelp_2011_2010treat.tex)
*      Unweighted, same sample as A, 2010 treatment variables.
*
* Formatted to match the 2009 main table style with mgroups.
* =================================================================

global root "c:\Users\fcanda01\Desktop\Recherche\kelp"
global data "$root\data\generated"
global output "$root\output"
global fig "$output\figures"
global tab "$output\tables"

clear all
set scheme s2color
graph set window fontface "Arial"
set maxvar 30000

* --- Install packages if needed ---
foreach pkg in ppmlhdfe estout {
	cap which `pkg'
	if _rc {
		ssc install `pkg', replace
	}
}


* =================================================================
* DATA PREPARATION
* =================================================================
{
import delimited "$data\new_model_kelp_exposure_3.csv", clear

* Fixed effects
egen FE_i = group(eca_index_official year_eca)
egen FE_t = group(year quarter)

* Rename controls to short names (CSV has _mean suffix)
rename ln_sst_mean ln_sst
rename ln_precipitation_mean ln_precipitation
rename ln_swh_mean ln_swh

save "$data\temp_2011_panel.dta", replace
}


* =================================================================
* A. UNWEIGHTED 2011 PPML TABLE (main text)
*    Restricted to cells with >= 2 vessels/month (ln_mean_count >= ln(2))
* =================================================================
{
use "$data\temp_2011_panel.dta", clear

drop if year > 2016
drop if latitude < 32
drop if latitude > 36

gen policy_eca_2011 = (year > 2011)

duplicates drop FE_i FE_t, force

* No vessel count threshold for 2011: boundary modification redistributed
* shipping, so low-exposure cells are part of the identifying variation.
drop if missing(ln_mean_count_2011_20km)

di _n "Unweighted 2011 PPML sample: " _N " obs"
distinct FE_i
di "Cells: " r(ndistinct)

* --- Unweighted regressions ---
* Use generic `exposure_2011` so all 4 columns share a single coefficient
* name in esttab (one row for the interaction term).
gen exposure_2011 = ln_mean_length_2011_20km

eststo clear

eststo col1: ppmlhdfe biomass_sum ///
	c.exposure_2011##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh, ab(FE_i FE_t) vce(cluster FE_i)

eststo col2: ppmlhdfe area_sum ///
	c.exposure_2011##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh, ab(FE_i FE_t) vce(cluster FE_i)

* Switch to vessel count measure
replace exposure_2011 = ln_mean_count_2011_20km

eststo col3: ppmlhdfe biomass_sum ///
	c.exposure_2011##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh, ab(FE_i FE_t) vce(cluster FE_i)

eststo col4: ppmlhdfe area_sum ///
	c.exposure_2011##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh, ab(FE_i FE_t) vce(cluster FE_i)

* Tag stored estimates with indicator rows
foreach m in col1 col2 col3 col4 {
	estadd local cellfe "Yes" : `m'
	estadd local timefe "Yes" : `m'
}

* --- Export LaTeX table (main text) ---
#delimit ;
esttab col1 col2 col3 col4
	using "$tab\ship_exposure_kelp_2011_b.tex",
	replace
	label
	se star(* 0.10 ** 0.05 *** 0.01)
	keep(1.policy_eca_2011#c.exposure_2011
	     ln_sst ln_precipitation ln_swh)
	order(1.policy_eca_2011#c.exposure_2011
	      ln_sst ln_precipitation ln_swh)
	coeflabels(1.policy_eca_2011#c.exposure_2011
	               "Post 2011 $\times$ Exposure"
	           ln_sst "SST"
	           ln_precipitation "Precipitation"
	           ln_swh "Significant Wave Height")
	mgroups("Distance Traveled" "Vessel Count",
	        pattern(1 0 1 0)
	        prefix(\multicolumn{@span}{c}{)
	        suffix(})
	        span
	        erepeat(\cmidrule(lr){@span}))
	mtitles("Biomass" "Area" "Biomass" "Area")
	nonotes
	s(cellfe timefe N, label("Cell FE" "Year-Quarter FE" "Observations"))
;
#delimit cr

di _n "Unweighted 2011 table (main) saved: $tab\ship_exposure_kelp_2011_b.tex"
}


* =================================================================
* B. WEIGHTED 2011 PPML TABLE (appendix)
*    Same restricted sample as A, with vessel count weights.
* =================================================================
{
use "$data\temp_2011_panel.dta", clear

drop if year > 2016
drop if latitude < 32
drop if latitude > 36

gen policy_eca_2011 = (year > 2011)

duplicates drop FE_i FE_t, force

* No vessel count threshold for 2011 (same rationale as Section A)
drop if missing(ln_mean_count_2011_20km)

* Additionally drop obs with zero or missing weight
drop if missing(ln_sum_count_2011_20km)
drop if ln_sum_count_2011_20km <= 0

di _n "Weighted 2011 PPML sample: " _N " obs"
distinct FE_i
di "Cells: " r(ndistinct)

* --- Weighted regressions ---
* Use generic `exposure_2011` (matching Section A approach)
gen exposure_2011 = ln_mean_length_2011_20km

eststo clear

eststo col1w: ppmlhdfe biomass_sum ///
	c.exposure_2011##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh ///
	[pw = ln_sum_count_2011_20km], ab(FE_i FE_t) vce(cluster FE_i)

eststo col2w: ppmlhdfe area_sum ///
	c.exposure_2011##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh ///
	[pw = ln_sum_count_2011_20km], ab(FE_i FE_t) vce(cluster FE_i)

* Switch to vessel count measure
replace exposure_2011 = ln_mean_count_2011_20km

eststo col3w: ppmlhdfe biomass_sum ///
	c.exposure_2011##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh ///
	[pw = ln_sum_count_2011_20km], ab(FE_i FE_t) vce(cluster FE_i)

eststo col4w: ppmlhdfe area_sum ///
	c.exposure_2011##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh ///
	[pw = ln_sum_count_2011_20km], ab(FE_i FE_t) vce(cluster FE_i)

* Tag stored estimates with indicator rows
foreach m in col1w col2w col3w col4w {
	estadd local cellfe "Yes" : `m'
	estadd local timefe "Yes" : `m'
}

* --- Export LaTeX table (appendix) ---
#delimit ;
esttab col1w col2w col3w col4w
	using "$tab\ship_exposure_kelp_2011.tex",
	replace
	label
	se star(* 0.10 ** 0.05 *** 0.01)
	keep(1.policy_eca_2011#c.exposure_2011
	     ln_sst ln_precipitation ln_swh)
	order(1.policy_eca_2011#c.exposure_2011
	      ln_sst ln_precipitation ln_swh)
	coeflabels(1.policy_eca_2011#c.exposure_2011
	               "Post 2011 $\times$ Exposure"
	           ln_sst "SST"
	           ln_precipitation "Precipitation"
	           ln_swh "Significant Wave Height")
	mgroups("Distance Traveled" "Vessel Count",
	        pattern(1 0 1 0)
	        prefix(\multicolumn{@span}{c}{)
	        suffix(})
	        span
	        erepeat(\cmidrule(lr){@span}))
	mtitles("Biomass" "Area" "Biomass" "Area")
	nonotes
	s(cellfe timefe N, label("Cell FE" "Year-Quarter FE" "Observations"))
;
#delimit cr

di _n "Weighted 2011 table (appendix) saved: $tab\ship_exposure_kelp_2011.tex"
}


* =================================================================
* C. ROBUSTNESS: 2010 TREATMENT VARIABLE (unweighted PPML)
*    Uses 2010 shipping as purely pre-treatment intensity measure.
*    Same sample/specification as A, only treatment variable changes.
* =================================================================
{
use "$data\temp_2011_panel.dta", clear

drop if year > 2016
drop if latitude < 32
drop if latitude > 36

gen policy_eca_2011 = (year > 2011)

duplicates drop FE_i FE_t, force

* No vessel count threshold for 2011 (same rationale as Section A)
drop if missing(ln_mean_count_2011_20km)

di _n "Robustness 2010 treatment — Unweighted PPML sample: " _N " obs"
distinct FE_i
di "Cells: " r(ndistinct)

* --- Unweighted regressions with 2010 treatment ---
* Use generic `exposure_2010` (same approach as Sections A/B)
gen exposure_2010 = ln_mean_length_2010_20km

eststo clear

eststo r1: ppmlhdfe biomass_sum ///
	c.exposure_2010##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh, ab(FE_i FE_t) vce(cluster FE_i)

eststo r2: ppmlhdfe area_sum ///
	c.exposure_2010##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh, ab(FE_i FE_t) vce(cluster FE_i)

* Switch to vessel count measure
replace exposure_2010 = ln_mean_count_2010_20km

eststo r3: ppmlhdfe biomass_sum ///
	c.exposure_2010##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh, ab(FE_i FE_t) vce(cluster FE_i)

eststo r4: ppmlhdfe area_sum ///
	c.exposure_2010##policy_eca_2011 ///
	ln_sst ln_precipitation ln_swh, ab(FE_i FE_t) vce(cluster FE_i)

* Tag stored estimates with indicator rows
foreach m in r1 r2 r3 r4 {
	estadd local cellfe "Yes" : `m'
	estadd local timefe "Yes" : `m'
}

* --- Export LaTeX table ---
#delimit ;
esttab r1 r2 r3 r4
	using "$tab\ship_exposure_kelp_2011_2010treat.tex",
	replace
	label
	se star(* 0.10 ** 0.05 *** 0.01)
	keep(1.policy_eca_2011#c.exposure_2010
	     ln_sst ln_precipitation ln_swh)
	order(1.policy_eca_2011#c.exposure_2010
	      ln_sst ln_precipitation ln_swh)
	coeflabels(1.policy_eca_2011#c.exposure_2010
	               "Post 2011 $\times$ Exposure (2010)"
	           ln_sst "SST"
	           ln_precipitation "Precipitation"
	           ln_swh "Significant Wave Height")
	mgroups("Distance Traveled" "Vessel Count",
	        pattern(1 0 1 0)
	        prefix(\multicolumn{@span}{c}{)
	        suffix(})
	        span
	        erepeat(\cmidrule(lr){@span}))
	mtitles("Biomass" "Area" "Biomass" "Area")
	nonotes
	s(cellfe timefe N, label("Cell FE" "Year-Quarter FE" "Observations"))
;
#delimit cr

di _n "2010 treatment robustness table saved: $tab\ship_exposure_kelp_2011_2010treat.tex"
}


* =================================================================
* CLEANUP
* =================================================================
cap erase "$data\temp_2011_panel.dta"

di _n "=========================================="
di "DONE. Outputs:"
di "  Unweighted (main text):  $tab\ship_exposure_kelp_2011_b.tex"
di "  Weighted (appendix):     $tab\ship_exposure_kelp_2011.tex"
di "  2010 treatment robustness: $tab\ship_exposure_kelp_2011_2010treat.tex"
di "=========================================="
