* =================================================================
* 2_d_dechaisemartin.do
*
* de Chaisemartin & D'Haultfoeuille estimators for the 2011 ECA
* boundary modification.
*
* IDENTIFICATION NOTE: The 2011 treatment is universal (all cells
* treated simultaneously). Variation comes from cross-sectional
* differences in shipping intensity. The continuous(1) option
* assumes baseline outcome evolution is linear in the initial
* treatment level, enabling estimation with continuous treatments.
*
* NOTE: We use annual frequency to avoid seasonal heteroscedasticity
* in quarterly kelp data (Q1/Q4 winter quarters have ~10x smaller
* variance than Q2/Q3 spring/summer, making quarterly event studies
* misleading).
*
* LIMITATION: did_multiplegt_dyn runs OLS, while the main spec uses
* unweighted PPML.
*
* Reference: de Chaisemartin & D'Haultfoeuille (2024, ReStud)
* =================================================================

global root "c:\Users\fcanda01\Desktop\Recherche\kelp"
global data "$root\data\generated"
global output "$root\output"
global fig "$output\figures"
global csv "$output\csv"

cap mkdir "$fig"
cap mkdir "$csv"

clear all
set scheme s2color
graph set window fontface "Arial"
set maxvar 30000

* --- Install packages if needed ---
cap which did_multiplegt_dyn
if _rc {
    ssc install did_multiplegt_dyn, replace
}
cap which did_multiplegt_stat
if _rc {
    ssc install did_multiplegt_stat, replace
}


* =================================================================
* 1. DATA PREPARATION
* =================================================================
{
import delimited "$data\new_model_kelp_exposure_3.csv", clear

* Rename controls to short names (CSV has _mean suffix)
rename ln_sst_mean ln_sst
rename ln_precipitation_mean ln_precipitation
rename ln_swh_mean ln_swh

* Sample restrictions (matching Python 2_a_estim.py)
keep if latitude >= 32 & latitude <= 36
keep if year <= 2016

* Cell identifier (matching Python: eca_index_official + year_eca)
egen FE_i = group(eca_index_official year_eca)

* Quarterly time variable (needed for balancing before collapse)
gen quarter_num = (year - 2005) * 4 + quarter

* Drop duplicates (cell x quarter) — report count
duplicates tag FE_i quarter_num, gen(_dup)
count if _dup > 0
local n_dup = r(N)
di "Duplicates dropped: `n_dup'"
duplicates drop FE_i quarter_num, force
drop _dup

* Drop missing key variables
local n_before = _N
foreach v in biomass_sum area_sum ///
    ln_mean_length_2011_20km ln_mean_count_2011_20km ///
    ln_sst ln_precipitation ln_swh {
    drop if missing(`v')
}
di "Obs dropped for missing values: " `n_before' - _N

* No vessel count threshold for 2011: boundary modification redistributed
* shipping, so low-exposure cells are part of the identifying variation.

* Sample: 2008-2016. SWH now available through 2016 (ERA5 extended download).
keep if year >= 2008 & year <= 2016

* --- Treatment variables ---
* D_2011: 0 before 2011, = shipping intensity from 2011 onward
gen D_2011_dist = 0
replace D_2011_dist = ln_mean_length_2011_20km if year >= 2011

* D_2009: 0 before 2009, = shipping intensity from 2009 onward
* (used as other_treatment in did_multiplegt_stat to absorb the 2009 ECA)
gen D_2009_dist = 0
replace D_2009_dist = ln_mean_length_2011_20km if year >= 2009

* --- Balance quarterly panel, then collapse to annual ---
bysort FE_i: gen n_periods = _N
su n_periods, detail
local max_periods = r(max)
keep if n_periods == `max_periods'
drop n_periods

* Collapse to annual means
collapse (mean) biomass_sum area_sum ///
    D_2011_dist D_2009_dist ln_sst ln_precipitation ln_swh, ///
    by(FE_i year)

* Annual time variable (sequential: 2008=1, 2009=2, ..., 2016=9)
gen year_num = year - 2007

* Balance the annual panel
bysort FE_i: gen n_yr = _N
su n_yr, detail
local max_yr = r(max)
keep if n_yr == `max_yr'
drop n_yr

distinct FE_i
local n_cells = r(ndistinct)
di _n "Annual panel: `n_cells' cells × `max_yr' years"
di "Total obs: " _N

sort FE_i year_num

save "$data\dechaisemartin_panel.dta", replace
}


* =================================================================
* 2. HELPER: Extract did_multiplegt_dyn results
*    Convention: Effect_1 = event time 0, Effect_2 = +1, etc.
* =================================================================
cap program drop extract_dyn
program define extract_dyn
    args filename

    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local nc = colsof(`b')
    local cnames : colnames `b'

    preserve
    clear
    set obs `nc'
    gen period = .
    gen beta = .
    gen se = .

    forvalues j = 1/`nc' {
        local cn : word `j' of `cnames'
        local cn_low = lower("`cn'")

        qui replace beta = `b'[1, `j'] in `j'
        qui replace se   = sqrt(`V'[`j', `j']) in `j'

        if substr("`cn_low'", 1, 8) == "placebo_" {
            local num = real(substr("`cn_low'", 9, .))
            qui replace period = -`num' in `j'
        }
        else if substr("`cn_low'", 1, 7) == "effect_" {
            local num = real(substr("`cn_low'", 8, .))
            qui replace period = `num' - 1 in `j'
        }
    }

    drop if period == .
    sort period
    gen lo = beta - 1.96 * se
    gen hi = beta + 1.96 * se

    save "`filename'", replace
    restore
end


* =================================================================
* 3. did_multiplegt_dyn — ANNUAL EVENT STUDY
*    Pre-treatment:  2009, 2010 (placebo(2)) — continuous(1) uses
*      2008 to fit the baseline polynomial, freeing 2 placebos
*    Post-treatment: effects(6) for 2008-2016
* =================================================================
{
use "$data\dechaisemartin_panel.dta", clear

di _n "=========================================="
di "did_multiplegt_dyn ANNUAL: Biomass"
di "=========================================="

did_multiplegt_dyn biomass_sum FE_i year_num D_2011_dist, ///
    effects(6) placebo(2) ///
    controls(ln_sst ln_precipitation ln_swh) ///
    continuous(1) ///
    cluster(FE_i) graph_off bootstrap(999, 12345)

extract_dyn "$data\dyn_2011_annual_bio.dta"

use "$data\dechaisemartin_panel.dta", clear

di _n "=========================================="
di "did_multiplegt_dyn ANNUAL: Area"
di "=========================================="

did_multiplegt_dyn area_sum FE_i year_num D_2011_dist, ///
    effects(6) placebo(2) ///
    controls(ln_sst ln_precipitation ln_swh) ///
    continuous(1) ///
    cluster(FE_i) graph_off bootstrap(999, 12345)

extract_dyn "$data\dyn_2011_annual_area.dta"
}


* =================================================================
* 4. EVENT STUDY FIGURE
* =================================================================
{
use "$data\dyn_2011_annual_bio.dta", clear

qui su period
local xmin = r(min)
local xmax = r(max)

twoway (rcap lo hi period if !missing(se), lcolor(navy) lwidth(medium)) ///
       (scatter beta period, mcolor(navy) msymbol(O) msize(medium)), ///
    yline(0, lcolor(black) lwidth(thin)) ///
    xline(-0.5, lcolor(gs8) lpattern(dash) lwidth(medium)) ///
    xtitle("Event time (years)") ///
    ytitle("Effect on kelp biomass") ///
    title("Panel A: Biomass", size(medium)) ///
    graphregion(color(white)) bgcolor(white) ///
    legend(off) ///
    xlabel(`xmin'(1)`xmax', labsize(small) grid) ///
    name(bio_ann, replace)

use "$data\dyn_2011_annual_area.dta", clear

qui su period
local xmin = r(min)
local xmax = r(max)

twoway (rcap lo hi period if !missing(se), lcolor(navy) lwidth(medium)) ///
       (scatter beta period, mcolor(navy) msymbol(O) msize(medium)), ///
    yline(0, lcolor(black) lwidth(thin)) ///
    xline(-0.5, lcolor(gs8) lpattern(dash) lwidth(medium)) ///
    xtitle("Event time (years)") ///
    ytitle("Effect on kelp area") ///
    title("Panel B: Area", size(medium)) ///
    graphregion(color(white)) bgcolor(white) ///
    legend(off) ///
    xlabel(`xmin'(1)`xmax', labsize(small) grid) ///
    name(area_ann, replace)

graph combine bio_ann area_ann, rows(1) ///
    graphregion(color(white)) ///
    title("2011 Boundary Modification — dCdH Event Study", size(small)) ///
    note("did_multiplegt_dyn, continuous(1). Annual 2008-2016, 999 bootstrap reps." ///
         "Controls: SST, precip, SWH. Clustered SEs (cell). Dashed line = treatment onset.", size(vsmall))

graph export "$fig\event_study_2011_dCdH.png", replace width(3000)
graph drop bio_ann area_ann


* --- Export coefficient CSVs ---
foreach spec in bio area {
    cap confirm file "$data\dyn_2011_annual_`spec'.dta"
    if !_rc {
        use "$data\dyn_2011_annual_`spec'.dta", clear
        export delimited "$csv\dyn_2011_annual_`spec'.csv", replace
        erase "$data\dyn_2011_annual_`spec'.dta"
    }
}

di _n "Event study figure saved."
}


* =================================================================
* 5. CLEANUP
* =================================================================
cap erase "$data\dechaisemartin_panel.dta"

di _n "=========================================="
di "DONE. Outputs:"
di "  Figure:  $fig\event_study_2011_dCdH.png"
di "  CSVs:    $csv\dyn_2011_annual_bio.csv"
di "           $csv\dyn_2011_annual_area.csv"
di "=========================================="
