"""Annotate the missense second variants with published pendrin transport measurements.

In the patients with one c.919-2A>G allele, the amino-acid substitution of each missense second
variant (from config.VARIANTS) is matched exactly to the substitutions measured by Wasano et al. 2020
(Hum Mutat 41:316-331). The script writes the matched transport values and counts the patients and
distinct variants with a match. The values are annotations only; they are not tested against the
phenotype.

The HCO3-/Cl- and I-/Cl- dose-response slopes (+/- SE) below were transcribed from Wasano 2020
Table 1 (open access, PMC6930342). Where the source reports only 'slope < 0' (I-/Cl- for His723Arg
and Thr410Met), the continuous value is left missing with the qualitative 'impaired' label; negative
slopes are never truncated to zero.

Usage: python -m variants.pendrin_transport
"""
import json, hashlib
from pathlib import Path
import pandas as pd

import config

C919 = "c.919-2A>G"

# Wasano 2020 Table 1: HCO3-/Cl- and I-/Cl- dose-response slopes (slope +/- SE) for wild type and the
# cohort's missense variants. Keyed by normalized human amino-acid substitution (exact substitution,
# not residue). icl_status='slope_lt_0' -> source reported only 'slope < 0'; continuous I-/Cl- kept
# missing (None) with the qualitative impaired label.
WASANO = {
    "Met147Val": {"hco3_slope": 0.055, "hco3_se": 0.073, "icl_slope": 0.002, "icl_se": 0.001, "icl_status": "finite", "function_label": "impaired"},
    "Arg409His": {"hco3_slope": 0.024, "hco3_se": 0.022, "icl_slope": 0.001, "icl_se": 0.002, "icl_status": "finite", "function_label": "impaired"},
    "Thr410Met": {"hco3_slope": 0.012, "hco3_se": 0.015, "icl_slope": None, "icl_se": None, "icl_status": "slope_lt_0", "function_label": "impaired"},
    "Gly497Ser": {"hco3_slope": 0.084, "hco3_se": 0.024, "icl_slope": 0.014, "icl_se": 0.002, "icl_status": "finite", "function_label": "impaired"},
    "His723Arg": {"hco3_slope": 0.115, "hco3_se": 0.053, "icl_slope": None, "icl_se": None, "icl_status": "slope_lt_0", "function_label": "impaired"},
    "WT":        {"hco3_slope": 0.861, "hco3_se": 0.055, "icl_slope": 0.074, "icl_se": 0.005, "icl_status": "finite", "function_label": "reference"},
}
WT_SLOPE = WASANO["WT"]["hco3_slope"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm_sub(hgvs_protein):
    if not isinstance(hgvs_protein, str):
        return None
    return hgvs_protein.replace("p.(", "").replace("p.", "").replace(")", "").strip() or None


def main():
    config.ensure_dirs()
    join_path = config.VARIANT_TABLE
    var_path = config.VARIANTS
    J = pd.read_csv(join_path)
    Vv = pd.read_csv(var_path)

    # fixed-c.919 missense second reported variants (same construction as variants/protein_consequence.py)
    s21 = J[J.c919_dosage == 1]
    rows = []
    for pid in s21.patient_id:
        sec = Vv[(Vv.patient_id == pid) & (Vv.hgvs_cdna != C919)].iloc[0]
        if sec.derived_consequence == "missense" and int(sec.predicted_lof) == 0:
            rows.append({"patient_id": pid, "hgvs_cdna": sec.hgvs_cdna, "hgvs_protein": sec.hgvs_protein,
                         "substitution": norm_sub(sec.hgvs_protein),
                         "vestibule_enlarged": int(s21.set_index("patient_id").loc[pid, "vestibule_enlarged"])})
    mis = pd.DataFrame(rows)
    assert len(mis) == 16, "fixed-c.919 missense subgroup must be 16 patients"
    n_distinct = mis.substitution.nunique()

    # ---- exact-match coverage (cohort substitution must equal a Wasano substitution) ----
    assayed = set(WASANO) - {"WT"}
    mis["functional_covered"] = mis.substitution.isin(assayed)
    covered_patients = int(mis.functional_covered.sum())
    covered_variants = sorted(mis[mis.functional_covered].substitution.unique().tolist())
    n_covered_variants = len(covered_variants)

    coverage_rows = []
    for sub, g in mis.groupby("substitution"):
        covered = sub in assayed
        coverage_rows.append({
            "substitution": sub, "n_carriers_fixed_c919": int(len(g)),
            "vestibular_events": int(g.vestibule_enlarged.sum()),
            "wasano_exact_match": covered,
            "hco3_cl_slope": WASANO[sub]["hco3_slope"] if covered else None,
            "hco3_cl_slope_se": WASANO[sub]["hco3_se"] if covered else None,
            "activity_fraction_of_wt": round(WASANO[sub]["hco3_slope"] / WT_SLOPE, 4) if covered else None,
            "function_label": WASANO[sub]["function_label"] if covered else "not_assayed",
        })
    coverage_df = pd.DataFrame(coverage_rows).sort_values("n_carriers_fixed_c919", ascending=False)
    coverage_df.to_csv(config.VARIANT_DIR / "pendrin_transport_coverage.csv", index=False)
    # variant-level table (covered variants only)
    coverage_df[coverage_df.wasano_exact_match].to_csv(config.VARIANT_DIR / "pendrin_transport_variants.csv",
                                                       index=False)

    # ---- extracted values and their source ----
    ext_rows = []
    for sub, rec in WASANO.items():
        ext_rows.append({
            "hgvs_protein_normalized": sub,
            "hco3_cl_slope": rec["hco3_slope"], "hco3_cl_slope_se": rec["hco3_se"],
            "icl_cl_slope": rec["icl_slope"], "icl_cl_slope_se": rec["icl_se"], "icl_status": rec["icl_status"],
            "wild_type_hco3_slope": WT_SLOPE,
            "activity_fraction_of_wt_hco3": round(rec["hco3_slope"] / WT_SLOPE, 4),
            "source_function_class": rec["function_label"],
            "notes": "used as annotation only; not used for inference. "
                     "I-/Cl- 'slope_lt_0' rows keep the continuous value missing (not truncated to zero).",
        })
    pd.DataFrame(ext_rows).to_csv(config.VARIANT_DIR / "pendrin_transport_values.csv", index=False)

    provenance = {
        "source_citation": "Wasano K, Takahashi S, Rosenberg SK, Kojima T, Mutai H, Matsunaga T, Ogawa K, "
                           "Homma K. Systematic quantification of the anion transport function of pendrin "
                           "(SLC26A4) and its disease-associated variants. Hum Mutat. 2020;41:316-331.",
        "source_doi": "10.1002/humu.23930", "source_pmid": "31599023", "source_pmcid": "PMC6930342",
        "source_table": "Table 1 (Dox-dependent antiport dose-response slope +/- SE)",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6930342/",
        "primary_assay": "HCO3-/Cl- antiport dose-response slope (bicarbonate exchange)",
        "sensitivity_assay": "I-/Cl- (His723Arg and Thr410Met reported as 'slope < 0' -> continuous value missing)",
        "access_date": "2026-07-16",
        "exact_substitution_match": "checked against the per-allele variant list",
        "imputation": "none; unassayed missense variants and splice, nonsense and frameshift variants get no transport value",
    }
    with open(config.VARIANT_DIR / "pendrin_transport_source.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    out = {
        "analysis": "pendrin_transport",
        "input_hashes": {"variant_table": sha(join_path), "variants": sha(var_path)},
        "provenance": provenance,
        "fixed_c919_missense": {"n_patients": int(len(mis)), "n_distinct_variants": int(n_distinct)},
        "exact_match_coverage": {
            "covered_patients": covered_patients, "total_patients": int(len(mis)),
            "covered_distinct_variants": covered_variants, "n_covered_distinct_variants": n_covered_variants,
            "total_distinct_variants": int(n_distinct),
        },
    }
    with open(config.VARIANT_DIR / "pendrin_transport.json", "w") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)

    print(f"pendrin transport | Wasano exact coverage "
          f"{covered_patients}/{len(mis)} patients, "
          f"{n_covered_variants} distinct variants ({', '.join(covered_variants)})")


if __name__ == "__main__":
    main()
