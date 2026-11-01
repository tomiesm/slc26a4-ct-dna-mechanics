"""Compare vestibular enlargement by the protein consequence and region of the second variant.

Among patients with exactly one reported c.919-2A>G allele (N=21, so the count of the recurrent
allele is held constant), the second reported variant is classified as missense or predicted loss of
function (LoF), and vestibular enlargement is compared between the two groups by a two-sided Fisher
exact test. Missense variants are further placed in the transporter (TMD) or STAS region using the
reviewed UniProt entry O43511 (STAS 535-729) and the pendrin cryo-EM domain boundaries, assigned
without reference to the outcome. All protein calls come from the per-allele variant list
(config.VARIANTS). The analysis is post hoc and exploratory.

Writes the patient-group table, the variant catalog, the group summary, the recurrent-variant
deletion sensitivity, the structural region table, a JSON summary and a pendrin map of the observed
coding variants.

Usage: python -m variants.protein_consequence
"""
import json, hashlib, re, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import contingency
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config
from cohort.build_cohort import IDENTITY_DECISIONS
from variants.anatomy_by_group import wilson, newcombe_rd, freeman_halton_2col
from grade_models.analysis import load_corrected_patient_data

C919 = "c.919-2A>G"
B_BOOT = 20000
SEED = 20260716
# Records merged into another record of the same person; none may appear in the variant subset.
COLLAPSED_ALIASES = {decision["alias_id"] for decision in IDENTITY_DECISIONS}

AA3 = {"Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile", "Leu", "Lys",
       "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val", "Ter"}

# Primary structural partition: reviewed human UniProt O43511 STAS (535-729) combined with the
# pendrin cryo-EM transporter boundary, set without reference to the outcome. Sources are listed in
# the JSON output.
REGION_PRIMARY = [("N_terminal", 1, 79), ("transporter_TMD", 80, 515),
                  ("TMD_STAS_transition", 516, 534), ("STAS", 535, 729),
                  ("boundary_730_735", 730, 735), ("post_STAS_tail", 736, 780)]
# Sensitivity partitions (must not change TMD-vs-STAS classification of the observed missense set).
REGION_BROAD_CRYOEM = [("N_terminal", 1, 79), ("transporter_TMD", 80, 515),
                       ("STAS", 516, 735), ("post_STAS_tail", 736, 780)]
REGION_UNIPROT_STAS = (535, 729)

ANNOTATION_SOURCES = {
    "reviewed_sequence": "UniProtKB O43511 (human pendrin/SLC26A4), reviewed; STAS feature 535-729",
    "structure_transporter": "pendrin cryo-EM 14-helix transporter + STAS (Liu 2023 Nat Commun 14:3012; "
                             "Wang 2024 Nat Commun 15:346)",
    "coordinate_system": "human O43511 residue numbering (1-780)",
    "access_date": "2026-07-16",
    "analysis_use": "outcome-blind structural region assignment for descriptive annotation",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_protein(hp):
    """(ref_aa, position, alt_aa) from an hgvs_protein string, else (None, None, None)."""
    if not isinstance(hp, str) or hp.startswith("p.?"):
        return (None, None, None)
    m = re.match(r"p\.\(?([A-Za-z]{3})(\d+)([A-Za-z]{3})", hp)
    if not m:
        return (None, None, None)
    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
    if ref not in AA3:
        return (None, None, None)
    return (ref, pos, alt)


def region_of(pos, partition):
    if pos is None:
        return "no_protein_coordinate"
    for name, lo, hi in partition:
        if lo <= pos <= hi:
            return name
    return "unmapped"


def tmd_stas_label(pos):
    """TMD (<=515) / STAS (535-729) / boundary_or_other, used only for the missense region split."""
    if pos is None:
        return "no_coord"
    if pos <= 515:
        return "TMD_missense"
    if REGION_UNIPROT_STAS[0] <= pos <= REGION_UNIPROT_STAS[1]:
        return "STAS_missense"
    return "boundary_or_other_missense"


def num_or_none(x):
    x = float(x)
    return None if (np.isnan(x) or np.isinf(x)) else x


def two_group(a, na, b, nb, label1, label2):
    """Group1 (a/na) vs Group2 (b/nb). Risk difference oriented group1 - group2."""
    p1, w1lo, w1hi = wilson(a, na)
    p2, w2lo, w2hi = wilson(b, nb)
    rd, rdlo, rdhi = newcombe_rd(a, na, b, nb)
    table = [[a, na - a], [b, nb - b]]
    sample_or, fisher_p = stats.fisher_exact(table)
    cond = contingency.odds_ratio(table, kind="conditional")
    ci = cond.confidence_interval(confidence_level=0.95)
    return {
        "group1": label1, "group2": label2,
        "group1_events": int(a), "group1_n": int(na), "group1_rate": p1,
        "group1_wilson_ci": [w1lo, w1hi],
        "group2_events": int(b), "group2_n": int(nb), "group2_rate": p2,
        "group2_wilson_ci": [w2lo, w2hi],
        "risk_difference_group1_minus_group2": rd, "risk_difference_newcombe_ci": [rdlo, rdhi],
        "sample_odds_ratio": {"status": "positive_infinity" if np.isinf(sample_or) else "finite",
                              "value": num_or_none(sample_or)},
        "conditional_exact_odds_ratio": {
            "status": "positive_infinity" if np.isinf(cond.statistic) else "finite",
            "value": num_or_none(cond.statistic),
            "ci_low": num_or_none(ci.low),
            "ci_high": num_or_none(ci.high),
            "ci_high_status": "positive_infinity" if np.isinf(ci.high) else "finite"},
        "fisher_exact_p_two_sided": float(fisher_p),
    }


def main():
    config.ensure_dirs()
    join_path = config.VARIANT_TABLE
    var_path = config.VARIANTS
    J = pd.read_csv(join_path)
    Vv = pd.read_csv(var_path)
    input_hashes = {"variant_table": sha(join_path), "variants": sha(var_path)}

    # ---- required input assertions ----
    assert len(J) == 35 and J.patient_id.nunique() == 35, "variant table must have 35 unique patients"
    assert len(Vv) == 70, "variants table must have 70 allele rows"
    assert (Vv.groupby("patient_id").size() == 2).all(), "every patient exactly two allele rows"
    assert set(Vv.transcript) == {"NM_000441.2"} and set(Vv.genome_build) == {"GRCh38/hg38"}
    assert bool(Vv.sequence_validation_pass.all()), "all 70 alleles must pass sequence validation"
    assert not (COLLAPSED_ALIASES & set(J.patient_id)), "no collapsed alias in the N=35 set"
    assert "<PATIENT_ID_29>" in set(J.patient_id), "<PATIENT_ID_29> must be in the variant table"
    assert J.genotype_pair.nunique() == 20, "expected 20 unique genotype pairs"
    for dosage, n in {2: 9, 1: 21, 0: 5}.items():
        assert int((J.c919_dosage == dosage).sum()) == n, f"c.919 dosage {dosage} must have {n} patients"
    assert sorted(J.vestibule_worst.unique().tolist()) == [0, 3], "vestibule_worst must be {0,3}"
    assert (J.vestibule_enlarged == (J.vestibule_worst == 3).astype(int)).all(), "endpoint == (worst==3)"
    # c.919 homozygous patients (dosage 2) vs the other patients (dosage 1 or 0), Fisher exact test
    hom = J[J.c919_dosage == 2]
    two_het = J[J.c919_dosage.isin([1, 0])]
    hom_events = int(hom.vestibule_enlarged.sum())
    two_het_events = int(two_het.vestibule_enlarged.sum())
    hom_vs_other_p = stats.fisher_exact([[hom_events, len(hom) - hom_events],
                                         [two_het_events, len(two_het) - two_het_events]])[1]

    # ---- fixed-c.919 N=21 subgroup: identify the one non-c.919 second reported variant ----
    s21 = J[J.c919_dosage == 1].copy()
    assert len(s21) == 21
    rows = []
    for pid in s21.patient_id:
        sec = Vv[(Vv.patient_id == pid) & (Vv.hgvs_cdna != C919)]
        assert len(sec) == 1, f"{pid}: expected exactly one second reported variant, got {len(sec)}"
        r = sec.iloc[0]
        ref, pos, alt = parse_protein(r.hgvs_protein)
        rows.append({
            "patient_id": pid, "second_cdna": r.hgvs_cdna, "second_protein": r.hgvs_protein,
            "second_consequence": r.derived_consequence, "second_predicted_lof": int(r.predicted_lof),
            "ref_aa": ref, "protein_position": pos, "alt_aa": alt,
            "vestibule_enlarged": int(s21.set_index("patient_id").loc[pid, "vestibule_enlarged"]),
            "genotype_pair": s21.set_index("patient_id").loc[pid, "genotype_pair"],
        })
    d = pd.DataFrame(rows)

    # consequence classification from the variant list only
    d["consequence_group"] = np.where(
        d.second_predicted_lof == 1, "predicted_LoF",
        np.where((d.second_consequence == "missense") & (d.second_predicted_lof == 0), "missense", "OTHER"))
    assert (d.consequence_group != "OTHER").all(), \
        f"unclassified N=21 record(s): {d[d.consequence_group=='OTHER'][['second_cdna','second_consequence']].values.tolist()}"
    mis = d[d.consequence_group == "missense"].copy()
    lof = d[d.consequence_group == "predicted_LoF"].copy()
    assert len(mis) == 16 and len(lof) == 5, "fixed-c.919 subgroup must have 16 missense and 5 predicted-LoF patients"

    # ---- structural region for the missense subgroup (UniProt STAS partition + broad cryo-EM sensitivity partition) ----
    broad = {name: (lo, hi) for name, lo, hi in REGION_BROAD_CRYOEM}
    broad_tmd_hi = broad["transporter_TMD"][1]
    broad_stas_lo, broad_stas_hi = broad["STAS"]
    mis["missense_region"] = mis.protein_position.map(tmd_stas_label)
    mis["missense_region_broad"] = mis.protein_position.map(
        lambda p: "TMD_missense" if (p is not None and p <= broad_tmd_hi)
        else ("STAS_missense" if (p is not None and broad_stas_lo <= p <= broad_stas_hi)
              else "boundary_or_other_missense"))
    assert (mis.missense_region == mis.missense_region_broad).all(), \
        "TMD/STAS classification must be invariant across boundary sensitivity"
    assert (mis.missense_region != "boundary_or_other_missense").all(), \
        "every missense variant must fall in the TMD or STAS range"
    tmd = mis[mis.missense_region == "TMD_missense"]
    stas = mis[mis.missense_region == "STAS_missense"]
    # region checks for named variants
    for tag, want in (("His723Arg", "STAS_missense"), ("Leu703Val", "STAS_missense"),
                      ("Gly497Ser", "TMD_missense"), ("Gly497Arg", "TMD_missense")):
        hit = mis[mis.second_protein.str.contains(tag, na=False)]
        if len(hit):
            assert (hit.missense_region == want).all(), f"{tag} must be {want}"

    # ---- statistical family ----
    A = two_group(int(lof.vestibule_enlarged.sum()), len(lof),
                  int(mis.vestibule_enlarged.sum()), len(mis),
                  "predicted_LoF_second_variant", "missense_second_variant")
    B = two_group(int(tmd.vestibule_enlarged.sum()), len(tmd),
                  int(stas.vestibule_enlarged.sum()), len(stas),
                  "TMD_missense", "STAS_missense")
    ffh_C = freeman_halton_2col([int(tmd.vestibule_enlarged.sum()), int(stas.vestibule_enlarged.sum()),
                                 int(lof.vestibule_enlarged.sum())], [len(tmd), len(stas), len(lof)])
    ffh_D = freeman_halton_2col([int(hom.vestibule_enlarged.sum()), int(mis.vestibule_enlarged.sum()),
                                 int(lof.vestibule_enlarged.sum())], [len(hom), len(mis), len(lof)])

    # Holm adjustment over the three post-hoc tests {global TMD/STAS/LoF, LoF-vs-missense, TMD-vs-STAS}.
    # Test D (c.919-2A>G homozygous vs missense vs predicted LoF) is not part of this family.
    fam_labels = ["global_tmd_stas_lof_ffh", "lof_vs_missense_fisher", "tmd_vs_stas_fisher"]
    fam_raw = [ffh_C, A["fisher_exact_p_two_sided"], B["fisher_exact_p_two_sided"]]
    fam_holm = multipletests(fam_raw, method="holm")[1].tolist()

    # ---- recurrent second-variant deletion sensitivity (keyed on variant cDNA identity) ----
    recur = d.second_cdna.value_counts()
    recur = recur[recur >= 2]
    prot_of = d.drop_duplicates("second_cdna").set_index("second_cdna").second_protein
    deletions = []
    for cdna, cnt in recur.items():
        keep = d[d.second_cdna != cdna]
        km = keep[keep.consequence_group == "missense"]; kl = keep[keep.consequence_group == "predicted_LoF"]
        if len(km) and len(kl):
            p = stats.fisher_exact([[int(kl.vestibule_enlarged.sum()), len(kl) - int(kl.vestibule_enlarged.sum())],
                                    [int(km.vestibule_enlarged.sum()), len(km) - int(km.vestibule_enlarged.sum())]])[1]
            rd = newcombe_rd(int(kl.vestibule_enlarged.sum()), len(kl),
                             int(km.vestibule_enlarged.sum()), len(km))[0]
        else:
            p, rd = None, None
        deletions.append({"deleted_second_variant_cdna": cdna,
                          "deleted_second_variant_protein": prot_of.get(cdna),
                          "n_deleted": int(cnt),
                          "missense_events": int(km.vestibule_enlarged.sum()), "missense_n": len(km),
                          "lof_events": int(kl.vestibule_enlarged.sum()), "lof_n": len(kl),
                          "risk_difference_lof_minus_missense": rd, "fisher_exact_p_two_sided": p})
    del_df = pd.DataFrame(deletions)
    del_ps = [x["fisher_exact_p_two_sided"] for x in deletions if x["fisher_exact_p_two_sided"] is not None]

    # ---- genotype-pair clustered bootstrap of the risk difference (LoF - missense) ----
    rng = np.random.default_rng(SEED)
    pairs = d.genotype_pair.to_numpy()
    uniq_pairs = np.unique(pairs)
    pair_to_idx = {gp: np.where(pairs == gp)[0] for gp in uniq_pairs}
    boot_rd, both_present = [], 0
    grp = d.consequence_group.to_numpy(); enl = d.vestibule_enlarged.to_numpy()
    for _ in range(B_BOOT):
        chosen = rng.choice(uniq_pairs, size=len(uniq_pairs), replace=True)
        idx = np.concatenate([pair_to_idx[gp] for gp in chosen])
        g, y = grp[idx], enl[idx]
        nl, nm = (g == "predicted_LoF").sum(), (g == "missense").sum()
        if nl and nm:
            both_present += 1
            boot_rd.append(y[g == "predicted_LoF"].mean() - y[g == "missense"].mean())
    boot_rd = np.asarray(boot_rd)
    clustered = {"scheme": "genotype-pair-clustered percentile", "draws": B_BOOT, "seed": SEED,
                 "both_groups_present_fraction": float(both_present / B_BOOT),
                 "risk_difference_lof_minus_missense_ci95": [float(np.percentile(boot_rd, 2.5)),
                                                             float(np.percentile(boot_rd, 97.5))]
                 if len(boot_rd) else [None, None]}

    # ---- age at first CT by consequence group ----
    cov, _ = load_corrected_patient_data()
    cov = cov.set_index("patient_id")
    d["age_at_first_ct"] = [float(cov.loc[p, "age_at_first_ct"]) if p in cov.index else np.nan for p in d.patient_id]
    age_by_group = {}
    for g in ("predicted_LoF", "missense"):
        a = d[d.consequence_group == g].age_at_first_ct.dropna()
        age_by_group[g] = {"n": int(len(a)), "median": float(a.median()),
                           "iqr": [float(a.quantile(.25)), float(a.quantile(.75))],
                           "range": [float(a.min()), float(a.max())],
                           "missing": int(d[d.consequence_group == g].age_at_first_ct.isna().sum())}

    # ---- variant catalog over all N=35 (carrier-deduplicated) ----
    catalog = []
    for cdna, grp_rows in Vv.groupby("hgvs_cdna"):
        carriers = grp_rows.patient_id.unique()
        ref, pos, alt = parse_protein(grp_rows.hgvs_protein.iloc[0])
        cons = grp_rows.derived_consequence.iloc[0]; plof = int(grp_rows.predicted_lof.iloc[0])
        enl_carriers = int(J.set_index("patient_id").loc[carriers, "vestibule_enlarged"].sum())
        catalog.append({
            "hgvs_cdna": cdna, "hgvs_protein": grp_rows.hgvs_protein.iloc[0],
            "derived_consequence": cons, "predicted_lof": plof,
            "protein_position": pos, "ref_aa": ref, "alt_aa": alt,
            "structural_region_primary": region_of(pos, REGION_PRIMARY),
            "allele_occurrence_count": int(len(grp_rows)),
            "unique_carrier_count": int(len(carriers)),
            "carriers_with_vestibular_enlargement": enl_carriers,
            "carrier_denominator": int(len(carriers)),
            "non_independence_warning": "carrier-level; patients with two non-c.919 variants contribute to two markers",
        })
    cat_df = pd.DataFrame(catalog).sort_values(["derived_consequence", "protein_position"], na_position="last")

    # ---- structural region table ----
    ann = cat_df[cat_df.protein_position.notna()][["hgvs_cdna", "hgvs_protein", "ref_aa", "alt_aa",
                                                   "protein_position", "derived_consequence"]].copy()
    ann["region_name"] = ann.protein_position.map(lambda p: region_of(p, REGION_PRIMARY))
    ann["species"] = "human"; ann["coordinate_system"] = "UniProt O43511 (1-780)"
    ann["source"] = ANNOTATION_SOURCES["reviewed_sequence"] + " | " + ANNOTATION_SOURCES["structure_transporter"]
    ann["source_version_or_access_date"] = ANNOTATION_SOURCES["access_date"]
    ann["analysis_use"] = ANNOTATION_SOURCES["analysis_use"]
    ann["pdb_reference"] = ("mouse and pig pendrin cryo-EM structures (Liu 2023 Nat Commun 14:3012; Wang 2024 "
                            "Nat Commun 15:346) used only for the broad division into transmembrane and STAS "
                            "domains; no residue-level alignment")
    ann["resolved_in_structure"] = "not_individually_curated"
    ann["notes"] = "region assigned solely from residue position and cited boundaries; no outcome used"
    ann.to_csv(config.VARIANT_DIR / "pendrin_regions.csv", index=False)

    # ---- patient-group table (one row per N=35) ----
    pg = J[["patient_id", "genotype_pair", "c919_dosage", "vestibule_worst", "vestibule_enlarged"]].copy()
    dmap = d.set_index("patient_id")
    pg["in_fixed_c919_subgroup"] = pg.c919_dosage == 1
    pg["second_reported_variant_cdna"] = pg.patient_id.map(dmap.second_cdna)
    pg["second_reported_variant_protein"] = pg.patient_id.map(dmap.second_protein)
    pg["second_consequence_group"] = pg.patient_id.map(dmap.consequence_group)
    pg["second_protein_position"] = pg.patient_id.map(dmap.protein_position)
    mismap = mis.set_index("patient_id")
    pg["missense_region"] = pg.patient_id.map(mismap.missense_region)
    pg["age_at_first_ct"] = pg.patient_id.map(dmap.age_at_first_ct)
    pg["excluded_from_consequence_test_reason"] = np.where(
        pg.c919_dosage == 1, "", np.where(pg.c919_dosage == 2, "c919_homozygous (context row only)",
                                          "two_non_c919_reported_variants (context row only)"))

    # ---- group summary table ----
    gs = []
    for name, rows_g in (("c919_homozygous_context", hom), ("fixed_c919_missense", mis),
                         ("fixed_c919_predicted_LoF", lof), ("fixed_c919_TMD_missense", tmd),
                         ("fixed_c919_STAS_missense", stas)):
        ev, n = int(rows_g.vestibule_enlarged.sum()), len(rows_g)
        pr, lo, hi = wilson(ev, n)
        gs.append({"group": name, "events": ev, "n": n, "rate": round(pr, 4),
                   "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)})
    gs_df = pd.DataFrame(gs)

    # ---- write tables ----
    pg.to_csv(config.VARIANT_DIR / "protein_patient_groups.csv", index=False)
    cat_df.to_csv(config.VARIANT_DIR / "protein_variant_catalog.csv", index=False)
    gs_df.to_csv(config.VARIANT_DIR / "protein_group_summary.csv", index=False)
    del_df.to_csv(config.VARIANT_DIR / "protein_recurrent_pair_sensitivity.csv", index=False)

    out = {
        "analysis": "protein_consequence",
        "input_hashes": input_hashes,
        "annotation_sources": ANNOTATION_SOURCES,
        "region_partitions": {"primary": REGION_PRIMARY, "broad_cryoem_sensitivity": REGION_BROAD_CRYOEM,
                              "uniprot_stas": list(REGION_UNIPROT_STAS)},
        "cohort_invariants": {"n_patients": int(len(J)), "n_allele_rows": int(len(Vv)),
                              "n_genotype_pairs": int(J.genotype_pair.nunique()),
                              "homozygous_vs_other_fisher_p": float(hom_vs_other_p)},
        "fixed_c919_subgroup": {"n": int(len(d)),
                                "second_variant_consequences": d.second_consequence.value_counts().to_dict()},
        "counts": {name: {"events": int(rows_g.vestibule_enlarged.sum()), "n": int(len(rows_g))}
                   for name, rows_g in (("missense", mis), ("predicted_LoF", lof),
                                        ("TMD_missense", tmd), ("STAS_missense", stas))},
        "test_A_missense_vs_predicted_lof": A,
        "test_B_tmd_vs_stas_missense": B,
        "test_C_three_group_ffh_tmd_stas_lof": {"p_two_sided": ffh_C},
        "test_D_homozygous_vs_missense_vs_lof_ffh": {
            "p_two_sided": ffh_D},
        "multiplicity_family": {"members": fam_labels, "raw_p": fam_raw, "holm_p": fam_holm,
                                "note": "test D is not part of the family"},
        "recurrent_pair_deletion": {"deletions": deletions,
                                    "fisher_p_range": [min(del_ps), max(del_ps)] if del_ps else [None, None]},
        "genotype_pair_clustered_bootstrap": clustered,
        "age_by_consequence_group": age_by_group,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": __import__("scipy").__version__},
        "seed": SEED, "bootstrap_replicates": B_BOOT,
    }
    with open(config.VARIANT_DIR / "protein_consequence.json", "w") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)

    _figure(cat_df, len(J))

    print(f"protein consequence | fixed-c.919 N={len(d)} | "
          f"missense {int(mis.vestibule_enlarged.sum())}/{len(mis)}, "
          f"LoF {int(lof.vestibule_enlarged.sum())}/{len(lof)} | A Fisher p={A['fisher_exact_p_two_sided']:.4f} "
          f"Holm={fam_holm[1]:.3f} | TMD {int(tmd.vestibule_enlarged.sum())}/{len(tmd)} "
          f"STAS {int(stas.vestibule_enlarged.sum())}/{len(stas)} (B p={B['fisher_exact_p_two_sided']:.3f}) | "
          f"FFH C p={ffh_C:.4f} | recurrent-del p-range [{min(del_ps):.3f},{max(del_ps):.3f}]")


def _figure(cat_df, n_patients):
    # Pendrin structural map of all observed coding variants of the patients with variant data.
    fig, axA = plt.subplots(figsize=(12.4, 5.2))
    bands = [("N-term", 1, 79, "#dfe6e9"), ("Transporter / TMD", 80, 515, "#74b9ff"),
             ("TMD-STAS", 516, 534, "#b2bec3"), ("STAS", 535, 729, "#55efc4"),
             ("", 730, 735, "#dfe6e9"), ("post-STAS", 736, 780, "#dfe6e9")]
    for name, lo, hi, col in bands:
        axA.add_patch(Rectangle((lo, 0.4), hi - lo, 0.2, facecolor=col, edgecolor="black", lw=0.6))
        if name:
            axA.text((lo + hi) / 2, 0.5, name, ha="center", va="center", fontsize=8)
    coding = cat_df[cat_df.protein_position.notna()].copy()
    shape = {"missense": "o", "nonsense": "s", "frameshift": "D"}
    cmap = matplotlib.cm.coolwarm
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    for _, r in coding.iterrows():
        frac = r.carriers_with_vestibular_enlargement / r.carrier_denominator if r.carrier_denominator else 0
        axA.scatter(r.protein_position, 0.75, s=45 + 60 * r.unique_carrier_count,
                    marker=shape.get(r.derived_consequence, "o"),
                    color=cmap(norm(frac)), edgecolor="black", lw=0.7, zorder=3)
        if r.unique_carrier_count >= 2:
            axA.text(r.protein_position, 0.95, r.hgvs_protein.replace("p.(", "").replace(")", ""),
                     rotation=90, ha="center", va="bottom", fontsize=6.6)
    axA.set_xlim(0, 785); axA.set_ylim(0.3, 1.32); axA.set_yticks([])
    axA.set_xlabel("Human pendrin residue (UniProt O43511, 1-780)")
    axA.text(0.01, 0.02, "Splice/intronic variants are not shown. Patients with two coding variants can contribute to two markers.",
             transform=axA.transAxes, fontsize=7.5, color="#636e72", va="bottom")
    # color scale = carrier vestibular-enlargement fraction
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axA, fraction=0.03, pad=0.008, aspect=10)
    cbar.set_label("Proportion of carriers with\nvestibular enlargement", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    # shape legend (consequence) upper-right, size legend (carrier count) upper-left
    sh = [axA.scatter([], [], marker=m, facecolor="lightgrey", edgecolor="black", label=lab)
          for m, lab in (("o", "missense"), ("s", "nonsense"), ("D", "frameshift"))]
    leg1 = axA.legend(handles=sh, loc="upper right", fontsize=7.5, frameon=False, ncol=3,
                      title="consequence", title_fontsize=7.5)
    axA.add_artist(leg1)
    sz = [axA.scatter([], [], s=45 + 60 * nc, marker="o", facecolor="lightgrey", edgecolor="black",
                      label=f"{nc}") for nc in (1, 2, 4)]
    axA.legend(handles=sz, loc="upper left", fontsize=7.5, frameon=False, ncol=3,
               title="marker area = unique carriers", title_fontsize=7.5,
               handletextpad=0.4, columnspacing=1.0)
    fig.suptitle("Pendrin structural map of observed SLC26A4 coding variants\n"
                 f"{n_patients} patients with detailed variant data; post-hoc, exploratory",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(config.VARIANT_DIR / "protein_consequence.pdf", bbox_inches="tight")
    fig.savefig(config.VARIANT_DIR / "protein_consequence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
