"""Compare inner-ear abnormality rates by SLC26A4 status and by c.919-2A>G dosage group.

For each of the six inner-ear structures, a structure is abnormal when its worst-ear grade is above 0.
In the analysis cohort (N=229), recorded SLC26A4 patients are compared with the other tested patients:
risk difference with Newcombe interval, sample odds ratio from the 2x2 table
(scipy.stats.fisher_exact) and Fisher exact p. In the variant subset (N=35), three groups defined by
the c.919-2A>G dosage (2 = homozygous; 1 = c.919-2A>G plus a second reported heterozygous variant;
0 = two reported non-c.919 heterozygous variants) are compared by the exact Freeman-Halton test
(3x2), and homozygous patients are compared with the rest by Fisher's exact test. The gene-level
Fisher p values and the Freeman-Halton p values are also Holm-adjusted across the six structures; the
homozygous-vs-rest p values are not adjusted. The variant-subset results are exploratory (N=35,
founder-enriched, trans phase unconfirmed, rows assumed independent).
Direction: higher = more often abnormal.

Usage: python -m variants.anatomy_by_group
"""
import json
from math import lgamma, exp
from itertools import product
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

STRUCTS = ["cochlea", "vestibule", "scc", "iac", "eva", "aperture"]
STRUCT_LABEL = {"cochlea": "Cochlea", "vestibule": "Vestibule", "scc": "Semicircular canals",
                "iac": "Internal auditory canal", "eva": "Vestibular aqueduct (EVA)",
                "aperture": "Cochlear aperture"}


def worst_abnormal(df, struct):
    g = df[[f"{struct}_R", f"{struct}_L"]].max(axis=1).fillna(0)
    return (g > 0).astype(int)


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = stats.binomtest(k, n).proportion_ci(method="wilson")
    return (k / n, float(lo), float(hi))


def newcombe_rd(a, na, b, nb):
    # risk difference (group1 - group2) with Newcombe (Wilson-based) 95% CI
    p1, l1, u1 = wilson(a, na); p2, l2, u2 = wilson(b, nb)
    rd = p1 - p2
    lo = rd - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = rd + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return rd, float(lo), float(hi)


def freeman_halton_2col(rows_abn, rows_tot):
    """Exact Freeman-Halton p for an r x 2 table given observed abnormal counts per row and row
    totals. Enumerates all tables with the same margins; sums probability of tables no more
    probable than the observed one."""
    r = len(rows_tot)
    total = sum(rows_tot)
    col_abn = sum(rows_abn)
    col_norm = total - col_abn
    # log-prob of a table = const + sum over cells of -lgamma(cell+1); const cancels in ratio,
    # but we compute full multivariate hypergeometric probability for clarity.
    logconst = (sum(lgamma(t + 1) for t in rows_tot) + lgamma(col_abn + 1) + lgamma(col_norm + 1)
                - lgamma(total + 1))

    def logp(abn_cells):
        norm_cells = [rows_tot[i] - abn_cells[i] for i in range(r)]
        cells = list(abn_cells) + norm_cells
        return logconst - sum(lgamma(c + 1) for c in cells)

    obs = logp(rows_abn)
    ranges = [range(0, min(rows_tot[i], col_abn) + 1) for i in range(r)]
    total_p = 0.0
    tail_p = 0.0
    for combo in product(*ranges):
        if sum(combo) != col_abn:
            continue
        if any(combo[i] > rows_tot[i] for i in range(r)):
            continue
        lp = logp(list(combo))
        p = exp(lp)
        total_p += p
        if lp <= obs + 1e-9:
            tail_p += p
    return float(tail_p / total_p)


def sample_or_fisher(a, b, c, d):
    # 2x2 [[a,b],[c,d]]: exposed/unexposed x abnormal/normal.
    # scipy.stats.fisher_exact returns the SAMPLE (unconditional) odds ratio, not the
    # conditional MLE odds ratio; label it accordingly downstream.
    table = [[a, b], [c, d]]
    orr, p = stats.fisher_exact(table)
    return float(orr), float(p)


def holm_adjust(pmap):
    """Holm step-down adjusted p-values across a family of tests (dict in, dict out)."""
    items = sorted(pmap.items(), key=lambda kv: kv[1])
    m, prev, adj = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        a = max(min(1.0, (m - i) * p), prev)
        prev = a
        adj[k] = a
    return adj


def main():
    config.ensure_dirs()
    res = pd.read_parquet(config.PEOPLE).set_index("patient_id")
    man = pd.read_csv(config.FOLDS)
    d = man.merge(res.reset_index(), on="patient_id", how="left").set_index("patient_id")
    slc = (d.binary_label == 0).to_numpy()  # SLC26A4 positive

    gene_level = {}
    for s in STRUCTS:
        ab = worst_abnormal(d, s).to_numpy()
        a, na = int(ab[slc].sum()), int(slc.sum())
        b, nb = int(ab[~slc].sum()), int((~slc).sum())
        rd, lo, hi = newcombe_rd(a, na, b, nb)
        orr, p = sample_or_fisher(a, na - a, b, nb - b)
        gene_level[s] = {"slc_abnormal": a, "slc_n": na, "comp_abnormal": b, "comp_n": nb,
                         "slc_rate": a / na, "comp_rate": b / nb,
                         "risk_difference": rd, "rd_ci": [lo, hi],
                         "odds_ratio": orr, "fisher_p": p}

    # c.919-2A>G dosage groups: join the patients of the variant subset to their grades
    gj = pd.read_csv(config.VARIANT_TABLE).set_index("patient_id")
    gj = gj.join(res, how="left", rsuffix="_img")
    dosage = gj["c919_dosage"].astype(int)   # 2 homozygous, 1 c919+second, 0 two non-c919
    group_order = [2, 1, 0]
    group_label = {2: "c.919 homozygous", 1: "c.919 + second reported het",
                   0: "two non-c.919 reported het"}
    dosage_groups = {}
    for s in STRUCTS:
        ab = worst_abnormal(gj, s)
        rows_abn, rows_tot, per_group = [], [], {}
        for g in group_order:
            m = (dosage == g).to_numpy()
            k, n = int(ab[m].sum()), int(m.sum())
            rows_abn.append(k); rows_tot.append(n)
            pr, lo, hi = wilson(k, n)
            per_group[group_label[g]] = {"abnormal": k, "n": n, "rate": pr, "wilson_ci": [lo, hi]}
        ffh = freeman_halton_2col(rows_abn, rows_tot)
        # homozygous (dosage 2) vs rest (dosage 0/1)
        homo = (dosage == 2).to_numpy()
        a, na = int(ab[homo].sum()), int(homo.sum())
        b, nb = int(ab[~homo].sum()), int((~homo).sum())
        orr, fp = sample_or_fisher(a, na - a, b, nb - b)
        dosage_groups[s] = {"per_group": per_group, "freeman_halton_p": ffh,
                            "homozygous_vs_rest": {"homo_abnormal": a, "homo_n": na,
                                                   "rest_abnormal": b, "rest_n": nb,
                                                   "odds_ratio": orr, "fisher_p": fp}}

    # Multiplicity: Holm across the six structure tests within each comparison.
    gene_holm = holm_adjust({s: gene_level[s]["fisher_p"] for s in STRUCTS})
    dosage_holm = holm_adjust({s: dosage_groups[s]["freeman_halton_p"] for s in STRUCTS})
    for s in STRUCTS:
        gene_level[s]["fisher_p_holm6"] = gene_holm[s]
        dosage_groups[s]["freeman_halton_p_holm6"] = dosage_holm[s]
    min_nonvest = min(dosage_groups[s]["freeman_halton_p"] for s in STRUCTS if s != "vestibule")

    out = {"n_gene_level": int(len(d)), "n_dosage_groups": int(len(gj)),
           "gene_level": gene_level, "dosage_groups": dosage_groups,
           "multiplicity": {
               "method": "Holm step-down across the six structure tests within each comparison",
               "gene_level_all_significant_holm": bool(all(v < 0.05 for v in gene_holm.values())),
               "gene_level_max_holm_p": float(max(gene_holm.values())),
               "dosage_vestibule_ffh_p": float(dosage_groups["vestibule"]["freeman_halton_p"]),
               "dosage_vestibule_ffh_p_holm6": float(dosage_holm["vestibule"]),
               "dosage_min_nonvestibule_ffh_p": float(min_nonvest),
           },
           "definitions": ["abnormal = worst-ear grade > 0",
                           "the six structure tests of each comparison form one Holm family",
                           "odds ratios are sample (unconditional) odds ratios from the 2x2 table"]}
    with open(config.ANATOMY_BY_GROUP, "w") as fh:
        json.dump(out, fh, indent=2)

    # table
    rows = []
    for s in STRUCTS:
        g = gene_level[s]; a = dosage_groups[s]
        rows.append({"structure": STRUCT_LABEL[s],
                     "gene_SLC_rate": round(g["slc_rate"], 2), "gene_comp_rate": round(g["comp_rate"], 2),
                     "gene_RD": round(g["risk_difference"], 2),
                     "gene_RD_lo": round(g["rd_ci"][0], 2), "gene_RD_hi": round(g["rd_ci"][1], 2),
                     "gene_fisher_p": g["fisher_p"], "gene_fisher_p_holm6": g["fisher_p_holm6"],
                     "dosage_homo_rate": round(a["per_group"]["c.919 homozygous"]["rate"], 2),
                     "dosage_c919second_rate": round(a["per_group"]["c.919 + second reported het"]["rate"], 2),
                     "dosage_nonc919_rate": round(a["per_group"]["two non-c.919 reported het"]["rate"], 2),
                     "dosage_FFH_p": a["freeman_halton_p"], "dosage_FFH_p_holm6": a["freeman_halton_p_holm6"]})
    pd.DataFrame(rows).to_csv(config.VARIANT_DIR / "anatomy_by_group.csv", index=False)

    build_figure(out)
    print(f"gene level N={len(d)}, c.919-2A>G dosage groups N={len(gj)}")
    print("gene level, Fisher p by structure:", {s: round(gene_level[s]["fisher_p"], 4) for s in STRUCTS})
    print("c.919-2A>G dosage groups, Freeman-Halton p by structure:",
          {s: round(dosage_groups[s]["freeman_halton_p"], 4) for s in STRUCTS})


def build_figure(result):
    """Plot the abnormality rates by gene status and by c.919-2A>G dosage group."""
    gene_level = result["gene_level"]
    dosage_groups = result["dosage_groups"]
    group_label = {2: "c.919 homozygous", 1: "c.919 + second reported het",
                   0: "two non-c.919 reported het"}
    # figure: two-panel abnormal-rate comparison by gene status and by c.919-2A>G dosage group
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    labels = [STRUCT_LABEL[s] for s in STRUCTS]
    # panel A: recorded gene-status groups
    y = np.arange(len(STRUCTS))
    axA = axes[0]
    axA.barh(y - 0.2, [gene_level[s]["slc_rate"] for s in STRUCTS], height=0.4, color="#c0392b", label="SLC26A4")
    axA.barh(y + 0.2, [gene_level[s]["comp_rate"] for s in STRUCTS], height=0.4, color="#7f8c8d", label="Other tested patients")
    axA.set_yticks(y); axA.set_yticklabels(labels); axA.invert_yaxis()
    axA.set_xlabel("Abnormal (worst-ear grade > 0) rate"); axA.set_xlim(0, 1)
    axA.set_title(f"A  Recorded gene status (N={result['n_gene_level']})"); axA.legend(fontsize=8, loc="lower right")
    for i, s in enumerate(STRUCTS):
        p = gene_level[s]["fisher_p_holm6"]
        axA.text(1.01, i, ("Holm p<.001" if p < 0.001 else f"Holm p={p:.3f}"), va="center", fontsize=7.5)
    # panel B: c.919-2A>G dosage groups
    axB = axes[1]
    w = 0.26
    cols = {2: "#8e44ad", 1: "#e67e22", 0: "#16a085"}
    lbl = {2: "c.919 homozygous", 1: "c.919 + 2nd het", 0: "two non-c.919 het"}
    for j, g in enumerate([2, 1, 0]):
        rates = [dosage_groups[s]["per_group"][group_label[g]]["rate"] for s in STRUCTS]
        axB.barh(y + (j - 1) * w, rates, height=w, color=cols[g], label=lbl[g])
    axB.set_yticks(y); axB.set_yticklabels(labels); axB.invert_yaxis()
    axB.set_xlabel("Abnormal (worst-ear grade > 0) rate"); axB.set_xlim(0, 1)
    axB.set_title(f"B  c.919-2A>G dosage group (N={result['n_dosage_groups']}; exploratory)"); axB.legend(fontsize=7.5, loc="lower right")
    for i, s in enumerate(STRUCTS):
        p = dosage_groups[s]["freeman_halton_p_holm6"]
        axB.text(1.01, i, ("Holm p<.001" if p < 0.001 else f"Holm p={p:.3f}"), va="center", fontsize=7.5)
    fig.suptitle("Anatomical abnormality by gene status and by c.919-2A>G dosage group "
                 "(six-test Holm adjustment within each panel)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(config.VARIANT_DIR / "anatomy_by_group.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
