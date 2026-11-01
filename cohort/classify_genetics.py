"""Classify the recorded genetic results by gene and flag the recorded syndromes.

Each free-text genetic result is assigned a gene category by an ordered list of text rules; the
categories give the SLC26A4 status and the coarser gene classes. Syndrome flags are read from the
underlying-conditions text. The record table is updated in place, and the mapping from each
distinct result text to its category is saved.

Usage: python -m cohort.classify_genetics
"""

import json

import pandas as pd

import config

# ── Gene mapping rules (applied in order) ──────────────────────────────────
_GENE_RULES = [
    # (patterns_to_match, category)
    (["SLC26A4", "c.919", "Pendrin", "pendred", "DFNB4", "c.1226G>A", "c.2168A>G",
      "c.1229C>T", "c.1079C>T", "c.1489G>A", "c.1829C>A", "c.164+1G>C",
      "c.439A>G", "c.706C>G", "c.754T>C", "c.916dupG", "c.1160C>T",
      "c.1174A>T", "c.1225C>T", "c.1271G>A", "c.1343C>T", "c.1658C>A",
      "c.1692dup", "c.1707+5G>A", "c.1786C>T", "c.2086C>T", "c.2107C>G",
      "c.749delT"], "SLC26A4"),
    (["GJB2", "connexin 26", "Cx26"], "GJB2"),
    (["MYO15A"], "MYO15A"),
    (["OTOF", "otoferlin"], "OTOF"),
    (["CHD7", "CHARGE"], "CHD7"),
    (["SOX10"], "SOX10"),
    (["POU3F4"], "POU3F4"),
    (["MYO7A"], "MYO7A"),
    (["PCDH15"], "PCDH15"),
    (["PAX3"], "PAX3"),
    (["MYH9"], "MYH9"),
    (["TMC1"], "TMC1"),
    (["WFS1"], "WFS1"),
    (["TRIOBP"], "TRIOBP"),
    (["MYH14"], "MYH14"),
    (["P2RX2"], "other_gene"),
    (["EDNRB"], "other_gene"),
    (["PTPN11"], "other_gene"),
    (["MITF"], "other_gene"),
    (["USP53"], "other_gene"),
    (["m.3243", "mitochondri"], "MT_3243"),
    (["22q11", "DiGeorge", "di-George", "Di-George"], "DiGeorge_22q11"),
    (["TAF6"], "other_gene"),
    (["KCNJ16"], "other_gene"),
]


def classify_gene(raw_text: str) -> str:
    """Classify a raw genetic mutation string into a gene category."""
    if raw_text is None or pd.isna(raw_text):
        return "not_tested"

    s = str(raw_text).strip()
    if not s:
        return "not_tested"

    s_lower = s.lower()

    # Check "no test" first
    if s_lower in ("no test", "not test", "no", "none"):
        return "not_tested"

    # Gene-first rule:
    # If the string LEADS with a known gene name (e.g. "<GENE> <variant>, ...; small range NGS(-)"),
    # honor the gene classification — the NGS(-) refers to a secondary panel being negative
    # AFTER targeted gene testing identified the variant.
    s_lead = s_lower.split(",")[0].split(";")[0].strip()
    for patterns, category in _GENE_RULES:
        for pat in patterns:
            if pat.lower() in s_lead.split()[0:3]:  # gene name in first 3 tokens
                return category
            # Also handle a bare gene name as the lead token
            if s_lead.startswith(pat.lower() + " ") or s_lead == pat.lower():
                return category

    # Check negative results
    if any(neg in s_lower for neg in ["not found", "ngs(-)", "wes ngs(-)"]):
        return "negative"
    if s_lower.startswith("small range ngs"):
        return "negative"
    if s_lower.startswith("wes") and "(-)" in s_lower:
        return "negative"
    # "suspect pendred syndrome but PDS mutation(-)" → negative
    if "mutation(-)" in s_lower or "mutation (-)" in s_lower:
        return "negative"

    # Apply ordered rules
    for patterns, category in _GENE_RULES:
        for pat in patterns:
            if pat.lower() in s_lower or pat in s:
                return category

    # If we reach here, it's unclassified
    return "other_gene"


def build_gene_mapping(master_df: pd.DataFrame) -> dict:
    """Return the gene category of each distinct genetic result text, plus '__NaN__' -> 'not_tested'."""
    unique_mutations = master_df["genetic_mutation"].dropna().unique()
    mapping = {}
    for raw in sorted(unique_mutations, key=str):
        mapping[raw] = classify_gene(raw)

    # Also map None/NaN
    mapping["__NaN__"] = "not_tested"

    return mapping


def parse_underlying_syndromes(text: str) -> dict:
    """Extract syndrome flags from the underlying conditions column."""
    flags = {
        "charge_syndrome": False,
        "waardenburg": False,
        "digeorge": False,
        "goiter": False,
        "vacterl": False,
    }
    if text is None or pd.isna(text):
        return flags
    s = str(text).lower()
    if s in ("nil", "none", ""):
        return flags

    if "charge" in s:
        flags["charge_syndrome"] = True
    if "waardenburg" in s:
        flags["waardenburg"] = True
    if "digeorge" in s or "di-george" in s or "22q11" in s:
        flags["digeorge"] = True
    if "goiter" in s:
        flags["goiter"] = True
    if "vacterl" in s:
        flags["vacterl"] = True

    return flags


def main():
    config.ensure_dirs()
    print("=" * 60)
    print("Genetic results and syndromes")
    print("=" * 60)

    # ── Load the record table ──────────────────────────────────────────
    master_df = pd.read_parquet(config.RECORDS)
    print(f"\nLoaded records: {len(master_df)} rows")

    # ── Build gene mapping ─────────────────────────────────────────────
    print("\nBuilding gene category mapping...")
    gene_mapping = build_gene_mapping(master_df)

    # Apply mapping
    master_df["gene_category"] = master_df["genetic_mutation"].apply(classify_gene)

    # Print distribution
    print("\nGene category distribution (rows):")
    for cat, count in master_df["gene_category"].value_counts().items():
        n_patients = master_df[master_df["gene_category"] == cat]["patient_id"].nunique()
        print(f"  {cat}: {count} rows ({n_patients} patients)")

    # Print the distinct results classified as "other_gene"
    other_genes = master_df[master_df["gene_category"] == "other_gene"]["genetic_mutation"].unique()
    if len(other_genes) > 0:
        print(f"\nEntries classified as 'other_gene' ({len(other_genes)} unique):")
        for g in sorted(other_genes, key=str):
            print(f"  - {g}")

    # ── Verify gene constancy per patient ──────────────────────────────
    print("\nVerifying gene category is constant per patient...")
    gene_per_patient = master_df.groupby("patient_id")["gene_category"].nunique()
    inconsistent = gene_per_patient[gene_per_patient > 1]
    if len(inconsistent) > 0:
        print(f"  WARNING: {len(inconsistent)} patients have multiple gene categories:")
        for pid in inconsistent.index:
            cats = master_df[master_df["patient_id"] == pid]["gene_category"].unique()
            raw = master_df[master_df["patient_id"] == pid]["genetic_mutation"].unique()
            print(f"    {pid}: categories={cats}, raw={raw}")
    else:
        print(f"  All {len(gene_per_patient)} patients have consistent gene categories ✓")

    # ── Add gene_tested boolean ────────────────────────────────────────
    master_df["gene_tested"] = master_df["gene_category"].apply(
        lambda x: x != "not_tested"
    )
    n_tested = master_df.drop_duplicates("patient_id")["gene_tested"].sum()
    n_total_patients = master_df["patient_id"].nunique()
    print(f"\n  gene_tested: {n_tested}/{n_total_patients} patients tested "
          f"({100 * n_tested / n_total_patients:.1f}%)")

    # ── Create coarser gene labels ─────────────────────────────────────
    print("\nCreating gene_class_3 and gene_class_fine...")

    def to_class_3(cat):
        if cat == "SLC26A4":
            return "SLC26A4"
        elif cat in ("negative", "not_tested"):
            return "negative_or_untested"
        else:
            return "other_known_gene"

    master_df["gene_class_3"] = master_df["gene_category"].apply(to_class_3)

    # gene_class_fine: keep genes with ≥10 patients, merge rest
    patient_gene = master_df.drop_duplicates("patient_id")[["patient_id", "gene_category"]]
    gene_patient_counts = patient_gene["gene_category"].value_counts()
    genes_with_10_plus = set(gene_patient_counts[gene_patient_counts >= 10].index)
    # Always keep negative and not_tested as-is
    genes_with_10_plus.update(["negative", "not_tested"])

    def to_class_fine(cat):
        if cat in genes_with_10_plus:
            return cat
        else:
            return "other_known_gene"

    master_df["gene_class_fine"] = master_df["gene_category"].apply(to_class_fine)

    print("  gene_class_3 distribution:")
    for cls, count in master_df.drop_duplicates("patient_id")["gene_class_3"].value_counts().items():
        print(f"    {cls}: {count} patients")

    print("  gene_class_fine distribution:")
    for cls, count in master_df.drop_duplicates("patient_id")["gene_class_fine"].value_counts().items():
        print(f"    {cls}: {count} patients")

    # ── Parse underlying syndromes ─────────────────────────────────────
    print("\nParsing underlying syndromes...")
    syndrome_flags = master_df["underlying"].apply(parse_underlying_syndromes).apply(pd.Series)
    for col in syndrome_flags.columns:
        master_df[col] = syndrome_flags[col]
    for col in syndrome_flags.columns:
        n = master_df[col].sum()
        if n > 0:
            print(f"  {col}: {n} rows")

    # ── Save ───────────────────────────────────────────────────────────
    master_df.to_parquet(config.RECORDS, index=False)
    print(f"\nSaved updated {config.RECORDS} ({len(master_df)} rows)")

    with open(config.GENE_MAPPING, "w") as f:
        json.dump(gene_mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved {config.GENE_MAPPING} ({len(gene_mapping)} entries)")

    return master_df


if __name__ == "__main__":
    main()
