# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------

def load_clean_merged(
    data_path: str = "../data_for_student.parquet",
    meta_path: str = "../metadata_for_student.parquet",
    drop_charges: tuple[int, ...] = (1, 6),
) -> pd.DataFrame:

    df = pd.read_parquet(data_path)
    meta = pd.read_parquet(meta_path)

    df = df.drop_duplicates(["raw_file", "scan_number"], keep=False)
    meta = meta.drop_duplicates(["raw_file", "scan_number"], keep=False)

    merged = df.merge(meta, on=["raw_file", "scan_number"], how="left")
    merged = merged[~merged["precursor_charge"].isin(drop_charges)].copy()

    merged["pep_len"] = merged["peptide_sequence"].map(peptide_length)
    return merged.reset_index(drop=True)
