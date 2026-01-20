import os
import re
import pandas as pd
from difflib import get_close_matches
from collections import defaultdict

# ==============================
# CONFIG
# ==============================
INPUT_DIR = "input_csv"     # drop raw csv files here
OUTPUT_DIR = "state_csv"    # output root
DATE_COL = "date"
REPORT_PATH = os.path.join(OUTPUT_DIR, "process_report.txt")

# ==============================
# Regex
# ==============================
RE_SPACE = re.compile(r"\s+")
RE_BAD_FILENAME = re.compile(r"[^A-Za-z0-9 _-]+")
RE_ONLY_DIGITS = re.compile(r"\D+")
RE_MOSTLY_DIGITS = re.compile(r"^\d{3,}$")  # numeric junk like 100000


# ==============================
# CANONICAL 28 STATES
# ==============================
STATES_28 = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
]
STATES_28_SET = set(STATES_28)

# ==============================
# CANONICAL 8 UNION TERRITORIES
# ==============================
UTS_8 = [
    "Andaman And Nicobar Islands",
    "Chandigarh",
    "Dadra And Nagar Haveli And Daman And Diu",
    "Delhi",
    "Jammu And Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
]
UTS_8_SET = set(UTS_8)


def _key(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[\s_\-]+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# canonical keys
STATE_KEY_TO_NAME = {_key(x): x for x in STATES_28}
UT_KEY_TO_NAME = {_key(x): x for x in UTS_8}

# ==============================
# ALIASES / NORMALIZATION
# ==============================
REGION_ALIASES = {
    # states
    "tamilnadu": "Tamil Nadu",
    "westbengal": "West Bengal",
    "chattisgarh": "Chhattisgarh",
    "chhatishgarh": "Chhattisgarh",
    "uttaranchal": "Uttarakhand",
    "orissa": "Odisha",
    "telengana": "Telangana",

    # UT spellings
    "nctofdelhi": "Delhi",
    "delhinct": "Delhi",
    "newdelhi": "Delhi",
    "andamannicobarislands": "Andaman And Nicobar Islands",
    "dadraandnagarhaveli": "Dadra And Nagar Haveli And Daman And Diu",
    "damananddiu": "Dadra And Nagar Haveli And Daman And Diu",
    "dadraandnagarhavelianddamananddiu": "Dadra And Nagar Haveli And Daman And Diu",
    "pondicherry": "Puducherry",
}

# ==============================
# UT CITY / DISTRICT MAPPING
# ==============================
UT_DISTRICT_MAP = {
    # Delhi
    "eastdelhi": "Delhi",
    "westdelhi": "Delhi",
    "northdelhi": "Delhi",
    "southdelhi": "Delhi",
    "southwestdelhi": "Delhi",
    "south_east_delhi": "Delhi",
    "northeastdelhi": "Delhi",
    "northwestdelhi": "Delhi",
    "centraldelhi": "Delhi",
    "newdelhi": "Delhi",

    # Jammu & Kashmir UT
    "srinagar": "Jammu And Kashmir",
    "jammu": "Jammu And Kashmir",
    "anantnag": "Jammu And Kashmir",
    "baramulla": "Jammu And Kashmir",
    "kupwara": "Jammu And Kashmir",
    "pulwama": "Jammu And Kashmir",
    "budgam": "Jammu And Kashmir",
    "ganderbal": "Jammu And Kashmir",
    "kulgam": "Jammu And Kashmir",
    "shopian": "Jammu And Kashmir",
    "bandipora": "Jammu And Kashmir",
    "punch": "Jammu And Kashmir",
    "poonch": "Jammu And Kashmir",
    "rajouri": "Jammu And Kashmir",
    "udhampur": "Jammu And Kashmir",
    "reasi": "Jammu And Kashmir",
    "ramban": "Jammu And Kashmir",
    "kathua": "Jammu And Kashmir",
    "samba": "Jammu And Kashmir",
    "doda": "Jammu And Kashmir",
    "kishtwar": "Jammu And Kashmir",

    # Ladakh
    "leh": "Ladakh",
    "kargil": "Ladakh",

    # Puducherry
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "karaikal": "Puducherry",
    "mahe": "Puducherry",
    "yanam": "Puducherry",

    # Chandigarh
    "chandigarh": "Chandigarh",

    # Andaman & Nicobar Islands
    "southandaman": "Andaman And Nicobar Islands",
    "northandmiddleandaman": "Andaman And Nicobar Islands",
    "nicobars": "Andaman And Nicobar Islands",

    # Lakshadweep
    "lakshadweep": "Lakshadweep",

    # Dadra & Nagar Haveli and Daman & Diu
    "dadraandnagarhaveli": "Dadra And Nagar Haveli And Daman And Diu",
    "daman": "Dadra And Nagar Haveli And Daman And Diu",
    "diu": "Dadra And Nagar Haveli And Daman And Diu",
}


# ==============================
# Helpers
# ==============================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_folder_name(name: str) -> str:
    s = (name or "").strip().replace("&", "and")
    s = RE_BAD_FILENAME.sub("", s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s if s else "Unknown"


def clean_col(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace(" ", "_")
    c = re.sub(r"_+", "_", c)
    return c


def norm_title(x) -> str:
    if pd.isna(x):
        return ""
    s = RE_SPACE.sub(" ", str(x).strip())
    return s.title()


def norm_pincode(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = RE_ONLY_DIGITS.sub("", s)
    return s


def infer_dataset(filename: str) -> str:
    f = filename.lower()
    if "demo" in f or "demographic" in f:
        return "demographic"
    if "bio" in f or "biometric" in f:
        return "biometric"
    if "enrol" in f or "enroll" in f:
        return "enrolment"
    return "dataset"


# ==============================
# Strong date parsing
# ==============================
def parse_dates_strong(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    s = s.replace(
        {
            "": pd.NA,
            "na": pd.NA,
            "NA": pd.NA,
            "nan": pd.NA,
            "null": pd.NA,
            "None": pd.NA,
            "-": pd.NA,
            "--": pd.NA,
        }
    )

    s = s.str.replace(r"[./]", "-", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)

    d = pd.to_datetime(s, errors="coerce", dayfirst=True)

    mask = d.isna() & s.notna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], errors="coerce", dayfirst=False)
        d.loc[mask] = d2

    mask = d.isna() & s.notna()
    if mask.any():
        candidates = s[mask]
        formats = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y%m%d", "%d%m%Y", "%m%d%Y"]
        fixed = pd.Series(pd.NaT, index=candidates.index)
        for fmt in formats:
            parsed = pd.to_datetime(candidates, format=fmt, errors="coerce")
            fill = fixed.isna() & parsed.notna()
            fixed.loc[fill] = parsed.loc[fill]
        d.loc[mask] = fixed

    return d


# ==============================
# Canonicalize region
# ==============================
def canonicalize_region(state_raw: str, district_raw: str) -> tuple[str, str]:
    s = (state_raw or "").strip()
    d = (district_raw or "").strip()

    if RE_MOSTLY_DIGITS.match(s):
        s = ""

    sk = _key(s)
    dk = _key(d)

    if sk in REGION_ALIASES:
        name = REGION_ALIASES[sk]
        if name in STATES_28_SET:
            return ("STATE", name)
        if name in UTS_8_SET:
            return ("UT", name)

    if sk in STATE_KEY_TO_NAME:
        return ("STATE", STATE_KEY_TO_NAME[sk])
    if sk in UT_KEY_TO_NAME:
        return ("UT", UT_KEY_TO_NAME[sk])

    if dk in UT_DISTRICT_MAP:
        return ("UT", UT_DISTRICT_MAP[dk])

    if "delhi" in (dk or ""):
        return ("UT", "Delhi")

    if dk in REGION_ALIASES:
        name = REGION_ALIASES[dk]
        if name in STATES_28_SET:
            return ("STATE", name)
        if name in UTS_8_SET:
            return ("UT", name)

    if dk in STATE_KEY_TO_NAME:
        return ("STATE", STATE_KEY_TO_NAME[dk])
    if dk in UT_KEY_TO_NAME:
        return ("UT", UT_KEY_TO_NAME[dk])

    state_choices = list(STATE_KEY_TO_NAME.keys())
    close_state = get_close_matches(sk, state_choices, n=1, cutoff=0.92)
    if close_state:
        return ("STATE", STATE_KEY_TO_NAME[close_state[0]])

    ut_choices = list(UT_KEY_TO_NAME.keys())
    close_ut = get_close_matches(sk, ut_choices, n=1, cutoff=0.92)
    if close_ut:
        return ("UT", UT_KEY_TO_NAME[close_ut[0]])

    return ("UNKNOWN", "")


# ==============================
# Clean dataframe
# ==============================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [clean_col(c) for c in df.columns]

    required = {"state", "district", "pincode", DATE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df["state"] = df["state"].map(norm_title)
    df["district"] = df["district"].map(norm_title)
    df["pincode"] = df["pincode"].map(norm_pincode)

    region_type = []
    region_name = []
    for s, d in zip(df["state"].fillna(""), df["district"].fillna("")):
        t, name = canonicalize_region(s, d)
        region_type.append(t)
        region_name.append(name if name else pd.NA)

    df["_region_type"] = region_type
    df["_region_name"] = region_name

    df[DATE_COL] = parse_dates_strong(df[DATE_COL])

    key_cols = {DATE_COL, "state", "district", "pincode", "_region_type", "_region_name"}
    for c in df.columns:
        if c in key_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["district", "pincode", DATE_COL], kind="mergesort")


# ==============================
# Output paths
# ==============================
def get_output_folder(region_type: str, region_name: str) -> str:
    if region_type == "STATE":
        return os.path.join(OUTPUT_DIR, safe_folder_name(region_name))
    if region_type == "UT":
        return os.path.join(OUTPUT_DIR, "UT", safe_folder_name(region_name))
    return os.path.join(OUTPUT_DIR, "_UNKNOWN")


# ==============================
# Merge + Write dataset file inside region folder
# ==============================
def merge_write_dataset(region_folder: str, dataset: str, sdf: pd.DataFrame) -> None:
    ensure_dir(region_folder)
    out_path = os.path.join(region_folder, f"{dataset}.csv")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        old = pd.read_csv(out_path)
        old.columns = [clean_col(c) for c in old.columns]
        if DATE_COL in old.columns:
            old[DATE_COL] = parse_dates_strong(old[DATE_COL])
        if "district" in old.columns:
            old["district"] = old["district"].map(norm_title)
        if "pincode" in old.columns:
            old["pincode"] = old["pincode"].map(norm_pincode)
        if "state" in old.columns:
            old["state"] = old["state"].map(norm_title)

        combined = pd.concat([old, sdf], ignore_index=True)
    else:
        combined = sdf.copy()

    combined = combined.drop_duplicates()
    combined = sort_df(combined)
    combined = combined[combined[DATE_COL].notna()].copy()

    combined[DATE_COL] = combined[DATE_COL].dt.strftime("%Y-%m-%d")
    combined.to_csv(out_path, index=False)


# ==============================
# Ensure folders exist
# ==============================
def ensure_base_folders() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "UT"))

    for st in STATES_28:
        ensure_dir(os.path.join(OUTPUT_DIR, safe_folder_name(st)))

    for ut in UTS_8:
        ensure_dir(os.path.join(OUTPUT_DIR, "UT", safe_folder_name(ut)))


# ==============================
# Cleanup input folder
# ==============================
def cleanup_input_folder() -> None:
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(".csv"):
            try:
                os.remove(os.path.join(INPUT_DIR, file))
            except Exception as e:
                print(f"[WARN] Could not delete {file}: {e}")


# ==============================
# Counting utilities
# ==============================
def count_rows_in_csv(path: str) -> int:
    """
    Counts data rows (excluding header) without loading the full CSV into pandas.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # subtract header
        return max(sum(1 for _ in f) - 1, 0)


def write_report(report_lines: list[str]) -> None:
    ensure_dir(OUTPUT_DIR)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


# ==============================
# MAIN
# ==============================
def main():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    ensure_base_folders()

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    if not files:
        print(f"[WARN] No CSV files found in {INPUT_DIR}")
        return

    # ---- Audit counters (per run) ----
    global_counts = defaultdict(int)
    report = []
    report.append("UIDAI PIPELINE – PROCESS REPORT")
    report.append("=" * 60)

    # We'll track how many records THIS RUN routes into each output file
    run_out_counts = defaultdict(int)  # key: (region_type, region_name, dataset) -> count

    for file in files:
        dataset = infer_dataset(file)
        path = os.path.join(INPUT_DIR, file)

        df_raw = pd.read_csv(path, dtype=str)
        total_read = len(df_raw)

        df = clean_dataframe(df_raw)

        valid_date = df[df[DATE_COL].notna()].copy()
        invalid_date_count = total_read - len(valid_date)

        # now split mapped/unmapped on region
        mapped = valid_date[valid_date["_region_name"].notna()].copy()
        unmapped_region_count = len(valid_date) - len(mapped)

        # state rows / ut rows
        state_rows = mapped[(mapped["_region_type"] == "STATE") & (mapped["_region_name"].isin(STATES_28_SET))].copy()
        ut_rows = mapped[(mapped["_region_type"] == "UT") & (mapped["_region_name"].isin(UTS_8_SET))].copy()

        # omitted also includes mapped-but-not-in-sets (rare, but accounted)
        mapped_but_not_allowed = len(mapped) - (len(state_rows) + len(ut_rows))

        # ---- Update report (per file) ----
        report.append(f"\nINPUT FILE: {file}")
        report.append("-" * 60)
        report.append(f"Dataset inferred                    : {dataset}")
        report.append(f"Total records read                  : {total_read}")
        report.append(f"Valid date records                  : {len(valid_date)}")
        report.append(f"Written to 28 States                : {len(state_rows)}")
        report.append(f"Written to 8 UTs                    : {len(ut_rows)}")
        report.append(f"Omitted (invalid/missing date)      : {invalid_date_count}")
        report.append(f"Omitted (unmapped region)           : {unmapped_region_count}")
        report.append(f"Omitted (mapped but not allowed)    : {mapped_but_not_allowed}")

        # state breakdown for this input file
        if len(state_rows) > 0:
            report.append("\nSTATE-WISE COUNTS (this input file):")
            vc = state_rows["_region_name"].value_counts()
            for name, cnt in vc.items():
                report.append(f"  {name:<30} : {cnt}")

        # UT breakdown for this input file
        if len(ut_rows) > 0:
            report.append("\nUT-WISE COUNTS (this input file):")
            vc = ut_rows["_region_name"].value_counts()
            for name, cnt in vc.items():
                report.append(f"  {name:<30} : {cnt}")

        # ---- Write outputs + track run counts ----
        if not state_rows.empty:
            for st, sdf in state_rows.groupby("_region_name"):
                folder = get_output_folder("STATE", st)
                run_out_counts[("STATE", st, dataset)] += len(sdf)
                merge_write_dataset(
                    folder,
                    dataset,
                    sdf.drop(columns=["_region_type", "_region_name"], errors="ignore")
                )

        if not ut_rows.empty:
            for ut, sdf in ut_rows.groupby("_region_name"):
                folder = get_output_folder("UT", ut)
                run_out_counts[("UT", ut, dataset)] += len(sdf)
                merge_write_dataset(
                    folder,
                    dataset,
                    sdf.drop(columns=["_region_type", "_region_name"], errors="ignore")
                )

        # ---- Global counters ----
        global_counts["TOTAL_READ"] += total_read
        global_counts["VALID_DATE"] += len(valid_date)
        global_counts["STATE_WRITTEN"] += len(state_rows)
        global_counts["UT_WRITTEN"] += len(ut_rows)
        global_counts["OMIT_INVALID_DATE"] += invalid_date_count
        global_counts["OMIT_UNMAPPED_REGION"] += unmapped_region_count
        global_counts["OMIT_MAPPED_NOT_ALLOWED"] += mapped_but_not_allowed

        print(f"[OK] Finished dataset from file: {file}")

    # ---- Post-run: FINAL COUNTS INSIDE OUTPUT CSV FILES ----
    report.append("\n\nFINAL OUTPUT FILE ROW COUNTS (after merge)")
    report.append("=" * 60)

    # Count each dataset CSV per state folder
    # States
    for st in STATES_28:
        st_folder = os.path.join(OUTPUT_DIR, safe_folder_name(st))
        for ds in ["demographic", "biometric", "enrolment", "dataset"]:
            p = os.path.join(st_folder, f"{ds}.csv")
            if os.path.exists(p):
                report.append(f"{st:<30} | {ds:<12} | rows={count_rows_in_csv(p)}")

    # UTs
    report.append("\nUT OUTPUT FILE ROW COUNTS (after merge)")
    report.append("-" * 60)
    for ut in UTS_8:
        ut_folder = os.path.join(OUTPUT_DIR, "UT", safe_folder_name(ut))
        for ds in ["demographic", "biometric", "enrolment", "dataset"]:
            p = os.path.join(ut_folder, f"{ds}.csv")
            if os.path.exists(p):
                report.append(f"{('UT/' + ut):<30} | {ds:<12} | rows={count_rows_in_csv(p)}")

    # ---- Summary ----
    report.append("\n\nGLOBAL SUMMARY (this run)")
    report.append("=" * 60)
    report.append(f"Total records read                  : {global_counts['TOTAL_READ']}")
    report.append(f"Valid date records                  : {global_counts['VALID_DATE']}")
    report.append(f"Written to 28 States                : {global_counts['STATE_WRITTEN']}")
    report.append(f"Written to 8 UTs                    : {global_counts['UT_WRITTEN']}")
    report.append(f"Omitted (invalid/missing date)      : {global_counts['OMIT_INVALID_DATE']}")
    report.append(f"Omitted (unmapped region)           : {global_counts['OMIT_UNMAPPED_REGION']}")
    report.append(f"Omitted (mapped but not allowed)    : {global_counts['OMIT_MAPPED_NOT_ALLOWED']}")

    # sanity check line
    total_accounted = (
        global_counts["STATE_WRITTEN"] +
        global_counts["UT_WRITTEN"] +
        global_counts["OMIT_INVALID_DATE"] +
        global_counts["OMIT_UNMAPPED_REGION"] +
        global_counts["OMIT_MAPPED_NOT_ALLOWED"]
    )
    report.append(f"\nSanity check (accounted vs read)    : {total_accounted} vs {global_counts['TOTAL_READ']}")

    write_report(report)

    print(f"[DONE] Outputs updated in: {OUTPUT_DIR}")
    print(f"[REPORT] Detailed report saved at: {REPORT_PATH}")

    # Empty input folder only after full success
    cleanup_input_folder()
    print(f"[CLEANUP] Emptied input folder: {INPUT_DIR}")


if __name__ == "__main__":
    main()
