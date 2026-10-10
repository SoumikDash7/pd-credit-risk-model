import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

FILE = "data/raw/accepted_2007_to_2018Q4.csv.gz"
SNAPSHOT = pd.Timestamp("2019-01-01")
GOOD, BAD = ["Fully Paid"], ["Charged Off", "Late (31-120 days)", "Default"]

# ---- A. composition of the label in the development sample ----
dev = pd.read_csv("data/processed/development_sample_ids_target.csv", dtype={"id": str})
dev["issue_year"] = pd.to_datetime(dev["issue_d"], format="%b-%Y").dt.year
print(f"Development sample: {len(dev):,} loans, {int(dev['target'].sum()):,} bad ({dev['target'].mean()*100:.2f}%)\n")
bad = dev[dev["target"] == 1]
comp = bad["loan_status"].value_counts().to_frame("n")
comp["pct_of_bad"] = (comp["n"] / len(bad) * 100).round(2)
comp["pct_of_all_loans"] = (comp["n"] / len(dev) * 100).round(2)
print("=== Composition of the bad label ===")
print(comp.to_string())
print("\n=== Bad-label status as % of loans issued each year ===")
print((pd.crosstab(dev["issue_year"], dev["loan_status"], normalize="index") * 100).round(2).to_string())

# ---- B. mature loans excluded for an ambiguous / censored status (script 06 logic) ----
rows = []
reader = pd.read_csv(FILE, compression="gzip", usecols=["id", "issue_d", "term", "loan_status"],
                     dtype={"id": str}, chunksize=200000, low_memory=False)
for ch in reader:
    issue = pd.to_datetime(ch["issue_d"], format="%b-%Y", errors="coerce")
    months = ch["term"].str.extract(r"(\d+)")[0].astype(float)
    mature = (issue + pd.to_timedelta(months * 30.44, unit="D")) <= SNAPSHOT
    excl = mature & ~ch["loan_status"].isin(GOOD + BAD)
    sub = ch.loc[excl, ["loan_status", "term"]].copy()
    sub["issue_year"] = issue[excl].dt.year
    sub["loan_status"] = sub["loan_status"].fillna("(blank)")
    rows.append(sub)
ex = pd.concat(rows, ignore_index=True)
print(f"\n=== Mature loans excluded for ambiguous / censored status: {len(ex):,} (script 06 reported 2,902) ===")
print(ex["loan_status"].value_counts().to_string())
ex["in_dev_window"] = ex["term"].str.contains("36") & (ex["issue_year"] >= 2013)
print("\nBy status and whether the loan falls in the development window (36 months, issued 2013+):")
print(pd.crosstab(ex["loan_status"], ex["in_dev_window"]).to_string())
print("\nBy issue year:")
print(ex.groupby("issue_year").size().to_string())

# ---- C. how much could these loans move the bad rate? ----
def bounds(label, n, b, k):
    print(f"{label}: base {b / n * 100:.2f}% on {n:,}; with {k:,} excluded loans -> "
          f"{b / (n + k) * 100:.2f}% if all good, {(b + k) / (n + k) * 100:.2f}% if all bad")

print("\n=== Sensitivity of the bad rate to the excluded loans ===")
bounds("Maturity-filtered sample", 696232, 103165, len(ex))
bounds("Development sample      ", len(dev), int(dev["target"].sum()), int(ex["in_dev_window"].sum()))
