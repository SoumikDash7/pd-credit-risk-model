import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)

by_year = pd.read_csv("reports/sample_missingness_by_year.csv", index_col=0)
overall = pd.read_csv("reports/sample_missingness.csv", index_col=0)["pct_missing"]
mid = overall[(overall >= 1) & (overall < 50)].index
print("=== % missing by issue year (columns 1-50% missing overall) ===")
print(by_year.loc[mid].round(1).to_string())

df = pd.read_csv("data/processed/modeling_sample_ids_target.csv",
                 usecols=["issue_d", "term", "target"])
df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year
g = df.groupby(["issue_year", "term"])["target"].agg(n="size", bad_rate="mean")
g["bad_rate"] = (g["bad_rate"] * 100).round(2)
print("\n=== Sample size and bad rate (%) by issue year x term ===")
print(g.to_string())
