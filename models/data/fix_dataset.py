import pandas as pd

IN_CSV = "data/dataset.csv"
OUT_CSV = "data/dataset_fixed.csv"

# Read WITHOUT headers (because your file has none)
df = pd.read_csv(IN_CSV, header=None)

# 63 features + label + person_id = 65 columns
df.columns = [f"f{i}" for i in range(63)] + ["label", "person_id"]

df.to_csv(OUT_CSV, index=False)
print("✅ Saved fixed dataset to:", OUT_CSV)
print(df.head())