import os
import kagglehub
import pandas as pd

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# download dataset
path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")

# load only sale files, one per month
frames = []
for filename in sorted(os.listdir(path)):
    if filename.startswith("apartments_pl_") and filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, filename))
        df["month"] = filename[len("apartments_pl_"):-len(".csv")]
        frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
print("Cities:", sorted(all_df["city"].unique()))

# deduplicate: keep the first month each listing appeared
df = all_df.sort_values("month").drop_duplicates(subset="id", keep="first")

# cleaning
df = df.dropna(subset=["floorCount"])
floor_median = df["floor"].median()
build_year_median = df["buildYear"].median()
df["floor"] = df["floor"].fillna(floor_median)
df["buildYear"] = df["buildYear"].fillna(build_year_median)
print(f"Medians used: floor={floor_median}, buildYear={build_year_median}")

# one-hot encode city (original city column is kept)
dummies = pd.get_dummies(df["city"], prefix="city", dtype=int)
df = pd.concat([df, dummies], axis=1)

print(f"Rows after cleaning: {len(df)}")
print(df["month"].value_counts().sort_index())

df.to_csv(os.path.join(PROCESSED_DIR, "cities_filtered.csv"), index=False)