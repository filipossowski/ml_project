import os
import kagglehub
import pandas as pd

#folder paths in variables
PROCESSED_DIR = "data/processed"

#import dataset
path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")


frames = []

#only sale files
for filename in sorted(os.listdir(path)):
    if filename.startswith("apartments_pl_") and filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, filename))
        df["month"] = filename[len("apartments_pl_"):-len(".csv")]
        frames.append(df)

#combining all montly tables into one
all_df = pd.concat(frames, ignore_index=True)

#what cities are available in the data
cities =[]
for city in all_df['city']:
    if city in cities:
        continue
    else:
        cities.append(city)
print(cities)

#data only from Szczecin (Poland)
szn = all_df[all_df["city"] == "szczecin"]
szn = szn.sort_values("month").drop_duplicates(subset="id", keep="first")

#how many apartments are covered
print(f"Rows after filtering and deduplication: {len(szn)}")
print(szn["month"].value_counts().sort_index())

#saving data
szn.to_csv(os.path.join(PROCESSED_DIR, "szczecin.csv"), index=False)