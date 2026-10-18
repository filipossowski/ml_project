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


cities_filtered = all_df.sort_values("month").drop_duplicates(subset="id", keep="first")
cities_no_warsaw = cities_filtered[cities_filtered["city"] != "warszawa"]
#how many apartments are covered
print(f"Rows after filtering and deduplication: {len(cities_filtered)}")
print(cities_filtered["month"].value_counts().sort_index())

#how many apartments are covered without Warsaw
print(f"Rows after filtering and deduplication: {len(cities_no_warsaw)}")
print(cities_no_warsaw["month"].value_counts().sort_index())

#saving data
cities_filtered.to_csv(os.path.join(PROCESSED_DIR, "cities_filtered.csv"), index=False)
cities_no_warsaw.to_csv(os.path.join(PROCESSED_DIR, "cities_no_warsaw.csv"), index=False)
