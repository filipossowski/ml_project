import os
import pandas as pd
from pyproj import Transformer  
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. load
df = pd.read_csv(
    "training/rcn_raw.csv",
    dtype={"TERYT": str},
    low_memory=False,
)
df["DOK_DATA"] = pd.to_datetime(df["DOK_DATA"], errors="coerce")
print("Loaded:", df.shape)

# 2. transaction filters
df = df[df["NIER_UDZIAL"] == "1/1"]
df = df[df["TRAN_RODZAJ_TRANS"] == "wolnyRynek"]
df = df[df["LOK_FUNKCJA"] == "mieszkalna"]
print("After filters:", df.shape)

# 3. keep deeds with exactly one flat
deed_cols = ["DOK_OZNACZENIE", "DOK_TWORCA", "DOK_DATA"]
df["flats_in_deed"] = df.groupby(deed_cols, dropna=False)["DOK_OZNACZENIE"].transform("size")
df = df[df["flats_in_deed"] == 1]
df = df.drop(columns="flats_in_deed")
print("After 1 flat per deed:", df.shape)

# 4. sale year, keep 2023-2025
df["year"] = df["DOK_DATA"].dt.year
df = df[df["year"].between(2023, 2025)]
df["year"] = df["year"].astype(int)
print("After years 2023-2025:", df.shape)
print(df["year"].value_counts().sort_index())

# 5. target and area cleaning
df["LOK_CENA_BRUTTO"] = df["LOK_CENA_BRUTTO"].fillna(df["TRAN_CENA_BRUTTO"])
df["LOK_CENA_BRUTTO"] = pd.to_numeric(df["LOK_CENA_BRUTTO"], errors="coerce")
df["LOK_POW_UZYT"] = pd.to_numeric(df["LOK_POW_UZYT"], errors="coerce")
df = df.dropna(subset=["LOK_CENA_BRUTTO", "LOK_POW_UZYT"])

df = df[df["LOK_POW_UZYT"].between(15, 250)]
price_per_m2 = df["LOK_CENA_BRUTTO"] / df["LOK_POW_UZYT"]
df = df[price_per_m2.between(2000, 60000)]
print("After price/area limits:", df.shape)
print(df["LOK_CENA_BRUTTO"].describe())

# 6a. distance to city centre (km)
centres = pd.read_csv("training/city_centres.csv")
to_2180 = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
centres["cx"], centres["cy"] = to_2180.transform(centres["lon"].values, centres["lat"].values)

df = df.merge(centres[["city", "cx", "cy"]], on="city", how="left")
df["centreDistance"] = ((df["x_2180"] - df["cx"])**2 + (df["y_2180"] - df["cy"])**2) ** 0.5 / 1000
df = df.drop(columns=["cx", "cy"])
print(df.groupby("city")["centreDistance"].median().round(1))

# 6b. market type
df["market_primary"] = (df["TRAN_RODZAJ_RYNKU"] == "rynekPierwotny").astype(int)
df["market_unknown"] = df["TRAN_RODZAJ_RYNKU"].isna().astype(int)
print(df[["market_primary", "market_unknown"]].mean().round(3))

# 6c. rooms and floor
rooms = pd.to_numeric(df["LOK_LICZBA_IZB"], errors="coerce")
floor = pd.to_numeric(df["LOK_NR_KOND"], errors="coerce")
print("rooms < 1:", (rooms < 1).sum(), " rooms > 10:", (rooms > 10).sum())
print("floor < -1:", (floor < -1).sum(), " floor > 50:", (floor > 50).sum())
print(rooms.value_counts().sort_index().head(5))
print(df.groupby("city")["LOK_LICZBA_IZB"].apply(lambda s: s.isna().mean()).round(3))

df["rooms"] = pd.to_numeric(df["LOK_LICZBA_IZB"], errors="coerce")
df = df[df["rooms"].between(1, 10)]

df["floor"] = pd.to_numeric(df["LOK_NR_KOND"], errors="coerce")
floor_median = df["floor"].median()
df["floor"] = df["floor"].fillna(floor_median)
df = df[df["floor"].between(-1, 50)]
print("Floor median used:", floor_median)
print("After rooms/floor:", df.shape)
print(df[["rooms", "floor"]].describe().round(1))


# 6d. one-hot encode city (original city column is kept)
dummies = pd.get_dummies(df["city"], prefix="city", dtype=int)
df = pd.concat([df, dummies], axis=1)

# 7. save
out_path = os.path.join(PROCESSED_DIR, "rcn_cities.csv")
df.to_csv(out_path, index=False)
print("Saved:", out_path, df.shape)