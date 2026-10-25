import pandas as pd

TARGET = "price"
NUMERIC = ["squareMeters", "rooms", "floor", "floorCount", "buildYear", "centreDistance"]
DATA_PATH = "data/processed/cities_filtered.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    city_cols = [c for c in df.columns if c.startswith("city_")]
    features = NUMERIC + city_cols
    return df[features], df[TARGET], features