from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd

df_train = pd.read_csv("data/train.csv")
df_live = pd.read_csv("data/live.csv")

report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=df_train, current_data=df_live)
result.save_html("reports/drift_report.html")

