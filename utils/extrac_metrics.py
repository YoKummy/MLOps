import pandas as pd
import json

# 1. Read the YOLO results CSV
df = pd.read_csv("runs/detect/train/results.csv")

# 2. Strip the weird spaces from YOLO's column names
df.columns = df.columns.str.strip()

# 3. Get the last row (the final epoch)
final_epoch = df.iloc[-1]

# 4. Extract exactly what we care about
metrics = {
    "mAP50": round(float(final_epoch["metrics/mAP50(B)"]), 4),
    "mAP50-95": round(float(final_epoch["metrics/mAP50-95(B)"]), 4)
}

# 5. Save it as a clean JSON file
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)