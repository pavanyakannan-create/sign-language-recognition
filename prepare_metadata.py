import json
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
VIDEO_DIR = RAW / "videos"

# class index -> gloss (word)
classes = {}
with open(RAW / "wlasl_class_list.txt") as f:
    for line in f:
        idx, gloss = line.strip().split("\t")
        classes[int(idx)] = gloss

# official WLASL-100 split: video_id -> {"subset": ..., "action": [label, start, end]}
with open(RAW / "nslt_100.json") as f:
    nslt = json.load(f)

rows, missing = [], 0
for video_id, info in nslt.items():
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        missing += 1
        continue
    label, start, end = info["action"]
    rows.append({
        "gloss": classes[label],
        "label": label,
        "split": info["subset"],        # train / val / test
        "video_id": video_id,
        "path": str(video_path),
        "frame_start": start,
        "frame_end": end,
    })

df = pd.DataFrame(rows)
df.to_csv("data/wlasl100.csv", index=False)

print(f"Videos found: {len(df)} | missing: {missing}")
print(df["split"].value_counts())
print("Classes with at least one video:", df["label"].nunique())
print(df.groupby("gloss").size().describe())