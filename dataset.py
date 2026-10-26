import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN = 48          # frames per sample
N_POINTS = 49         # 7 pose + 21 left hand + 21 right hand
L_SH, R_SH = 1, 2     # shoulder indices inside the pose block


def normalize(arr):
    """arr: (T, 147) raw landmarks -> (T, 147) body-relative, scale-free."""
    T = len(arr)
    pts = arr.reshape(T, N_POINTS, 3).copy()
    present = np.abs(pts).sum(axis=2) > 0            # (T, 49) which points exist

    ok = present[:, L_SH] & present[:, R_SH]         # frames with both shoulders
    if ok.any():
        left, right = pts[ok, L_SH], pts[ok, R_SH]
        center = np.median((left + right) / 2, axis=0)
        scale = np.median(np.linalg.norm((left - right)[:, :2], axis=1))
    else:                                            # no pose found at all
        center = pts[present].mean(axis=0) if present.any() else np.zeros(3)
        scale = 1.0
    if scale < 1e-6:
        scale = 1.0

    pts = (pts - center) / scale
    pts[~present] = 0                                # keep missing points at zero
    return pts.reshape(T, -1).astype(np.float32)


def resample(arr, n=SEQ_LEN):
    """Pick n evenly spaced frames (repeats frames if the video is short)."""
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


class SignDataset(Dataset):
    def __init__(self, split, csv_path="data/wlasl100.csv",
                 landmark_dir="data/landmarks"):
        df = pd.read_csv(csv_path, dtype={"video_id": str})
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.landmark_dir = landmark_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = np.load(f"{self.landmark_dir}/{row.video_id}.npy")
        x = resample(normalize(arr))
        return torch.from_numpy(x), int(row.label)


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = SignDataset(split)
        print(f"{split}: {len(ds)} samples")

    ds = SignDataset("train")
    x, y = ds[0]
    print("sample shape:", tuple(x.shape), "| label:", y,
          "| gloss:", ds.df.iloc[0].gloss)
    print("value range:", float(x.min()), "to", float(x.max()))
    print("classes in train:", ds.df["label"].nunique())