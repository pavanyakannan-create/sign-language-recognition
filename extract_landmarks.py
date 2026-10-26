import os
os.environ["GLOG_minloglevel"] = "2"  # hide MediaPipe log spam

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

CSV_PATH = "data/wlasl100.csv"
OUT_DIR = Path("data/landmarks")
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists

mp_holistic = mp.solutions.holistic


def to_array(landmark_list, n_points):
    """Convert a MediaPipe landmark list to (n_points*3,) array; zeros if missing."""
    if landmark_list is None:
        return np.zeros(n_points * 3, dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in landmark_list.landmark],
                    dtype=np.float32).flatten()


def frame_features(results):
    if results.pose_landmarks is not None:
        pts = results.pose_landmarks.landmark
        pose = np.array([[pts[i].x, pts[i].y, pts[i].z] for i in POSE_IDX],
                        dtype=np.float32).flatten()
    else:
        pose = np.zeros(len(POSE_IDX) * 3, dtype=np.float32)
    left = to_array(results.left_hand_landmarks, 21)
    right = to_array(results.right_hand_landmarks, 21)
    return np.concatenate([pose, left, right])  # 147 values


def extract_video(path, frame_start, frame_end):
    cap = cv2.VideoCapture(str(path))
    start = max(int(frame_start) - 1, 0)  # json frames are 1-indexed
    frames, idx = [], 0
    # new Holistic per video so tracking state resets
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= start and (frame_end == -1 or idx < frame_end):
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(frame_features(results))
            idx += 1
            if frame_end != -1 and idx >= frame_end:
                break
    cap.release()
    return np.stack(frames) if frames else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N videos (for testing)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH, dtype={"video_id": str})
    if args.limit:
        df = df.head(args.limit)

    failed, hand_ratios, done, skipped = [], [], 0, 0
    for row in tqdm(df.itertuples(), total=len(df)):
        out_path = OUT_DIR / f"{row.video_id}.npy"
        if out_path.exists():          # resume support
            skipped += 1
            continue
        arr = extract_video(row.path, row.frame_start, row.frame_end)
        if arr is None:
            failed.append(row.video_id)
            continue
        np.save(out_path, arr)
        hands_present = (np.abs(arr[:, 21:]).sum(axis=1) > 0).mean()
        hand_ratios.append(hands_present)
        done += 1

    print(f"\nExtracted: {done} | already existed: {skipped} | failed: {len(failed)}")
    if hand_ratios:
        print(f"Average fraction of frames with a hand detected: {np.mean(hand_ratios):.2f}")
    if failed:
        print("Failed video ids:", failed[:20])


if __name__ == "__main__":
    main()