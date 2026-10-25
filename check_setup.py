import cv2
import mediapipe as mp
import numpy as np
import torch
import os
os.environ["GLOG_minloglevel"] = "2"
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

holistic = mp.solutions.holistic.Holistic()
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print("Webcam works:", ok)
cap.release()

if ok:
    result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print("Hands detected:", result.left_hand_landmarks is not None or result.right_hand_landmarks is not None)