import os
import csv
import cv2
import numpy as np
import mediapipe as mp

DATA_PATH = "data/dataset.csv"
os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extract_features(results):
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    # Collect landmarks into array: (21, 3)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)

    # 1) Translate so wrist (landmark 0) is at the origin
    wrist = pts[0].copy()
    pts = pts - wrist

    # 2) Scale normalization: use distance wrist -> middle finger MCP (landmark 9) as reference
    ref = np.linalg.norm(pts[9])  # after translation, pts[9] is relative to wrist
    if ref < 1e-6:
        return None
    pts = pts / ref

    # Flatten to 63 features
    return pts.flatten()

label = input("Enter label (example: hello): ")
person_id = input("Enter person id (e.g., p1, p2, p3): ").strip()
cap = cv2.VideoCapture(0)

with mp_hands.Hands() as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

        cv2.imshow("Collect Data - Press C to capture, Q to quit", frame)

        key = cv2.waitKey(1)

        if key == ord('c'):
            print("C pressed")
            features = extract_features(results)

            if features is not None:
                file_exists = os.path.exists(DATA_PATH)

                with open(DATA_PATH, "a", newline="") as f:
                    writer = csv.writer(f)

                    if not file_exists:
                        header = [f"f{i}" for i in range(63)] + ["label", "person_id"]
                        writer.writerow(list(features) + [label, person_id])

                    writer.writerow(list(features) + [label])

                print("Captured!")
            else:
                print("No hand detected — nothing saved.")
            

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()