import os
import csv
import cv2
import numpy as np

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

SEQ_LEN = 10                 # number of frames per sample
FRAME_STRIDE = 2             # take every 2nd frame (reduces duplicates)
DATA_PATH = "models\data\seq_dataset.csv"
print("DATA PATH (absolute):", os.path.abspath(DATA_PATH))


def extract_features(results):
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)

    # Normalize (same as before)
    wrist = pts[0].copy()
    pts = pts - wrist
    ref = np.linalg.norm(pts[9])
    if ref < 1e-6:
        return None
    pts = pts / ref

    return pts.flatten()  # 63

def main():
    label = input("Enter label (e.g., hello / yes): ").strip()
    person_id = input("Enter person id (e.g., p1): ").strip() or "p1"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    file_exists = os.path.exists(DATA_PATH)
    print("About to open file for append...")
    with open(DATA_PATH, "a", newline="") as f:
        print("File opened OK. It should exist now.")
        writer = csv.writer(f)
        if not file_exists:
            header = [f"f{i}" for i in range(SEQ_LEN * 63)] + ["label", "person_id"]
            writer.writerow(header)

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            recording = False
            seq = []
            stride_count = 0

            print("\nControls:")
            print(" - Press 'r' to record ONE sequence sample")
            print(" - Press 'q' to quit\n")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                status = f"Label: {label} | Person: {person_id}"
                if recording:
                    status += f" | Recording: {len(seq)}/{SEQ_LEN}"
                else:
                    status += " | Press R to record"

                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                cv2.imshow("Collect Sequences", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("r") and not recording:
                    recording = True
                    seq = []
                    stride_count = 0
                    print("Recording started... hold the sign/motion steady.")

                if recording:
                    stride_count += 1
                    if stride_count % FRAME_STRIDE == 0:
                        feats = extract_features(results)
                        if feats is None:
                            # if no hand, skip this frame; keep recording until we fill SEQ_LEN
                            continue
                        seq.append(feats)

                    if len(seq) >= SEQ_LEN:
                        sample = np.concatenate(seq).tolist()  # 10*63 = 630 features
                        writer.writerow(sample + [label, person_id])
                        f.flush()
                        print(f"✅ Saved 1 sequence sample for '{label}' ({person_id})")
                        recording = False

                if key == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()