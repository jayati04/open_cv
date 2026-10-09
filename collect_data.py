import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Folder to store data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Define your appliance control gestures
actions = ["light_on", "light_off", "fan_on", "fan_off"]
no_sequences = 20       # number of samples per gesture
sequence_length = 30    # frames per sample

for action in actions:
    action_dir = os.path.join(DATA_DIR, action)
    os.makedirs(action_dir, exist_ok=True)

    for seq in range(no_sequences):
        data = []
        cap = cv2.VideoCapture(0)
        print(f"\nCollecting '{action}' - sample {seq+1}/{no_sequences}")
        input("Press ENTER to start recording...")

        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            landmarks = []

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    for lm in handLms.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks = [0] * 63  # 21 landmarks × 3 coords

            data.append(landmarks)

            cv2.putText(frame, f"{action} - Frame {frame_num+1}/{sequence_length}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Collecting Gesture Data", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        np.save(os.path.join(action_dir, f"{seq}.npy"), np.array(data))
