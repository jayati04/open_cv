import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
from collections import deque
from gui_simulation import SmartHomeGUI

# Load model and classes
MODEL_PATH = "gesture_appliance_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_LABELS = ["light_on", "light_off", "fan_on", "fan_off"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

sequence_length = 30
sequence = deque(maxlen=sequence_length)

# Initialize GUI
gui = SmartHomeGUI()

def control_appliance(gesture):
    if gesture == "light_on":
        gui.appliances["Light"] = True
        gui.update_status("Light")
    elif gesture == "light_off":
        gui.appliances["Light"] = False
        gui.update_status("Light")
    elif gesture == "fan_on":
        gui.appliances["Fan"] = True
        gui.update_status("Fan")
    elif gesture == "fan_off":
        gui.appliances["Fan"] = False
        gui.update_status("Fan")

def gesture_recognition():
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting real-time gesture recognition...")
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            keypoints = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                sequence.append(keypoints)

                if len(sequence) == sequence_length:
                    input_data = np.expand_dims(sequence, axis=0)
                    preds = model.predict(input_data, verbose=0)
                    gesture_idx = np.argmax(preds)
                    gesture_name = CLASS_LABELS[gesture_idx]
                    cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    control_appliance(gesture_name)
            else:
                cv2.putText(frame, "No Hand Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exiting...")

# Run GUI in main thread, recognition in background
threading.Thread(target=gesture_recognition, daemon=True).start()
gui.run()
