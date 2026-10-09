import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# CONFIGURATION

DATA_DIR = "data"
actions = ["light_on", "light_off", "fan_on", "fan_off"]  # change as per your gestures
sequence_length = 30  # number of frames per gesture
feature_dim = 63      # number of keypoints/features per frame

# LOAD DATA

sequences, labels = [], []
for i, action in enumerate(actions):
    action_path = os.path.join(DATA_DIR, action)
    if not os.path.exists(action_path):
        print(f"⚠️ Skipping missing folder: {action_path}")
        continue
    for file in os.listdir(action_path):
        if file.endswith(".npy"):  # assuming preprocessed .npy files
            seq_path = os.path.join(action_path, file)
            seq = np.load(seq_path)
            if seq.shape == (sequence_length, feature_dim):
                sequences.append(seq)
                labels.append(i)
            else:
                print(f"Skipping {file}: shape {seq.shape} mismatch")

X = np.array(sequences)
X = X / np.max(X)  # normalize
y = to_categorical(labels).astype(int)

print(f"✅ Loaded {len(X)} sequences, shape: {X.shape}")

# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# BUILD MODEL

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, feature_dim)),
    Dropout(0.3),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TRAIN MODEL

model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test))

# SAVE MODEL & LABELS

model.save("gesture_appliance_model.keras")
np.save("gesture_actions.npy", np.array(actions))

print("🎉 Training complete! Model saved as gesture_appliance_model.keras")
