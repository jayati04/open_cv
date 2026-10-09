import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data"
actions = np.array(["light_on", "light_off", "fan_on", "fan_off"])

# Load data
sequences, labels = [], []
for i, action in enumerate(actions):
    action_path = os.path.join(DATA_DIR, action)
    for file in os.listdir(action_path):
        seq = np.load(os.path.join(action_path, file))
        if seq.shape == (30, 63):  # ensure consistent shape
            sequences.append(seq)
            labels.append(i)

X = np.array(sequences)
X = X / np.max(X)  # normalize
y = to_categorical(labels).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    Dropout(0.2),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test))

# Save model and actions
model.save("gesture_control_model.keras")
np.save("actions.npy", actions)
