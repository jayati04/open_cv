import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json

# PATHS — CHANGE ONLY THIS if needed

RAW_DIR = r"C:\Users\PC\OneDrive\Desktop\AI-project\leap_gesture_project\data\raw"
FLAT_DIR = r"C:\Users\PC\OneDrive\Desktop\AI-project\leap_gesture_project\data\processed"

# FLATTEN THE NESTED STRUCTURE

if not os.path.exists(FLAT_DIR):
    os.makedirs(FLAT_DIR)

print("Flattening nested folder structure...")
for main_folder in sorted(os.listdir(RAW_DIR)):
    main_path = os.path.join(RAW_DIR, main_folder)
    if not os.path.isdir(main_path):
        continue
    for sub_folder in sorted(os.listdir(main_path)):
        sub_path = os.path.join(main_path, sub_folder)
        if not os.path.isdir(sub_path):
            continue
        # Create new combined class name
        new_class_name = f"{main_folder}_{sub_folder}"
        dest_class_dir = os.path.join(FLAT_DIR, new_class_name)
        os.makedirs(dest_class_dir, exist_ok=True)

        # Copy all image files into new flat directory
        for file in os.listdir(sub_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(sub_path, file)
                dst = os.path.join(dest_class_dir, file)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

print("✅ Folder flattening completed!")

# LOAD DATA USING ImageDataGenerator

img_size = (64, 64)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    FLAT_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    FLAT_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# BUILD CNN MODEL

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TRAIN MODEL

history = model.fit(train_data, validation_data=val_data, epochs=10)

# SAVE MODEL AND CLASS LABELS

model.save('sign_model.keras')
with open("class_labels.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("✅ Model and labels saved successfully!")

# PLOT ACCURACY

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
