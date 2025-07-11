# Model-Trainer.py

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Parameters ---
DATA_DIR = 'asl_data'
MODEL_SAVE_PATH = 'asl_model.p'
LABELS_SAVE_PATH = 'asl_labels.p'
IMG_SIZE = 400
NUM_CLASSES = 26 # We have 26 letters (A-Z)

# --- Load Data ---
print("Loading data...")
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path): continue
    
    for img_path in os.listdir(class_path):
        # Load image and convert to RGB (TensorFlow prefers RGB)
        img = cv2.imread(os.path.join(class_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img_rgb)
        labels.append(int(dir_))

# --- Data Preprocessing ---
print("Preprocessing data...")
# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize pixel values to be between 0 and 1
data = data / 255.0

# Convert labels to one-hot encoding (e.g., 2 -> [0, 0, 1, 0, ...])
labels = to_categorical(labels, num_classes=NUM_CLASSES)

# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# --- Build the AI Model (CNN) ---
print("Building the model...")
model = Sequential([
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    
    # Second convolutional layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Third convolutional layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten the results to feed into a dense layer
    Flatten(),
    
    # Dense layers for classification
    Dense(256, activation='relu'),
    Dropout(0.5), # Dropout helps prevent overfitting
    Dense(NUM_CLASSES, activation='softmax') # Softmax for multi-class probability
])

# --- Compile the Model ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() # Print a summary of the model architecture

# --- Train the Model ---
print("Starting training...")
# This is where the magic happens. The model learns from the training data.
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# --- Evaluate the Model ---
print("Evaluating the model...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# --- Save the Trained Model and Labels ---
print("Saving the model...")
# We save the entire model structure and its learned weights
model.save(MODEL_SAVE_PATH)

# Also save the labels for our recognizer script to use
with open(LABELS_SAVE_PATH, 'wb') as f:
    pickle.dump({v: k for k, v in enumerate(os.listdir(DATA_DIR))}, f)

print("Training complete and model saved!")

# Optional: Plot training history
# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.show()
