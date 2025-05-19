import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import seaborn as sns

# -------------------------------
# 1. Data Preparation & Augmentation
# -------------------------------
# Set the base directory where your images are stored.
base_dir = '/content/images'

# Define directories for training and validation.
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Define image parameters.
img_width, img_height = 150, 150  # You can adjust this size based on your images.
batch_size = 32

# Create an ImageDataGenerator for training data with augmentation.
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalize pixel values.
    rotation_range=40,          # Random rotations.
    width_shift_range=0.2,      # Horizontal shifts.
    height_shift_range=0.2,     # Vertical shifts.
    shear_range=0.2,            # Shear transformations.
    zoom_range=0.2,             # Random zoom.
    horizontal_flip=True,       # Horizontal flip.
    fill_mode='nearest'         # Fill mode for new pixels.
)

# Create an ImageDataGenerator for validation data (only rescaling).
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators that read images from the directory.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'   # One-hot encoding for multiple classes.
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# -------------------------------
# 2. Visualizing Preprocessed Data
# -------------------------------
# Plot 1: Grid of Augmented Images.
sample_images, sample_labels = next(train_generator)
plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample_images[i])
    plt.title(f"Class: {np.argmax(sample_labels[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plot 2: Histogram of Pixel Intensities (for the first image).
sample_image = sample_images[0]
plt.figure(figsize=(8, 6))
plt.hist(sample_image.flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Pixel Intensities (All Channels)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Plot 3: Boxplot for Each Color Channel.
channels = ['Red', 'Green', 'Blue']
data = [sample_image[:, :, i].flatten() for i in range(3)]
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=channels)
plt.title("Boxplot of Pixel Intensities by Channel")
plt.ylabel("Pixel Intensity")
plt.show()

# Plot 4: Heatmap of Channel Correlations.
corr_matrix = np.corrcoef([sample_image[:, :, i].flatten() for i in range(3)])
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, xticklabels=channels, yticklabels=channels, cmap="coolwarm")
plt.title("Correlation Heatmap Between Color Channels")
plt.show()

# -------------------------------
# 3. Building the CNN Model
# -------------------------------
model = Sequential([
    # First convolutional block.
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    
    # Second convolutional block.
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Third convolutional block.
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flattening and fully connected layers.
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer with neurons equal to the number of classes.
])

# Compile the model.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# 4. Training the Model
# -------------------------------
epochs = 30  # You can adjust the number of epochs as needed.
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# -------------------------------
# 5. Visualizing Training Metrics
# -------------------------------
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# -------------------------------
# 6. Evaluating the Model
# -------------------------------
loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# -------------------------------
# 7. Making Predictions on Sample Images
# -------------------------------
# Get a batch of validation images.
sample_val_images, sample_val_labels = next(validation_generator)
predictions = model.predict(sample_val_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(sample_val_labels, axis=1)

# Plot the first 9 images with predicted and true labels.
plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample_val_images[i])
    plt.title(f"True: {true_classes[i]}, Pred: {predicted_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
