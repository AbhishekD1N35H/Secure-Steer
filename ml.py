import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd



# Load synthetic sensory data
synthetic_sensory_data = pd.read_csv(r"C:\Users\Admin\Desktop\Georgin\Studies\ML/synthetic_sensory_data.csv")

# Path to the folder containing your images
image_folder_path = r"C:\Users\Admin\Desktop\Georgin\Studies\mlproject\train\images"

# List all image files in the drunk and sober subdirectories
drunk_image_folder_path = os.path.join(image_folder_path, 'c0')  # c0 contains the drunk images
sober_image_folder_path = os.path.join(image_folder_path, 'c1')  # c1 contains the sober images
print("loading done")

# Function to gather images from subdirectories
def gather_images(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

# Gather image files from subdirectory 'c0' (drunk images)
drunk_image_files = gather_images(drunk_image_folder_path)

# Data augmentation function
def augment_image(image, target_size=(224, 224)):
    # Resize image to a fixed size
    image = cv2.resize(image, target_size)

    # Rotate image by a random angle between -10 and 10 degrees
    angle = np.random.uniform(-10, 10)
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # Apply slight blurring
    blurred_image = cv2.GaussianBlur(rotated_image, (3, 3), 0)

    # Adjust brightness
    brightness_factor = np.random.uniform(0.8, 1.2)
    brightened_image = cv2.convertScaleAbs(blurred_image, alpha=brightness_factor, beta=0)

    return brightened_image

# Lists to store augmented images and labels
augmented_images = []
labels = []



# Batch processing of images
batch_size = 32
num_images = len(drunk_image_files)

for start in range(0, num_images, batch_size):
    end = min(start + batch_size, num_images)

    batch_images = [augment_image(cv2.imread(image_file)) for image_file in drunk_image_files[start:end]]

    # Store the label (c0 for drunk images)
    labels.extend([0] * len(batch_images))

    # Store the augmented images
    augmented_images.extend(batch_images)

# Convert lists to NumPy arrays
augmented_images = np.array(augmented_images)
labels = np.array(labels)



# Extract features from the synthetic sensory samples (adjust as needed based on your data structure)
synthetic_sensory_features = synthetic_sensory_data[['speed', 'pitch', 'acceleration-deceleration', 'steering_tilt_ratio']].to_numpy()

# Normalize pixel values to the range [0, 1]
augmented_images = augmented_images / 255.0

from sklearn.model_selection import train_test_split

# Ensure the correct length for synthetic sensory features
num_samples = min(len(augmented_images), len(synthetic_sensory_data))

# Split the data into training and testing sets (8:2 ratio)
X_train_images, X_test_images, X_train_sensory, X_test_sensory, y_train, y_test = train_test_split(
    augmented_images[:num_samples], synthetic_sensory_data[['speed', 'pitch', 'acceleration-deceleration', 'steering_tilt_ratio']][:num_samples], labels[:num_samples],
    test_size=0.2, random_state=42, stratify=labels[:num_samples]
)



# Now you can use X_train, X_test, y_train, y_test for further processing
# For example, you can save the augmented images and labels to new folders or files
# Adjust the code based on your specific requirements
# Input layer for images

image_input = layers.Input(shape=X_train_images[0].shape)

# Convolutional layers for image processing
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Flatten layer
x = layers.Flatten()(x)

# Dense (fully connected) layers for image processing
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# Input layer for synthetic sensory data
sensory_input = layers.Input(shape=(synthetic_sensory_features.shape[1],))

# Concatenate the flattened image features with sensory input
merged = layers.Concatenate()([x, sensory_input])

# Additional dense layers for combined features
merged = layers.Dense(128, activation='relu')(merged)
merged = layers.Dropout(0.5)(merged)
merged = layers.Dense(64, activation='relu')(merged)
merged = layers.Dropout(0.5)(merged)

# Output layer
output = layers.Dense(1, activation='sigmoid')(merged)

# Create the model
model = models.Model(inputs=[image_input, sensory_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the updated model summary
model.summary()



# Train the model
# Define the number of epochs
epochs = 2

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Mini-batch Preparation
    batch_size = 32
    for batch_start in range(0, len(X_train_images), batch_size):
        batch_end = batch_start + batch_size
        X_batch_images, X_batch_sensory, y_batch = (
            X_train_images[batch_start:batch_end],
            X_train_sensory[batch_start:batch_end],
            y_train[batch_start:batch_end]
        )

        # Reshape y_batch to match the shape of predictions
        y_batch = np.reshape(y_batch, (-1, 1))
        
        # Forward Pass
        with tf.GradientTape() as tape:
            predictions = model([X_batch_images, X_batch_sensory], training=True)
             # Ensure that y_batch has the same shape as predictions
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            loss = tf.keras.losses.binary_crossentropy(y_batch, predictions)

        # Loss Calculation
        training_loss = tf.reduce_mean(loss)

        # Backpropagation
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Parameter Update
        # (Handled by the optimizer)

        # Monitoring Training Loss and Accuracy
        training_accuracy = np.mean(np.round(predictions) == y_batch)

        print(f"Batch Loss: {training_loss}, Batch Accuracy: {training_accuracy}")

    # Model Validation
    predictions_val = model([X_test_images, X_test_sensory], training=False)
    val_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(np.expand_dims(y_test, axis=1), predictions_val))
    val_accuracy = np.mean(np.round(predictions_val) == y_test)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Testing
print("Testing")

# Forward Pass for Test Data
predictions_test = model.predict([X_test_images, X_test_sensory], batch_size=32)

# Convert predictions to binary values (0 or 1)
binary_predictions_test = np.round(predictions_test)

# Evaluate the model on the testing data
test_accuracy = np.mean(binary_predictions_test == y_test)

print(f"Test Accuracy: {test_accuracy}")
