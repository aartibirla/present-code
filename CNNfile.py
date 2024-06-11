
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from sklearn.model_selection import train_test_split

# Assuming you have extracted features and prepared them as X_train and y_train
# X_train should be your selected features as input
# y_train should be your classified labels (0 for Normal, 1 for Abnormal)

# Example: Placeholder data (replace with actual processed features and labels)
X_train = np.random.rand(100, 32, 32, 3)  # Example: 100 samples of 32x32x3 features
y_train = np.random.randint(0, 2, size=(100,))  # Example: Binary labels (0 or 1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize selected features and weights (if needed)
selected_features = X_train.shape[1:]  # Shape of the selected features
weight = 0.5  # Placeholder for weight initialization (adjust as needed)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=selected_features),
    PReLU(),  # Parametric ReLU activation
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3)),
    PReLU(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64),
    PReLU(),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(),
              metrics=[Accuracy()])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on validation data
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')

# Example of using the model to classify a new sample
# Replace `new_sample` with an actual new sample you want to classify
new_sample = np.random.rand(1, 32, 32, 3)  # Example: Single new sample
prediction = model.predict(new_sample)
predicted_class = "Abnormal" if prediction > 0.5 else "Normal"
print(f'Predicted class: {predicted_class} (Confidence: {prediction[0][0]:.4f})')