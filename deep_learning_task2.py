import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load the dataset (Handwritten digits)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the data (Make the pixel values between 0 and 1)
# This is like the "scaling" we did in Task 1!
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Visualize the first 5 images to see what the computer sees
plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Digit: {y_train[i]}")
    plt.axis('off')
plt.show()

print("Dataset loaded and ready for training!")



# 4. Build the Neural Network Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # Flattens the 28x28 image into a single line
    layers.Dense(128, activation='relu'),    # Hidden layer with 128 "neurons"
    layers.Dropout(0.2),                     # Prevents the model from just memorizing data
    layers.Dense(10, activation='softmax')   # Output layer (one for each digit 0-9)
])

# 5. Compile the model (Prepare it for training)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train the model (This is the actual "Learning" phase)
print("\n--- Starting Training ---")
history = model.fit(x_train, y_train, epochs=5)

# 7. Evaluate how well it learned
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f'\nFinal Accuracy on hidden images: {test_acc*100:.2f}%')



# 8. Visualization of Results (Mandatory Deliverable)
plt.figure(figsize=(12, 5))

# Plot Training Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy', color='blue', marker='o')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Plot Training Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss', color='red', marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 9. Final Test: Predicting a single image
import numpy as np
prediction = model.predict(x_test[:1])
print(f"\nPrediction for the first test image: {np.argmax(prediction)}")
print(f"Actual label: {y_test[0]}")



# Save the trained model to a file
model.save('mnist_model.h5')
print("✅ Model successfully saved as mnist_model.h5")