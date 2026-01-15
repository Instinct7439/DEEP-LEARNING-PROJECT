import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your trained model
# Ensure you have saved your model as 'mnist_model.h5' in your previous script
model = tf.keras.models.load_model('mnist_model.h5')

# 2. Load the test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 3. Pick a random image and prepare it
idx = np.random.randint(0, len(x_test))
img = x_test[idx]
prepared_img = img.reshape(1, 28, 28, 1) / 255.0

# 4. Get Predictions (Probabilities for 0-9)
predictions = model.predict(prepared_img)[0]
predicted_label = np.argmax(predictions)

# 5. Plotting the "Thought Process"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the Digit
ax1.imshow(img, cmap='gray')
ax1.set_title(f"Actual Digit: {y_test[idx]}")
ax1.axis('off')

# Plot the Probabilities
bars = ax2.bar(range(10), predictions, color='skyblue')
bars[predicted_label].set_color('orange') # Highlight the winner
ax2.set_xticks(range(10))
ax2.set_ylim(0, 1)
ax2.set_title("AI Confidence (0-9)")
ax2.set_ylabel("Probability")

plt.tight_layout()
plt.savefig('ai_thought_process.png')
plt.show()

print(f"The AI is {predictions[predicted_label]*100:.2f}% sure this is a {predicted_label}")