# DEEP-LEARNING-PROJECT

### PERSONAL INFORMATION
* **Company:** CODTECH IT SOLUTIONS
* **Name:** Vipin Nishad
* **Intern ID:** CTIS2391
* **Domain:** Data Science
* **Batch Duration:** 6 Weeks
* **Mentor:** Neela Santosh

---

### PROJECT DESCRIPTION
The objective of this task is to implement a Deep Learning model capable of recognizing handwritten digits using the classic MNIST dataset. While many beginners stop at training a model to a certain accuracy percentage, this project goes further by building a complete end-to-end inference system.

The model is built using a Convolutional Neural Network (CNN) architecture, which is the gold standard for computer vision. The training process involves multi-epoch learning where the model learns to identify spatial patterns (loops, lines, and curves) that define digits from 0 to 9. Once the model reaches an optimal accuracy, the "brain" of the AI is saved as an `mnist_model.h5` file.

A key highlight of this project is the **Inference and Confidence Visualizer**. In professional AI development, we need to know not just *what* the model thinks, but *how sure* it is. I developed a secondary prediction script that:
1. **Normalization:** Pre-processes external images to match the 28x28 grayscale format expected by the model.
2. **Probability Mapping:** Instead of just outputting a single digit, the script extracts the raw probability distribution from the final Softmax layer.
3. **Visualization:** It generates a side-by-side comparison. On the left is the digit image, and on the right is a bar chart showing the AI's "thought process" and its mathematical certainty for each possible number.

This transparency allows us to identify "edge cases" where the AI might be confused between a '4' and a '9', showcasing a deeper understanding of model interpretability.

### TOOLS & TECHNOLOGIES
* **Framework:** TensorFlow / Keras
* **Libraries:** NumPy, Matplotlib
* **Dataset:** MNIST Handwritten Digits
* **Editor:** VS Code

---

### OUTPUT

<img width="1499" height="709" alt="Image" src="https://github.com/user-attachments/assets/d220582d-47e7-4c52-abf7-e470a3bb876d" />

<img width="1499" height="710" alt="Image" src="https://github.com/user-attachments/assets/350167ec-7e74-45c2-b725-023ce19e311a" />
