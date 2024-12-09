import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Activation functions and derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Convolutional Neural Network (CNN) implementation
class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = []
        self.biases = []
        self.weights = []
        self.init_weights()

    def init_weights(self):
        # Convolution layer filters (3x3)
        self.filters = [
            np.random.randn(3, 3) * 0.1 for _ in range(4)
        ]
        self.biases = [
            np.zeros((1,)) for _ in range(4)
        ]
        # Fully connected layer weights
        flattened_size = ((self.input_shape[0] - 2) * (self.input_shape[1] - 2) * len(self.filters))
        self.weights.append(np.random.randn(flattened_size, self.num_classes) * 0.1)

    def convolve(self, image, filter, bias):
        h, w = image.shape
        conv_output = np.zeros((h - 2, w - 2))
        for i in range(h - 2):
            for j in range(w - 2):
                region = image[i:i+3, j:j+3]
                conv_output[i, j] = np.sum(region * filter) + bias.item()
        return relu(conv_output)

    def forward(self, x):
        self.conv_outputs = []
        for filter, bias in zip(self.filters, self.biases):
            conv_output = self.convolve(x, filter, bias)
            self.conv_outputs.append(conv_output)

        flattened = np.concatenate([output.flatten() for output in self.conv_outputs], axis=0)
        self.fc_input = flattened
        logits = np.dot(flattened, self.weights[0])
        self.output = softmax(logits)
        return self.output

    def backward(self, x, y, learning_rate):
        delta = self.output - y

        # Backpropagation for fully connected layer
        grad_w_fc = np.outer(self.fc_input, delta)
        self.weights[0] -= learning_rate * grad_w_fc

    def train(self, X, y, epochs, learning_rate, text_widget, canvas):
        for epoch in range(epochs):
            loss = 0
            for i, image in enumerate(X):
                output = self.forward(image)
                loss += -np.sum(y[i] * np.log(output))
                self.backward(image, y[i], learning_rate)

            loss /= len(X)
            if epoch % 10 == 0 or epoch == epochs - 1:
                text_widget.insert(tk.END, f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}\n")
                text_widget.see(tk.END)
                text_widget.update()
                self.update_canvas(canvas)

    def update_canvas(self, canvas):
        canvas.delete("all")
        x_spacing = 150
        y_spacing = 50
        node_radius = 15

        layers = [self.input_shape[0] * self.input_shape[1]] + [len(self.filters)] + [self.num_classes]
        positions = []

        # Draw input layer
        input_layer = [(x_spacing, y_spacing * (i + 1)) for i in range(layers[0])]
        positions.append(input_layer)
        for pos in input_layer:
            canvas.create_oval(
                pos[0] - node_radius, pos[1] - node_radius,
                pos[0] + node_radius, pos[1] + node_radius,
                fill="blue"
            )

        # Draw hidden layer
        hidden_layer = [(x_spacing * 2, y_spacing * (i + 1)) for i in range(layers[1])]
        positions.append(hidden_layer)
        for pos in hidden_layer:
            canvas.create_oval(
                pos[0] - node_radius, pos[1] - node_radius,
                pos[0] + node_radius, pos[1] + node_radius,
                fill="green"
            )

        # Draw output layer
        output_layer = [(x_spacing * 3, y_spacing * (i + 1)) for i in range(layers[2])]
        positions.append(output_layer)
        for pos in output_layer:
            canvas.create_oval(
                pos[0] - node_radius, pos[1] - node_radius,
                pos[0] + node_radius, pos[1] + node_radius,
                fill="red"
            )

        # Draw connections
        for layer_idx in range(len(positions) - 1):
            for start in positions[layer_idx]:
                for end in positions[layer_idx + 1]:
                    canvas.create_line(start[0], start[1], end[0], end[1], fill="black")

# Load dataset
def load_dataset(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:  # Skip invalid images
                    continue
                image = cv2.resize(image, (28, 28))
                images.append(image / 255.0)
                labels.append(label)
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("Dataset is empty or contains no valid images.")
    return np.array(images), np.array(labels)

def preprocess_data(X, y):
    if len(X.shape) != 3:
        raise ValueError("Input images should be a 3D array of shape (num_samples, height, width).")
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    return X, y, lb.classes_

# GUI Integration
def start_training():
    try:
        epochs = int(epochs_entry.get())
        learning_rate = float(learning_rate_entry.get())

        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        X, y = load_dataset(folder_path)
        if X.size == 0 or y.size == 0:
            messagebox.showerror("Error", "No valid images found in the selected folder.")
            return

        X, y, classes = preprocess_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        global cnn
        cnn = ConvolutionalNeuralNetwork(input_shape=(28, 28), num_classes=len(classes))

        result_text.insert(tk.END, "Training started...\n")
        cnn.train(X_train, y_train, epochs, learning_rate, result_text, canvas)

        result_text.insert(tk.END, "Training complete!\n")

    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("Image Classifier with Convolutions")

frame_left = tk.Frame(root)
frame_left.pack(side=tk.LEFT, padx=10, pady=10)

frame_right = tk.Frame(root)
frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

# Input Panel
tk.Label(frame_left, text="Epochs:").grid(row=0, column=0, padx=5, pady=5)
epochs_entry = tk.Entry(frame_left)
epochs_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(frame_left, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5)
learning_rate_entry = tk.Entry(frame_left)
learning_rate_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Button(frame_left, text="Start Training", command=start_training).grid(row=2, column=0, columnspan=2, pady=10)

# Result Text
tk.Label(frame_left, text="Training Log:").grid(row=3, column=0, columnspan=2, pady=5)
result_text = scrolledtext.ScrolledText(frame_left, width=40, height=20)
result_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

# Visualization Canvas
canvas = tk.Canvas(frame_right, width=500, height=500, bg="white")
canvas.pack(padx=10, pady=10)

root.mainloop()
