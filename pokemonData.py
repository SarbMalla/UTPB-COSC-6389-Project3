import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_and_preprocess_images(folder_path):
    images = []
    labels = []
    all_images = []

    for label in ['grass', 'fire']:
        label_folder = os.path.join(folder_path, label)
        if not os.path.exists(label_folder):
            continue
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img = np.array(img)
            if img.shape[2] == 4:
                img = img[:, :, :3]

            mask_red = img[:, :, 0] > 100
            mask_green = img[:, :, 1] > 100
            mask_white = np.all(img > 200, axis=-1)

            img[mask_white] = [0, 0, 0]

            red_pixels = np.sum(mask_red)
            green_pixels = np.sum(mask_green)


            if red_pixels > green_pixels:
                labels.append(1)  # Fire
            else:
                labels.append(0)  # Grass

            images.append(img)
            all_images.append(img)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, all_images


#CNN model
def build_cnn_model():
    model = models.Sequential([
        layers.InputLayer(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



class LivePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, plot_frame, history_data, update_interval=100, image_label=None, all_images=None):
        self.plot_frame = plot_frame
        self.history_data = history_data
        self.update_interval = update_interval
        self.image_label = image_label
        self.all_images = all_images
        self.image_index = 0

    def on_epoch_end(self, epoch, logs=None):

        self.history_data['loss'].append(logs['loss'])
        self.history_data['val_loss'].append(logs['val_loss'])
        self.history_data['accuracy'].append(logs['accuracy'])
        self.history_data['val_accuracy'].append(logs['val_accuracy'])


        self.update_image_display()


        self.update_plot()

    def update_plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        ax1.plot(self.history_data['loss'], label='Training Loss')
        ax1.plot(self.history_data['val_loss'], label='Validation Loss')
        ax1.set_title('Loss vs Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(self.history_data['accuracy'], label='Training Accuracy')
        ax2.plot(self.history_data['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy vs Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        self.plot_frame.after(self.update_interval, lambda: self.plot_frame.update_idletasks())

    def update_image_display(self):
        if self.image_index < len(self.all_images):
            image = self.all_images[self.image_index]
            image = Image.fromarray(image)
            image = image.resize((128, 128))

            img_tk = ImageTk.PhotoImage(image)

            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            self.image_index += 1


def start_training():
    try:
        epochs = int(epochs_entry.get())
        learning_rate = float(learning_rate_entry.get())

        folder_path = filedialog.askdirectory(title="Select Pokémon Dataset Folder")
        if not folder_path:
            return

        images, labels, all_images = load_and_preprocess_images(folder_path)
        if images.shape[0] == 0:
            messagebox.showerror("Error", "No valid images found in the folder.")
            return

        images = images / 255.0

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        model = build_cnn_model()

        datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=30)
        datagen.fit(X_train)

        history_data = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }

        live_plot_callback = LivePlotCallback(plot_frame, history_data, image_label=image_label, all_images=all_images)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        live_plot_callback.update_image_display()

        model.fit(datagen.flow(X_train, y_train, batch_size=32),
                  epochs=epochs, validation_data=(X_test, y_test),
                  callbacks=[live_plot_callback])

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        result_text.insert(tk.END, f"Training complete! Test Accuracy: {test_accuracy:.2f}\n")
        result_text.see(tk.END)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")



root = tk.Tk()
root.title("Pokémon Type Classifier with CNN")

window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

position_top = int(screen_height / 2 - window_height / 2)
position_left = int(screen_width / 2 - window_width / 2)

root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')

main_frame = tk.Frame(root)
main_frame.grid(row=0, column=0, padx=10, pady=10)

tk.Label(main_frame, text="Start Training Pokémon Dataset").grid(row=0, column=0, padx=5, pady=5)

tk.Label(main_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5)
epochs_entry = tk.Entry(main_frame)
epochs_entry.grid(row=1, column=1, padx=5, pady=5)
epochs_entry.insert(tk.END, "10")
tk.Label(main_frame, text="Learning Rate:").grid(row=2, column=0, padx=5, pady=5)
learning_rate_entry = tk.Entry(main_frame)
learning_rate_entry.grid(row=2, column=1, padx=5, pady=5)
learning_rate_entry.insert(tk.END, "0.001")

tk.Button(main_frame, text="Start Training", command=start_training).grid(row=3, column=0, pady=10, columnspan=2)

plot_frame = tk.Frame(main_frame)
plot_frame.grid(row=0, column=2, rowspan=3, padx=10, pady=10)

image_label = tk.Label(main_frame)
image_label.grid(row=4, column=0, padx=10, pady=10, columnspan=2)

result_text = scrolledtext.ScrolledText(main_frame, width=40, height=6)
result_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()
