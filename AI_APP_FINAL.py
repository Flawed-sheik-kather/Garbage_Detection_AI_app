import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
IMG_SIZE = 224

# MUST match training class order
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
# ----------------------------------------


# ---------- Choose model on startup ----------
def choose_model():
    path = filedialog.askopenfilename(
        title="Select Keras Model",
        filetypes=[("Keras model", "*.keras")]
    )
    if not path:
        messagebox.showerror("Error", "No model selected. App will close.")
        exit()
    return path


# ---------- Load model ----------
root = tk.Tk()
root.withdraw()  # hide window during model selection

model_path = choose_model()
model = tf.keras.models.load_model(model_path)

root.deiconify()  # show window after model is loaded


# ---------- Prediction (Top-3) ----------
def predict_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img), axis=0)

    preds = model.predict(img_array, verbose=0)
    probs = tf.nn.softmax(preds[0]).numpy()

    top3_idx = probs.argsort()[-3:][::-1]
    return [(class_names[i], probs[i] * 100) for i in top3_idx]


# ---------- File chooser ----------
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    top3 = predict_image(img)

    img_display = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_display)

    image_label.config(image=img_tk)
    image_label.image = img_tk

    text = "\n".join([f"{name}: {conf:.2f}%" for name, conf in top3])
    result_label.config(text=text)


# ---------- Camera ----------
def open_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not available")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        top3 = predict_image(img)

        y = 40
        for name, conf in top3:
            cv2.putText(
                frame,
                f"{name}: {conf:.1f}%",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y += 30

        cv2.imshow("Camera (Press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- GUI ----------
root.title("Image Classifier")
root.geometry("420x550")

Button(root, text="Choose Image File", command=open_file, width=30).pack(pady=10)
Button(root, text="Use Camera (press Q to quit)", command=open_camera, width=30).pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
