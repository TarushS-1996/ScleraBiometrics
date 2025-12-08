import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

# -----------------------------
# Load ML Models
# -----------------------------
SEG_MODEL_PATH = "Model/sclera_iris_segmentation_model.h5"
SIAMESE_WEIGHTS = "Model/siamese_model_trained.weights.h5"

seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)


def build_sclera_siamese(input_shape=(256,256,3)):
    inpA = tf.keras.layers.Input(shape=input_shape)
    inpB = tf.keras.layers.Input(shape=input_shape)

    def enc_block():
        inp = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(128, activation=None)(x)
        out = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(out)
        return tf.keras.Model(inp, out)

    encoder = enc_block()
    featA = encoder(inpA)
    featB = encoder(inpB)

    dist = tf.keras.layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]),
                                        axis=1,
                                        keepdims=True) + 1e-6)
    )([featA, featB])

    return tf.keras.Model([inpA, inpB], dist)


siamese = build_sclera_siamese()
siamese.load_weights(SIAMESE_WEIGHTS)

# ---------------------------------
# Utility: Convert numpy → QPixmap
# ---------------------------------
def np_to_pixmap(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w, c = img.shape
    bytes_per_line = c * w
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------
# Placeholder segmentation → replace with your full pipeline
# ---------------------------------------------------------
def process_eye_image(img):
    """Returns a 256×256 sclera/vein extracted image."""
    img = cv2.resize(img, (256,256))
    return img.astype(np.uint8)


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
class ScleraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sclera Vein Identity Matcher")
        self.setGeometry(100, 100, 900, 600)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.add_user_tab(), "Add User")
        self.tabs.addTab(self.verify_tab(), "Verify User")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    # ---------------------------------------------------------
    # TAB 1: Add User
    # ---------------------------------------------------------
    def add_user_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter User Name")

        self.eye_selector = QComboBox()
        self.eye_selector.addItems(["Left", "Right"])

        self.upload_label = QLabel("No Image Loaded")
        self.upload_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_new_image)

        save_btn = QPushButton("Process & Save")
        save_btn.clicked.connect(self.save_user_data)

        layout.addWidget(self.user_input)
        layout.addWidget(self.eye_selector)
        layout.addWidget(self.upload_label)
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)

        tab.setLayout(layout)
        return tab

    def load_new_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Eye Image", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            self.raw_img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            pix = np_to_pixmap(self.raw_img)
            self.upload_label.setPixmap(pix.scaled(300,300, Qt.AspectRatioMode.KeepAspectRatio))

    def save_user_data(self):
        username = self.user_input.text().strip()
        if not username:
            return

        processed = process_eye_image(self.raw_img)

        user_dir = f"data/users/{username}"
        os.makedirs(user_dir, exist_ok=True)

        save_path = f"{user_dir}/{self.eye_selector.currentText()}.png"
        cv2.imwrite(save_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

        self.upload_label.setText(f"Saved to {save_path}")

    # ---------------------------------------------------------
    # TAB 2: Verify User
    # ---------------------------------------------------------
    def verify_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # left side: stored image
        self.stored_label = QLabel("No Stored Image")
        self.stored_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # right side: new incoming image
        self.new_label = QLabel("No New Image")
        self.new_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # choose user
        self.user_selector = QComboBox()
        self.user_selector.addItems(os.listdir("data/users"))
        self.user_selector.currentIndexChanged.connect(self.load_stored_image)

        load_new_btn = QPushButton("Load New Image")
        load_new_btn.clicked.connect(self.load_new_compare_image)

        infer_btn = QPushButton("Check Similarity")
        infer_btn.clicked.connect(self.compute_similarity)

        layout.addWidget(self.user_selector)
        layout.addWidget(self.stored_label)
        layout.addWidget(self.new_label)
        layout.addWidget(load_new_btn)
        layout.addWidget(infer_btn)

        tab.setLayout(layout)
        return tab

    def load_stored_image(self):
        user = self.user_selector.currentText()
        path = f"data/users/{user}/Right.png"
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.stored_img = img
        self.stored_label.setPixmap(np_to_pixmap(img).scaled(300,300, Qt.AspectRatioMode.KeepAspectRatio))

    def load_new_compare_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Test Image", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            self.new_img = img
            self.new_label.setPixmap(np_to_pixmap(img).scaled(300,300, Qt.AspectRatioMode.KeepAspectRatio))

    def compute_similarity(self):
        seg1 = process_eye_image(self.stored_img)
        seg2 = process_eye_image(self.new_img)

        X1 = np.expand_dims(seg1, 0)
        X2 = np.expand_dims(seg2, 0)

        dist = siamese.predict([X1, X2])[0][0]
        similarity = 1.0 - dist

        self.new_label.setText(f"Similarity: {similarity:.4f}")


# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScleraApp()
    window.show()
    sys.exit(app.exec())
