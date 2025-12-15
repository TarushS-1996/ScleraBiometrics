import sys
import os
import cv2
import requests
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer   # ✅ QTimer goes here

API_BASE = "http://127.0.0.1:8000"


# ---------------------------------
# Utility: numpy → QPixmap
# ---------------------------------
def np_to_pixmap(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w, c = img.shape
    bytes_per_line = c * w
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


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
    # TAB 1: Add User (calls /segment)
    # ---------------------------------------------------------
    def add_user_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # --------------------
        # Inputs
        # --------------------
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter User ID")

        self.eye_selector = QComboBox()
        self.eye_selector.addItems(["Left", "Right"])

        # --------------------
        # Camera selector
        # --------------------
        self.camera_selector = QComboBox()
        self.available_cameras = self.list_available_cameras()
        for cam in self.available_cameras:
            self.camera_selector.addItem(f"Camera {cam}", cam)
        self.camera_selector.currentIndexChanged.connect(self.switch_camera)

        # --------------------
        # Live video label
        # --------------------
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(400, 300)

        # --------------------
        # Buttons
        # --------------------
        btn_row = QHBoxLayout()

        capture_btn = QPushButton("📸 Take Picture")
        capture_btn.clicked.connect(self.capture_frame)

        load_btn = QPushButton("🖼 Load Image")
        load_btn.clicked.connect(self.load_new_image)

        send_btn = QPushButton("🚀 Send to Backend")
        send_btn.clicked.connect(self.send_to_segment_api)

        btn_row.addWidget(capture_btn)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(send_btn)

        # --------------------
        # Status
        # --------------------
        self.add_user_status = QLabel("Ready")
        self.add_user_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --------------------
        # Layout
        # --------------------
        layout.addWidget(self.user_input)
        layout.addWidget(self.eye_selector)
        layout.addWidget(self.camera_selector)
        layout.addWidget(self.video_label)
        layout.addLayout(btn_row)
        layout.addWidget(self.add_user_status)

        tab.setLayout(layout)

        # Start first camera
        if self.available_cameras:
            self.start_camera(self.available_cameras[0])

        return tab
    
    def list_available_cameras(self, max_devices=10):
        cams = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(i)
                cap.release()
        return cams
    
    def start_camera(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)

    def switch_camera(self):
        cam_index = self.camera_selector.currentData()
        self.stop_camera()
        self.start_camera(cam_index)

    def stop_camera(self):
        if hasattr(self, "timer"):
            self.timer.stop()
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

    def update_camera_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pix = np_to_pixmap(self.current_frame)
        self.video_label.setPixmap(
            pix.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio)
        )

    def capture_frame(self):
        if hasattr(self, "current_frame"):
            self.frozen_frame = self.current_frame.copy()
            self.timer.stop()
            self.video_label.setPixmap(
                np_to_pixmap(self.frozen_frame).scaled(400,300, Qt.AspectRatioMode.KeepAspectRatio)
            )
            self.add_user_status.setText("📸 Image captured")


    def load_new_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Eye Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if fname:
            img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            self.frozen_frame = img
            self.video_label.setPixmap(
                np_to_pixmap(img).scaled(400,300, Qt.AspectRatioMode.KeepAspectRatio)
            )
            self.timer.stop()
            self.add_user_status.setText("🖼 Image loaded")


    def send_to_segment_api(self):
        user_id = self.user_input.text().strip()
        eye_side = self.eye_selector.currentText()

        if not user_id or not hasattr(self, "frozen_frame"):
            self.add_user_status.setText("❌ Missing user ID or image")
            return

        _, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(self.frozen_frame, cv2.COLOR_RGB2BGR)
        )

        files = {"image": ("capture.jpg", buf.tobytes(), "image/jpeg")}
        data = {"user_id": user_id, "eye_side": eye_side}

        r = requests.post(f"{API_BASE}/segment", files=files, data=data)

        if r.status_code == 200:
            self.add_user_status.setText("✅ User image stored")
            self.timer.start(30)  # Resume live camera
        else:
            self.add_user_status.setText("❌ Backend failed")


    # ---------------------------------------------------------
    # TAB 2: Verify User (calls /compare)
    # ---------------------------------------------------------
    def verify_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.compare_label = QLabel("No image loaded")
        self.compare_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_label = QLabel("Results will appear here")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)

        load_btn = QPushButton("Load Image to Identify")
        load_btn.clicked.connect(self.load_compare_image)

        infer_btn = QPushButton("Identify Across Database")
        infer_btn.clicked.connect(self.call_identify_api)

        layout.addWidget(self.compare_label)
        layout.addWidget(load_btn)
        layout.addWidget(infer_btn)
        layout.addWidget(self.result_label)

        tab.setLayout(layout)
        return tab

    def load_compare_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Eye Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if fname:
            self.compare_img_path = fname
            img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            self.compare_label.setPixmap(
                np_to_pixmap(img).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
            )
            self.result_label.setText("Image loaded. Ready to identify.")

    def call_identify_api(self):
        if not hasattr(self, "compare_img_path"):
            self.result_label.setText("❌ No image selected")
            return

        try:
            with open(self.compare_img_path, "rb") as f:
                files = {"image": f}
                r = requests.post(f"{API_BASE}/identify", files=files)

            if r.status_code != 200:
                self.result_label.setText("❌ Identification failed")
                return

            res = r.json()
            matches = res.get("matches", [])

            if not matches:
                self.result_label.setText("❌ No matching eyes found")
                return

            # ----------------------------
            # Display results
            # ----------------------------
            best = matches[0]
            text = (
                f"✅ BEST MATCH\n\n"
                f"Name: {best['name']}\n"
                f"Similarity: {best['similarity']:.4f}\n"
                f"Distance: {best['distance']:.4f}\n\n"
            )

            if len(matches) > 1:
                text += "Other matches:\n"
                for m in matches[1:]:
                    text += f"- {m['name']} ({m['similarity']:.4f})\n"

            self.result_label.setText(text)

        except Exception as e:
            self.result_label.setText(f"❌ Error: {str(e)}")



# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScleraApp()
    window.show()
    sys.exit(app.exec())
