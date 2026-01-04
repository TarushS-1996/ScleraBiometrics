import sys
import os
import cv2
import requests
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer

# -------------------------
# CONFIG
# -------------------------
API_BASE = "http://127.0.0.1:8000"

# -------------------------
# Utils
# -------------------------
def np_to_pixmap(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w, c = img.shape
    bytes_per_line = c * w
    qimg = QImage(
        img.data, w, h, bytes_per_line,
        QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(qimg)

def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

# =========================================================
# Main App
# =========================================================
class ScleraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sclera Vein Identity System")
        self.setGeometry(100, 100, 1200, 700)

        self.records = []

        self.tabs = QTabWidget()
        self.tabs.addTab(self.capture_tab(), "Capture / Match / Register")
        self.tabs.addTab(self.records_tab(), "Records")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    # =====================================================
    # TAB 1 — CAPTURE / MATCH / REGISTER
    # =====================================================
    def capture_tab(self):
        tab = QWidget()
        grid = QGridLayout()
        grid.setSpacing(20)
        grid.setContentsMargins(30, 30, 30, 30)

        # =========================
        # CAMERA PANEL (Top Left)
        # =========================
        cam_panel = QVBoxLayout()
        cam_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        cam_title = QLabel("Live Camera")
        cam_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam_title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.camera_selector = QComboBox()
        self.cameras = self.list_cameras()
        for c in self.cameras:
            self.camera_selector.addItem(f"Camera {c}", c)
        self.camera_selector.currentIndexChanged.connect(self.switch_camera)

        self.video_label = QLabel()
        self.video_label.setFixedSize(480, 320)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #111; border: 1px solid #444;"
        )

        cam_panel.addWidget(cam_title)
        cam_panel.addWidget(self.camera_selector)
        cam_panel.addWidget(self.video_label)

        # =========================
        # PREVIEW PANEL (Top Right)
        # =========================
        preview_panel = QVBoxLayout()
        preview_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        preview_title = QLabel("Captured / Preview")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(480, 320)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #111; border: 1px solid #444;"
        )

        preview_panel.addWidget(preview_title)
        preview_panel.addWidget(self.preview_label)

        # =========================
        # CAPTURE CONTROLS (Bottom Left)
        # =========================
        capture_controls = QVBoxLayout()
        capture_controls.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_capture = QPushButton("📸 Take Picture")
        self.btn_capture.setFixedHeight(40)
        self.btn_capture.clicked.connect(self.capture_image)

        self.btn_load = QPushButton("🖼 Load Image")
        self.btn_load.setFixedHeight(40)
        self.btn_load.clicked.connect(self.load_image)

        capture_controls.addWidget(self.btn_capture)
        capture_controls.addWidget(self.btn_load)

        # =========================
        # ACTIONS PANEL (Bottom Right)
        # =========================
        action_panel = QVBoxLayout()
        action_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("User ID")
        self.user_input.setFixedHeight(36)

        self.eye_selector = QComboBox()
        self.eye_selector.addItems(["Left", "Right"])
        self.eye_selector.setFixedHeight(36)

        self.btn_match = QPushButton("🔍 MATCH")
        self.btn_match.setFixedHeight(45)
        self.btn_match.clicked.connect(self.call_identify)

        self.btn_register = QPushButton("📝 REGISTER")
        self.btn_register.setFixedHeight(45)
        self.btn_register.clicked.connect(self.call_register)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)

        action_panel.addWidget(self.user_input)
        action_panel.addWidget(self.eye_selector)
        action_panel.addWidget(self.btn_match)
        action_panel.addWidget(self.btn_register)
        action_panel.addWidget(self.status_label)

        # =========================
        # ADD TO GRID
        # =========================
        grid.addLayout(cam_panel, 0, 0)
        grid.addLayout(preview_panel, 0, 1)
        grid.addLayout(capture_controls, 1, 0)
        grid.addLayout(action_panel, 1, 1)

        tab.setLayout(grid)

        # Start camera
        if self.cameras:
            self.start_camera(self.cameras[0])

        return tab


    # =====================================================
    # CAMERA LOGIC
    # =====================================================
    def list_cameras(self, max_devices=5):
        cams = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(i)
                cap.release()
        return cams

    def start_camera(self, index):
        self.cap = cv2.VideoCapture(index)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def switch_camera(self):
        self.timer.stop()
        self.cap.release()
        cam = self.camera_selector.currentData()
        self.start_camera(cam)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.video_label.setPixmap(
            np_to_pixmap(self.current_frame).scaled(
                480, 360, Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def capture_image(self):
        best = None
        best_score = 0
        for _ in range(5):
            ret, frame = self.cap.read()
            if not ret:
                continue
            s = sharpness(frame)
            if s > best_score:
                best = frame
                best_score = s

        if best is not None:
            self.captured = cv2.cvtColor(best, cv2.COLOR_BGR2RGB)
            self.preview_label.setPixmap(
                np_to_pixmap(self.captured).scaled(
                    480, 360, Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            self.status_label.setText("Image captured (sharpest frame)")

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.jpg *.png *.jpeg)"
        )
        if fname:
            img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            self.captured = img
            self.preview_label.setPixmap(
                np_to_pixmap(img).scaled(
                    480, 360, Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            self.status_label.setText("Image loaded")

    # =====================================================
    # API CALLS
    # =====================================================
    def call_register(self):
        if not hasattr(self, "captured"):
            self.status_label.setText("❌ No image")
            return

        user = self.user_input.text().strip()
        eye = self.eye_selector.currentText()
        if not user:
            self.status_label.setText("❌ User ID required")
            return

        _, buf = cv2.imencode(".jpg", cv2.cvtColor(self.captured, cv2.COLOR_RGB2BGR))
        files = {"image": ("capture.jpg", buf.tobytes(), "image/jpeg")}
        data = {"user_id": user, "eye_side": eye}

        r = requests.post(f"{API_BASE}/segment", files=files, data=data)

        self.log_record("REGISTER", user, 1.0 if r.status_code == 200 else 0.0)

        self.status_label.setText(
            "✅ Registered" if r.status_code == 200 else "❌ Registration failed"
        )

    def call_identify(self):
        if not hasattr(self, "captured"):
            self.status_label.setText("❌ No image")
            return

        _, buf = cv2.imencode(".jpg", cv2.cvtColor(self.captured, cv2.COLOR_RGB2BGR))
        files = {"image": ("query.jpg", buf.tobytes(), "image/jpeg")}

        r = requests.post(f"{API_BASE}/identify", files=files)

        if r.status_code != 200:
            self.status_label.setText("❌ Identification failed")
            return

        matches = r.json().get("matches", [])
        if not matches:
            self.status_label.setText("No matches found")
            return

        best = matches[0]
        self.status_label.setText(
            f"BEST MATCH: {best['name']} | Similarity: {best['similarity']:.3f}"
        )

        self.log_record("MATCH", best["name"], best["similarity"])

    # =====================================================
    # TAB 2 — RECORDS
    # =====================================================
    def records_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Action", "User / Match", "Similarity"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        layout.addWidget(self.table)
        tab.setLayout(layout)
        return tab

    def log_record(self, action, name, similarity):
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self.table.setItem(row, 1, QTableWidgetItem(action))
        self.table.setItem(row, 2, QTableWidgetItem(name))
        self.table.setItem(row, 3, QTableWidgetItem(f"{similarity:.3f}"))


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ScleraApp()
    win.show()
    sys.exit(app.exec())
