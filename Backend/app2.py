import sys
import os
import cv2
import requests
import numpy as np
from datetime import datetime
import base64
import json

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout,
    QSplashScreen, QProgressBar, QFrame, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QPainter, QPen, QPainterPath, QRegion
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QRectF, QPointF
import math

# -------------------------
# CONFIG
# -------------------------
API_BASE = "http://127.0.0.1:8000"

# -------------------------
# TRICORDER Color Palette - Neon Sci-Fi
# -------------------------
COLORS = {
    'bg_dark': '#0a0e1a',
    'bg_card': '#131824',
    'bg_panel': '#1a1f2e',
    'accent_cyan': '#00ffff',
    'accent_magenta': '#ff00ff',
    'accent_purple': '#9d4edd',
    'success': '#00ff88',
    'error': '#ff0055',
    'text': '#e0f4ff',
    'text_muted': '#7d8da6',
    'border_glow': '#00d4ff',
    'camera_bg': '#050810',
    'grid_line': '#1a3d5c'
}


def fetch_logs_from_stream(api_url: str) -> list[dict]:
    """
    Fetch and parse the streamed JSONL log from /logs/stream.
    Processes line-by-line so memory stays flat even for large files.
    """
    logs = []
    with requests.get(f"{api_url}/logs/stream", stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return logs

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

# -------------------------
# Custom Circular Display Frame
# -------------------------
class CircularFrame(QLabel):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.angle = 0
        self.setMinimumSize(380, 380)
        self.pixmap_data = None

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.rotate_border)
        self.anim_timer.start(50)

    def rotate_border(self):
        self.angle = (self.angle + 1) % 360
        self.update()

    def set_pixmap_data(self, pixmap):
        if pixmap is None:
            print("WARNING: CircularFrame received None pixmap")
            return
        if pixmap.isNull():
            print("WARNING: CircularFrame received null pixmap")
            return
        self.pixmap_data = pixmap
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(COLORS['camera_bg']))

        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(self.width(), self.height()) / 2 - 20

        if self.pixmap_data and not self.pixmap_data.isNull():
            painter.save()
            path = QPainterPath()
            path.addEllipse(QPointF(center_x, center_y), radius - 10, radius - 10)
            painter.setClipPath(path)

            scaled_pixmap = self.pixmap_data.scaled(
                int(radius * 2 - 20), int(radius * 2 - 20),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            img_x = center_x - scaled_pixmap.width() / 2
            img_y = center_y - scaled_pixmap.height() / 2
            painter.drawPixmap(int(img_x), int(img_y), scaled_pixmap)
            painter.restore()

        for i in range(3):
            pen = QPen(QColor(COLORS['border_glow']))
            pen.setWidth(2)
            painter.setPen(pen)
            glow_radius = radius + (i * 3)
            painter.drawEllipse(QPointF(center_x, center_y), glow_radius, glow_radius)

        pen = QPen(QColor(COLORS['accent_cyan']))
        pen.setWidth(3)
        painter.setPen(pen)

        for i in range(0, 360, 30):
            start_angle = (i + self.angle) % 360
            painter.drawArc(
                int(center_x - radius), int(center_y - radius),
                int(radius * 2), int(radius * 2),
                int(start_angle * 16), int(20 * 16)
            )

        if self.title:
            painter.setPen(QColor(COLORS['accent_cyan']))
            font = QFont('Orbitron', 11, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, self.title)

# -------------------------
# Animated Processing Visualization
# -------------------------
class ProcessingViz(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 350)
        self.nodes = []
        self.connections = []
        self.pulse = 0
        self.is_active = False

        self.generate_network()

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)

    def generate_network(self):
        center_x = 175
        center_y = 175

        for i in range(30):
            angle = (i / 30) * 2 * math.pi
            radius = 80 + (i % 3) * 30
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            self.nodes.append({'x': x, 'y': y, 'phase': i * 0.2})

        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes[i+1:], i+1):
                dist = math.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
                if dist < 100 and len(self.connections) < 50:
                    self.connections.append((i, j))

    def start_animation(self):
        self.is_active = True
        self.timer.start(50)

    def stop_animation(self):
        self.is_active = False
        self.timer.stop()

    def animate(self):
        self.pulse += 0.1
        self.update()

    def paintEvent(self, event):
        if not self.is_active:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS['camera_bg']))

        for conn in self.connections:
            node1 = self.nodes[conn[0]]
            node2 = self.nodes[conn[1]]
            alpha = int(128 + 127 * math.sin(self.pulse + conn[0] * 0.1))
            pen = QPen(QColor(COLORS['accent_magenta']))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(int(node1['x']), int(node1['y']), int(node2['x']), int(node2['y']))

        for node in self.nodes:
            size = 3 + 2 * math.sin(self.pulse + node['phase'])
            alpha = int(200 + 55 * math.sin(self.pulse + node['phase']))

            for r in range(3, 0, -1):
                color = QColor(COLORS['accent_cyan'])
                color.setAlpha(alpha // (4 - r))
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(node['x'], node['y']), size + r * 2, size + r * 2)

            painter.setBrush(QColor(COLORS['accent_cyan']))
            painter.drawEllipse(QPointF(node['x'], node['y']), size, size)

# -------------------------
# Custom Splash Screen
# -------------------------
class ModernSplash(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.setFixedSize(600, 400)
        self.angle = 0

        self.setStyleSheet(f"""
            QSplashScreen {{
                background-color: {COLORS['bg_dark']};
                border: 3px solid {COLORS['accent_cyan']};
                border-radius: 20px;
            }}
        """)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)

    def animate(self):
        self.angle += 5
        self.update()

    def drawContents(self, painter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center_x = self.width() / 2
        center_y = self.height() / 2 - 40
        radius = 60

        for i in range(6):
            angle = (self.angle + i * 60) * math.pi / 180
            next_angle = (self.angle + (i + 1) * 60) * math.pi / 180

            x1 = center_x + math.cos(angle) * radius
            y1 = center_y + math.sin(angle) * radius
            x2 = center_x + math.cos(next_angle) * radius
            y2 = center_y + math.sin(next_angle) * radius

            pen = QPen(QColor(COLORS['accent_cyan']))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        title_font = QFont('Orbitron', 28, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(COLORS['accent_cyan']))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                        "TRICORDER\nSCLERA VEIN IDENTITY")

        sub_font = QFont('Rajdhani', 14)
        painter.setFont(sub_font)
        painter.setPen(QColor(COLORS['text_muted']))
        painter.drawText(
            20, self.height() - 40,
            self.width() - 40, 30,
            Qt.AlignmentFlag.AlignCenter,
            "BIOMETRIC AUTHENTICATION SYSTEM v2.1"
        )

# -------------------------
# Loading Overlay
# -------------------------
class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(10, 14, 26, 230);
            }}
        """)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.viz = ProcessingViz()

        self.text = QLabel("PROCESSING...")
        self.text.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent_cyan']};
                font-size: 20px;
                font-weight: bold;
                font-family: 'Orbitron';
                background: transparent;
                letter-spacing: 3px;
            }}
        """)
        self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.viz)
        layout.addWidget(self.text)
        self.setLayout(layout)

    def show_loading(self, text="PROCESSING..."):
        self.text.setText(text)
        self.viz.start_animation()
        self.show()

    def hide_loading(self):
        self.viz.stop_animation()
        self.hide()


# =========================================================
# Main App
# =========================================================
class ScleraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRICORDER - Sclera Vein Identity System")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1200, 700)
        self.apply_stylesheet()

        self.records = []
        self._current_mode = "add"   # 'add' | 'verify'

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 2px solid {COLORS['border_glow']};
                background: {COLORS['bg_dark']};
                border-radius: 0px;
                top: -2px;
            }}
            QTabBar::tab {{
                background: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
                padding: 15px 30px;
                margin-right: 2px;
                font-size: 13px;
                font-weight: 700;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
                text-transform: uppercase;
                border: 2px solid {COLORS['grid_line']};
            }}
            QTabBar::tab:selected {{
                background: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border-bottom: 3px solid {COLORS['accent_cyan']};
            }}
            QTabBar::tab:hover {{
                background: {COLORS['bg_panel']};
                color: {COLORS['accent_cyan']};
                border: 2px solid {COLORS['accent_cyan']};
            }}
        """)

        self.tabs.addTab(self.capture_tab(), "⬡ MAIN SCREEN")
        self.tabs.addTab(self.records_tab(), "⬡ RECORDS SCREEN")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.loading = LoadingOverlay(self)
        self.loading.hide()

    def apply_stylesheet(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text']};
                font-family: 'Rajdhani', 'Segoe UI', Arial, sans-serif;
            }}
            QLabel {{
                color: {COLORS['text']};
            }}
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['accent_cyan']};
                border: 2px solid {COLORS['accent_cyan']};
                border-radius: 0px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 700;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
            }}
            QLineEdit {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text']};
                border: 2px solid {COLORS['grid_line']};
                border-radius: 0px;
                padding: 10px 15px;
                font-size: 14px;
                font-family: 'Rajdhani';
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['accent_cyan']};
            }}
            QComboBox {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text']};
                border: 2px solid {COLORS['grid_line']};
                border-radius: 0px;
                padding: 10px 15px;
                font-size: 14px;
                font-family: 'Rajdhani';
            }}
            QComboBox:hover {{
                border: 2px solid {COLORS['accent_cyan']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text']};
                selection-background-color: {COLORS['accent_cyan']};
                selection-color: {COLORS['bg_dark']};
                border: 2px solid {COLORS['border_glow']};
            }}
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'loading'):
            self.loading.setGeometry(self.rect())

    # =====================================================
    # TAB 1 — CAPTURE / MATCH / REGISTER
    # =====================================================
    def capture_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QLabel("TRICORDER")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['accent_cyan']};
                font-size: 32px;
                font-weight: 900;
                font-family: 'Orbitron';
                padding: 20px;
                letter-spacing: 8px;
                border-bottom: 3px solid {COLORS['accent_cyan']};
            }}
        """)
        header.setFixedHeight(80)

        grid = QGridLayout()
        grid.setSpacing(15)
        grid.setContentsMargins(20, 20, 20, 20)

        # =========================
        # TOP LEFT - LIVE CAMERA PREVIEW
        # =========================
        live_panel, live_layout = self.create_panel("LIVE CAMERA PREVIEW")
        live_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.camera_selector = QComboBox()
        self.cameras = self.list_cameras()
        for c in self.cameras:
            self.camera_selector.addItem(f"⬡ CAMERA {c}", c)
        self.camera_selector.currentIndexChanged.connect(self.switch_camera)
        self.camera_selector.setFixedHeight(45)

        self.video_frame = QLabel()
        self.video_frame.setMinimumSize(400, 300)
        self.video_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_frame.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['camera_bg']};
                border: 3px solid {COLORS['accent_cyan']};
                border-radius: 0px;
            }}
        """)
        self.video_frame.setScaledContents(False)

        live_layout.addWidget(self.camera_selector)
        live_layout.addWidget(self.video_frame)
        live_layout.addStretch()

        # =========================
        # TOP RIGHT - CAPTURED/LOADED IMAGE OUTPUT
        # =========================
        output_panel, output_layout = self.create_panel("CAPTURED / LOADED IMAGE")
        output_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.output_frame = QLabel()
        self.output_frame.setMinimumSize(400, 300)
        self.output_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_frame.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['camera_bg']};
                border: 3px solid {COLORS['accent_magenta']};
                border-radius: 0px;
            }}
        """)
        self.output_frame.setScaledContents(False)

        output_layout.addWidget(self.output_frame)
        output_layout.addStretch()

        # =========================
        # BOTTOM LEFT - ANALYSIS RESULTS
        # =========================
        results_panel, results_layout = self.create_panel("ANALYSIS RESULTS")
        results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        results_layout.setSpacing(15)

        self.confidence_label = QLabel("--")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent_magenta']};
                font-size: 48px;
                font-weight: 900;
                font-family: 'Orbitron';
                background-color: {COLORS['bg_dark']};
                border: 4px solid {COLORS['accent_magenta']};
                border-radius: 80px;
                min-width: 160px;
                min-height: 160px;
                max-width: 160px;
                max-height: 160px;
            }}
        """)

        self.status_label = QLabel("READY TO SCAN")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['accent_cyan']};
                padding: 15px;
                border: 2px solid {COLORS['accent_cyan']};
                font-size: 13px;
                font-weight: 600;
                font-family: 'Rajdhani';
                letter-spacing: 1px;
                min-height: 120px;
            }}
        """)

        self.matched_preview = QLabel("MATCHED SAMPLE\n(Preview)")
        self.matched_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matched_preview.setMinimumSize(200, 150)
        self.matched_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['camera_bg']};
                border: 2px solid {COLORS['border_glow']};
                color: {COLORS['text_muted']};
                font-size: 11px;
                font-family: 'Rajdhani';
            }}
        """)

        confidence_row = QHBoxLayout()
        confidence_row.addWidget(self.confidence_label)
        confidence_row.addWidget(self.matched_preview)
        confidence_row.addStretch()

        results_layout.addLayout(confidence_row)
        results_layout.addWidget(self.status_label)
        results_layout.addStretch()

        # =========================
        # BOTTOM RIGHT - CONTROL INTERFACE
        # =========================
        control_panel, control_layout = self.create_panel("CONTROL INTERFACE")
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        control_layout.setSpacing(12)

        # --- Mode toggle row ---
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)

        self.btn_mode_add = QPushButton("⬡ ADD USER")
        self.btn_mode_add.setMinimumHeight(48)
        self.btn_mode_add.clicked.connect(lambda: self.set_mode("add"))

        self.btn_mode_verify = QPushButton("⬡ VERIFY USER")
        self.btn_mode_verify.setMinimumHeight(48)
        self.btn_mode_verify.clicked.connect(lambda: self.set_mode("verify"))

        mode_row.addWidget(self.btn_mode_add)
        mode_row.addWidget(self.btn_mode_verify)

        # --- User ID ---
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("⬡ ENTER USER ID")
        self.user_input.setMinimumHeight(48)

        # --- First Name / Last Name (Add User only) ---
        name_row = QHBoxLayout()
        name_row.setSpacing(8)

        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("⬡ FIRST NAME")
        self.first_name_input.setMinimumHeight(48)

        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("⬡ LAST NAME")
        self.last_name_input.setMinimumHeight(48)

        self.name_row_widget = QWidget()
        self.name_row_widget.setLayout(name_row)
        name_row.addWidget(self.first_name_input)
        name_row.addWidget(self.last_name_input)

        self.eye_selector = QComboBox()
        self.eye_selector.addItems(["⬡ Left Eye", "⬡ Right Eye"])
        self.eye_selector.setMinimumHeight(48)

        # --- Capture / Load (shared by both modes) ---
        self.btn_capture = QPushButton("⬡ CAPTURE IMAGE")
        self.btn_capture.setMinimumHeight(55)
        self.btn_capture.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['accent_cyan']};
                border: 3px solid {COLORS['accent_cyan']};
                font-size: 15px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
            }}
        """)
        self.btn_capture.clicked.connect(self.capture_image)

        self.btn_load = QPushButton("⬡ LOAD IMAGE")
        self.btn_load.setMinimumHeight(55)
        self.btn_load.clicked.connect(self.load_image)

        # --- Save (Add User mode only) ---
        self.btn_register = QPushButton("⬡ SAVE")
        self.btn_register.setMinimumHeight(55)
        self.btn_register.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['success']};
                border: 3px solid {COLORS['success']};
                font-size: 15px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success']};
                color: {COLORS['bg_dark']};
            }}
        """)
        self.btn_register.clicked.connect(self.call_register)

        # --- Start Scan (Verify User mode only) ---
        self.btn_match = QPushButton("⬡ START SCAN")
        self.btn_match.setMinimumHeight(55)
        self.btn_match.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['accent_magenta']};
                border: 3px solid {COLORS['accent_magenta']};
                font-size: 15px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_magenta']};
                color: {COLORS['bg_dark']};
            }}
        """)
        self.btn_match.clicked.connect(self.call_identify)

        # --- Timestamp ---
        self.timestamp_label = QLabel()
        self.timestamp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timestamp_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_muted']};
                font-size: 12px;
                font-family: 'Rajdhani';
                letter-spacing: 1px;
                padding: 10px;
                background: transparent;
            }}
        """)
        self.update_timestamp()

        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_timestamp)
        self.time_timer.start(1000)

        # Assemble control panel
        control_layout.addLayout(mode_row)
        control_layout.addWidget(self.user_input)
        control_layout.addWidget(self.name_row_widget)
        control_layout.addWidget(self.eye_selector)
        control_layout.addWidget(self.btn_capture)
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_register)   # Add User only
        control_layout.addWidget(self.btn_match)      # Verify User only
        control_layout.addStretch()
        control_layout.addWidget(self.timestamp_label)

        # =========================
        # ADD TO GRID (2x2)
        # =========================
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        grid.addWidget(live_panel, 0, 0)
        grid.addWidget(output_panel, 0, 1)
        grid.addWidget(results_panel, 1, 0)
        grid.addWidget(control_panel, 1, 1)

        main_layout.addWidget(header)
        main_layout.addLayout(grid)
        tab.setLayout(main_layout)

        if self.cameras:
            self.start_camera(self.cameras[0])

        # Apply initial mode styling
        self.set_mode("add")

        return tab

    # =====================================================
    # MODE TOGGLE
    # =====================================================
    def set_mode(self, mode: str):
        """Switch control interface between 'add' and 'verify' modes."""
        self._current_mode = mode

        is_add = mode == "add"

        # Active / inactive toggle button styling
        active_style = f"""
            QPushButton {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
                border: 3px solid {COLORS['accent_cyan']};
                font-size: 13px;
                font-weight: 900;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
            }}
        """
        inactive_style = f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text_muted']};
                border: 2px solid {COLORS['grid_line']};
                font-size: 13px;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
            }}
            QPushButton:hover {{
                border: 2px solid {COLORS['accent_cyan']};
                color: {COLORS['accent_cyan']};
            }}
        """
        self.btn_mode_add.setStyleSheet(active_style if is_add else inactive_style)
        self.btn_mode_verify.setStyleSheet(inactive_style if is_add else active_style)

        # Show / hide name fields and action buttons
        self.name_row_widget.setVisible(is_add)
        self.btn_register.setVisible(is_add)
        self.btn_match.setVisible(not is_add)

        # Eye selector: show in both modes (needed for add; optional for verify but harmless)
        self.eye_selector.setVisible(is_add)

        # Reset status
        mode_label = "ADD USER MODE — Fill in details and capture an eye image, then press SAVE." \
                     if is_add else \
                     "VERIFY USER MODE — Capture or load an eye image, then press START SCAN."
        self.update_status(mode_label, "info")

    def create_panel(self, title):
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_panel']};
                border: 2px solid {COLORS['grid_line']};
                border-radius: 0px;
            }}
        """)

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(15, 15, 15, 15)
        panel_layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent_cyan']};
                font-size: 13px;
                font-weight: 700;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
                padding: 5px;
                background: transparent;
                border-bottom: 2px solid {COLORS['grid_line']};
            }}
        """)

        panel_layout.addWidget(title_label)
        panel.setLayout(panel_layout)
        return panel, panel_layout

    def update_timestamp(self):
        now = datetime.now()
        timestamp = now.strftime("%a %b %d %Y %I:%M:%S %p %Z").upper()
        self.timestamp_label.setText(timestamp)

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
        if not self.cap.isOpened():
            print(f"ERROR: Failed to open camera {index}")
            return
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        self.camera_timer.start(30)

    def pause_camera(self):
        if hasattr(self, "camera_timer") and self.camera_timer and self.camera_timer.isActive():
            try:
                self.camera_timer.stop()
            except Exception:
                pass
        if hasattr(self, "cap") and self.cap:
            try:
                if hasattr(self.cap, "release"):
                    self.cap.release()
            except Exception:
                pass
            self.cap = None

    def resume_camera(self):
        if getattr(self, "cameras", None):
            cam = None
            try:
                cam = self.camera_selector.currentData()
            except Exception:
                pass
            if cam is None:
                cam = self.cameras[0]
            self.start_camera(cam)

    def switch_camera(self):
        self.camera_timer.stop()
        self.cap.release()
        cam = self.camera_selector.currentData()
        self.start_camera(cam)

    def update_frame(self):
        if not hasattr(self, 'cap') or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixmap = np_to_pixmap(self.current_frame)
        scaled_pixmap = pixmap.scaled(
            self.video_frame.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_frame.setPixmap(scaled_pixmap)

    def capture_image(self):
        self.loading.show_loading("CAPTURING SHARPEST FRAME...")
        QApplication.processEvents()

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
            pixmap = np_to_pixmap(self.captured)
            scaled_pixmap = pixmap.scaled(
                self.output_frame.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.output_frame.setPixmap(scaled_pixmap)
            self.update_status("✓ IMAGE CAPTURED SUCCESSFULLY", "success")

        self.loading.hide_loading()

    def load_image(self):
        self.pause_camera()

        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Image", os.getcwd(), "Images (*.jpg *.png *.jpeg)"
        )

        try:
            self.resume_camera()
        except Exception:
            pass

        if not fname:
            self.update_status("✗ NO FILE SELECTED", "error")
            return

        self.loading.show_loading("LOADING IMAGE...")
        QApplication.processEvents()

        img_bgr = cv2.imread(fname)
        if img_bgr is None:
            self.update_status("✗ FAILED TO READ IMAGE", "error")
            self.loading.hide_loading()
            return

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.captured = img
        pixmap = np_to_pixmap(img)
        scaled_pixmap = pixmap.scaled(
            self.output_frame.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.output_frame.setPixmap(scaled_pixmap)
        self.update_status("✓ IMAGE LOADED SUCCESSFULLY", "success")
        self.loading.hide_loading()

    # =====================================================
    # API CALLS
    # =====================================================
    def call_register(self):
        """Add User — calls POST /segment."""
        if not hasattr(self, "captured"):
            self.update_status("✗ NO IMAGE CAPTURED OR LOADED", "error")
            return

        user    = self.user_input.text().strip()
        fname   = self.first_name_input.text().strip()
        lname   = self.last_name_input.text().strip()
        eye     = self.eye_selector.currentText().replace("⬡ ", "").replace(" Eye", "")

        if not user:
            self.update_status("✗ USER ID REQUIRED", "error")
            return

        self.loading.show_loading("REGISTERING USER...")
        QApplication.processEvents()

        _, buf = cv2.imencode(".jpg", cv2.cvtColor(self.captured, cv2.COLOR_RGB2BGR))
        files = {"image": ("capture.jpg", buf.tobytes(), "image/jpeg")}
        data  = {
            "user_id":    user,
            "eye_side":   eye,
            "first_name": fname,
            "last_name":  lname,
        }

        try:
            r = requests.post(f"{API_BASE}/segment", files=files, data=data, timeout=10)

            if r.status_code == 200:
                response_data = r.json()

                if "processed_image" in response_data:
                    self.display_base64_image(response_data["processed_image"], self.output_frame)

                sample = response_data.get("sample", "--")
                self.confidence_label.setText("100%")

                display_name = f"{fname} {lname}".strip() or user
                self.update_status(
                    f"✓ SUCCESSFULLY REGISTERED\n\n"
                    f"⬡ USER: {display_name} ({user})\n"
                    f"⬡ EYE: {eye}\n"
                    f"⬡ SAMPLE: {sample}",
                    "success"
                )

                # Add live row to Records tab
                self.log_record(
                    action="REGISTER",
                    user_id=user,
                    similarity=1.0,
                    eye_side=eye,
                    sample=sample,
                    record_id=response_data.get("record_id", ""),
                )
            else:
                self.update_status(f"✗ REGISTRATION FAILED — {r.status_code}", "error")

        except Exception as e:
            self.update_status(f"✗ CONNECTION ERROR: {str(e)}", "error")

        self.loading.hide_loading()

    def call_identify(self):
        """Verify User — calls POST /identify."""
        if not hasattr(self, "captured"):
            self.update_status("✗ NO IMAGE CAPTURED OR LOADED", "error")
            return

        self.loading.show_loading("IDENTIFYING...")
        QApplication.processEvents()

        _, buf = cv2.imencode(".jpg", cv2.cvtColor(self.captured, cv2.COLOR_RGB2BGR))
        files = {"image": ("query.jpg", buf.tobytes(), "image/jpeg")}

        try:
            r = requests.post(f"{API_BASE}/identify", files=files, timeout=10)

            if r.status_code != 200:
                self.update_status("✗ IDENTIFICATION FAILED — SERVER ERROR", "error")
                self.confidence_label.setText("0%")
                self.loading.hide_loading()
                return

            response_data = r.json()

            if "processed_query_image" in response_data:
                self.display_base64_image(response_data["processed_query_image"], self.output_frame)

            best_match = response_data.get("best_match")

            if not best_match:
                self.update_status("✗ NO MATCHES FOUND IN DATABASE", "error")
                self.confidence_label.setText("0%")
                self.loading.hide_loading()
                return

            user   = best_match.get("user_id", "Unknown")
            eye    = best_match.get("eye_side", "?")
            sample = best_match.get("sample", "")
            sim    = best_match.get("similarity", 0.0)

            if "matched_image" in best_match and best_match["matched_image"]:
                self.display_base64_image(best_match["matched_image"], self.matched_preview)

            self.confidence_label.setText(f"{int(sim * 100)}%")

            if sim >= 0.8:
                status_text = (
                    f"FINAL VERIFICATION: ACCESS GRANTED\n\n"
                    f"⬡ USER: {user}\n⬡ EYE: {eye}\n"
                    f"⬡ SAMPLE: {sample}\n⬡ CONFIDENCE: {sim:.3f}"
                )
                self.update_status(status_text, "success")
            else:
                status_text = (
                    f"FINAL VERIFICATION: LOW CONFIDENCE\n\n"
                    f"⬡ MATCH: {user}\n⬡ EYE: {eye}\n"
                    f"⬡ SAMPLE: {sample}\n⬡ CONFIDENCE: {sim:.3f}"
                )
                self.update_status(status_text, "error")

            self.log_record(
                action="MATCH ✓" if sim >= 0.8 else "MATCH ✗",
                user_id=user,
                similarity=sim,
                eye_side=eye,
                sample=sample,
                record_id=response_data.get("record_id", ""),
            )

        except Exception as e:
            self.update_status(f"✗ CONNECTION ERROR: {str(e)}", "error")

        self.loading.hide_loading()

    def update_status(self, text, status_type="info"):
        color_map = {
            "success": COLORS['success'],
            "error":   COLORS['error'],
            "info":    COLORS['accent_cyan'],
        }
        border_color = color_map.get(status_type, COLORS['accent_cyan'])
        text_color   = color_map.get(status_type, COLORS['accent_cyan'])

        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_card']};
                color: {text_color};
                padding: 15px 30px;
                border: 3px solid {border_color};
                font-size: 14px;
                font-weight: 700;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
                min-height: 60px;
            }}
        """)

    def display_base64_image(self, base64_string: str, target_label: QLabel):
        try:
            img_bytes = base64.b64decode(base64_string)
            nparr     = np.frombuffer(img_bytes, np.uint8)
            img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                print("ERROR: Failed to decode base64 image")
                return

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pixmap  = np_to_pixmap(img_rgb)

            scaled_pixmap = pixmap.scaled(
                target_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            target_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"ERROR: Failed to display base64 image: {e}")

    # =====================================================
    # TAB 2 — RECORDS
    # =====================================================
    def populate_records_from_logs(self):
        try:
            logs = fetch_logs_from_stream(API_BASE)
        except Exception as e:
            print(f"[Records] Could not load logs on startup: {e}")
            return

        for entry in logs:
            action_type = entry.get("action", "")
            ts_raw      = entry.get("timestamp", "")
            record_id   = entry.get("record_id", "")

            try:
                ts = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = ts_raw

            if action_type == "new_user":
                self._insert_log_row(
                    ts        = ts,
                    action    = "REGISTER",
                    user_id   = entry.get("user_id", "--"),
                    eye_side  = entry.get("eye_side", "--"),
                    sample    = entry.get("sample", "--"),
                    sim       = None,
                    record_id = record_id,
                )
            elif action_type == "match":
                matched      = entry.get("matched", False)
                action_label = "MATCH ✓" if matched else "MATCH ✗"
                self._insert_log_row(
                    ts        = ts,
                    action    = action_label,
                    user_id   = entry.get("best_match_user_id") or "--",
                    eye_side  = entry.get("best_match_eye_side") or "--",
                    sample    = entry.get("best_match_sample") or "--",
                    sim       = entry.get("best_match_similarity"),
                    record_id = record_id,
                )

    def _insert_log_row(self, ts, action, user_id, eye_side, sample, sim, record_id=""):
        """Append one row to the Records table. Stores record_id as hidden UserRole data."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        ts_item     = QTableWidgetItem(ts)
        action_item = QTableWidgetItem(action)
        user_item   = QTableWidgetItem(user_id)
        eye_item    = QTableWidgetItem(eye_side)
        sample_item = QTableWidgetItem(sample)

        # Stash record_id on the timestamp cell for later retrieval
        ts_item.setData(Qt.ItemDataRole.UserRole, record_id)

        ts_item.setForeground(QColor(COLORS['text_muted']))

        if action == "REGISTER":
            action_item.setForeground(QColor(COLORS['success']))
        elif action == "MATCH ✓":
            action_item.setForeground(QColor(COLORS['success']))
        else:
            action_item.setForeground(QColor(COLORS['error']))

        if sim is None:
            sim_item = QTableWidgetItem("--")
            sim_item.setForeground(QColor(COLORS['text_muted']))
        else:
            sim_item = QTableWidgetItem(f"{sim:.3f}")
            if sim >= 0.75:
                sim_item.setForeground(QColor(COLORS['success']))
            elif sim >= 0.5:
                sim_item.setForeground(QColor(COLORS['accent_cyan']))
            else:
                sim_item.setForeground(QColor(COLORS['error']))

        self.table.setItem(row, 0, ts_item)
        self.table.setItem(row, 1, action_item)
        self.table.setItem(row, 2, user_item)
        self.table.setItem(row, 3, eye_item)
        self.table.setItem(row, 4, sample_item)
        self.table.setItem(row, 5, sim_item)

    def records_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("SYSTEM LOGS - SCAN HISTORY")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['accent_cyan']};
                font-size: 20px;
                font-weight: 900;
                font-family: 'Rajdhani';
                padding: 20px;
                letter-spacing: 4px;
                border-bottom: 3px solid {COLORS['accent_cyan']};
            }}
        """)
        header.setFixedHeight(70)

        # Action bar (Export + Delete)
        action_bar = QWidget()
        action_bar.setStyleSheet(f"background-color: {COLORS['bg_card']};")
        action_bar.setFixedHeight(60)
        ab_layout = QHBoxLayout()
        ab_layout.setContentsMargins(20, 8, 20, 8)
        ab_layout.setSpacing(12)

        btn_export = QPushButton("⬡ EXPORT LOGS")
        btn_export.setFixedHeight(42)
        btn_export.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['accent_cyan']};
                border: 2px solid {COLORS['accent_cyan']};
                font-size: 13px;
                font-weight: 700;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
                padding: 0 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
            }}
        """)
        btn_export.clicked.connect(self.export_logs)

        btn_delete = QPushButton("⬡ DELETE RECORD")
        btn_delete.setFixedHeight(42)
        btn_delete.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['error']};
                border: 2px solid {COLORS['error']};
                font-size: 13px;
                font-weight: 700;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
                padding: 0 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['error']};
                color: {COLORS['bg_dark']};
            }}
        """)
        btn_delete.clicked.connect(self.delete_selected_record)

        ab_layout.addWidget(btn_export)
        ab_layout.addWidget(btn_delete)
        ab_layout.addStretch()
        action_bar.setLayout(ab_layout)

        # Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "⬡ TIMESTAMP", "⬡ ACTION", "⬡ USER ID", "⬡ EYE SIDE", "⬡ SAMPLE", "⬡ SIMILARITY"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text']};
                border: 2px solid {COLORS['grid_line']};
                gridline-color: {COLORS['grid_line']};
                font-family: 'Rajdhani';
                font-size: 13px;
            }}
            QTableWidget::item {{
                padding: 15px;
                border-bottom: 1px solid {COLORS['grid_line']};
            }}
            QTableWidget::item:selected {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['accent_cyan']};
                padding: 15px;
                border: none;
                font-weight: bold;
                font-size: 13px;
                font-family: 'Rajdhani';
                letter-spacing: 2px;
                border-bottom: 2px solid {COLORS['accent_cyan']};
            }}
        """)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        layout.addWidget(header)
        layout.addWidget(action_bar)
        layout.addWidget(self.table)
        tab.setLayout(layout)
        return tab

    def log_record(self, action: str, user_id: str, similarity: float,
                   eye_side: str = "--", sample: str = "--", record_id: str = ""):
        """
        Insert a new record at the TOP of the table (newest first).
        Called live after a successful API response.
        """
        self.table.insertRow(0)

        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_item = QTableWidgetItem(ts)
        ts_item.setData(Qt.ItemDataRole.UserRole, record_id)
        ts_item.setForeground(QColor(COLORS['text_muted']))

        action_item = QTableWidgetItem(action)
        if action == "REGISTER":
            action_item.setForeground(QColor(COLORS['success']))
        elif action == "MATCH ✓":
            action_item.setForeground(QColor(COLORS['success']))
        elif action == "MATCH ✗":
            action_item.setForeground(QColor(COLORS['error']))
        else:
            action_item.setForeground(QColor(COLORS['accent_cyan']))

        user_item   = QTableWidgetItem(user_id  or "--")
        eye_item    = QTableWidgetItem(eye_side or "--")
        sample_item = QTableWidgetItem(sample   or "--")

        if action == "REGISTER":
            sim_item = QTableWidgetItem("--")
            sim_item.setForeground(QColor(COLORS['text_muted']))
        else:
            sim_item = QTableWidgetItem(f"{similarity:.3f}")
            if similarity >= 0.75:
                sim_item.setForeground(QColor(COLORS['success']))
            elif similarity >= 0.5:
                sim_item.setForeground(QColor(COLORS['accent_cyan']))
            else:
                sim_item.setForeground(QColor(COLORS['error']))

        self.table.setItem(0, 0, ts_item)
        self.table.setItem(0, 1, action_item)
        self.table.setItem(0, 2, user_item)
        self.table.setItem(0, 3, eye_item)
        self.table.setItem(0, 4, sample_item)
        self.table.setItem(0, 5, sim_item)

    # =====================================================
    # RECORDS — EXPORT
    # =====================================================
    def export_logs(self):
        """Download the audit log from GET /logs/export and save locally."""
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Audit Log", "audit_log.jsonl",
            "JSONL Files (*.jsonl);;All Files (*)"
        )
        if not save_path:
            return

        try:
            r = requests.get(f"{API_BASE}/logs/export", timeout=15, stream=True)
            r.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

            QMessageBox.information(
                self, "Export Complete",
                f"Audit log exported successfully to:\n{save_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export logs:\n{str(e)}")

    # =====================================================
    # RECORDS — DELETE
    # =====================================================
    def delete_selected_record(self):
        """Delete the selected row's log entry via DELETE /logs/{record_id}."""
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a row to delete.")
            return

        row = self.table.currentRow()
        ts_item   = self.table.item(row, 0)
        record_id = ts_item.data(Qt.ItemDataRole.UserRole) if ts_item else ""

        if not record_id:
            QMessageBox.warning(
                self, "Cannot Delete",
                "This record has no ID (it may have been loaded from a legacy log).\n"
                "Only records created after the last backend update can be deleted."
            )
            return

        confirm = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete this log entry?\n\nTimestamp: {ts_item.text()}\nRecord ID: {record_id}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            r = requests.delete(f"{API_BASE}/logs/{record_id}", timeout=10)
            if r.status_code == 200:
                self.table.removeRow(row)
            elif r.status_code == 404:
                QMessageBox.warning(self, "Not Found", "Record not found on server — removing from view only.")
                self.table.removeRow(row)
            else:
                QMessageBox.critical(self, "Delete Failed", f"Server returned {r.status_code}:\n{r.text}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Could not reach server:\n{str(e)}")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = ModernSplash()
    splash.show()
    app.processEvents()

    QTimer.singleShot(2000, lambda: (splash.close(), show_main()))

    def show_main():
        global win
        win = ScleraApp()
        win.show()
        QTimer.singleShot(100, win.populate_records_from_logs)

    sys.exit(app.exec())
