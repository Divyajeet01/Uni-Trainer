import sys
import os
import pandas as pd

from image_classification.yolo_cls import (
    train_with_user_data as yolo_cls_train_with_user_data,
    validate_dataset_structure as cls_validate_dataset_structure,
    split_dataset,
)
from image_classification.resnet import (
    train_with_user_data as resnet_train_with_user_data,
)
from object_detection.yolo_obj import (
    train_with_user_data as yolo_det_train_with_user_data,
    validate_dataset_structure as det_validate_dataset_structure,
    prepare_data_yaml,
)
from object_detection.rt_detr_obj import (
    train_with_user_data as rtdetr_train_with_user_data,
)
from segmentation.yolo_seg import (
    train_with_user_data as yolo_seg_train_with_user_data,
    validate_dataset_structure as seg_validate_dataset_structure,
    prepare_data_yaml as seg_prepare_data_yaml,
)
from segmentation.sam2_seg import (
    train_with_user_data as sam2_train_with_user_data,
)
from ml_regression.xgboost_reg import (
    train_with_user_data as xgboost_train_with_user_data,
)
from ml_regression.lightgbm_reg import (
    train_with_user_data as lightgbm_train_with_user_data,
)
from ml_regression.random_forest_reg import (
    train_with_user_data as rf_train_with_user_data,
)
from ml_regression.neural_net_reg import (
    train_with_user_data as nn_train_with_user_data,
)

from PyQt5.QtWidgets import (QApplication,QSizePolicy, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QStackedWidget, QFileDialog, QScrollArea,
                             QComboBox, QSlider, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import psutil

# Import YOLO backend

class NoWheelComboBox(QComboBox):
    """ComboBox that ignores mouse wheel events to prevent accidental scrolling"""
    def wheelEvent(self, event):
        event.ignore()

class GPUDetectWorker(QThread):
    gpu_ready = pyqtSignal(str)

    def run(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self.gpu_ready.emit(f"{name} Detected")
            else:
                self.gpu_ready.emit("No GPU Detected")
        except Exception:
            self.gpu_ready.emit("GPU Detection Error")

class TrainingWorker(QThread):
    """Background thread for model training to keep UI responsive."""

    progress_updated = pyqtSignal(int, int, dict)  # epoch, total_epochs, metrics
    training_finished = pyqtSignal(str, bool)  # model_path, success
    log_message = pyqtSignal(str)  # log messages

    def __init__(self, data_path, config, framework="YOLO-Cls", task="Image Classification"):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.framework = framework
        self.task = task
        self.is_running = True

    def run(self):
        """Execute training in background thread."""
        import sys
        import io

        # Create a custom stdout/stderr capture class
        class OutputCapture(io.StringIO):
            def __init__(self, signal, original_stream):
                super().__init__()
                self.signal = signal
                self.original = original_stream
                self.buffer_line = ""

            def write(self, text):
                # Write to original stream
                if self.original:
                    self.original.write(text)
                    self.original.flush()

                # Buffer and emit complete lines
                self.buffer_line += text
                while '\n' in self.buffer_line:
                    line, self.buffer_line = self.buffer_line.split('\n', 1)
                    if line.strip():  # Only emit non-empty lines
                        self.signal.emit(line.strip())

                # Handle carriage return (progress bars)
                if '\r' in self.buffer_line:
                    lines = self.buffer_line.split('\r')
                    if lines[-1].strip():
                        self.signal.emit(lines[-1].strip())
                    self.buffer_line = lines[-1]

            def flush(self):
                if self.original:
                    self.original.flush()

        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            # Set up output capture
            sys.stdout = OutputCapture(self.log_message, old_stdout)
            sys.stderr = OutputCapture(self.log_message, old_stderr)

            self.log_message.emit(f"[INFO] Starting training with {self.config['model']}")
            self.log_message.emit(f"[INFO] Task: {self.task}")
            self.log_message.emit(f"[INFO] Framework: {self.framework}")
            self.log_message.emit(f"[INFO] Dataset: {self.data_path}")
            if 'epochs' in self.config:
                self.log_message.emit(f"[INFO] Epochs: {self.config['epochs']}")
            if 'n_estimators' in self.config:
                self.log_message.emit(f"[INFO] Trees/Estimators: {self.config['n_estimators']}")

            # Select training function based on task and framework
            if self.task == "Regression":
                if self.framework == "XGBoost":
                    model_path, results = xgboost_train_with_user_data(
                        data_path=self.data_path,
                        config=self.config
                    )
                elif self.framework == "LightGBM":
                    model_path, results = lightgbm_train_with_user_data(
                        data_path=self.data_path,
                        config=self.config
                    )
                elif self.framework == "RandomForest":
                    model_path, results = rf_train_with_user_data(
                        data_path=self.data_path,
                        config=self.config
                    )
                elif self.framework == "NeuralNetwork":
                    model_path, results = nn_train_with_user_data(
                        data_path=self.data_path,
                        config=self.config
                    )
                else:
                    raise ValueError(f"Unknown regression framework: {self.framework}")
            elif self.task == "Object Detection":
                if self.framework == "RT-DETR":
                    model_path, results = rtdetr_train_with_user_data(
                        data_path=self.data_path,
                        config=self.config
                    )
                else:
                    model_path, results = yolo_det_train_with_user_data(
                        data_path=self.data_path,
                        config=self.config
                    )
            elif self.task == "Segmentation":
                if self.framework == "SAM":
                    model_path, results = sam2_train_with_user_data(
                        data_path=self.data_path,
                        config=self.config
                    )
                else:
                    model_path, results = yolo_seg_train_with_user_data(
                        data_path=self.data_path,
                        config=self.config
                    )
            elif self.framework == "ResNet":
                model_path, results = resnet_train_with_user_data(
                    data_path=self.data_path,
                    config=self.config
                )
            else:
                model_path, results = yolo_cls_train_with_user_data(
                    data_path=self.data_path,
                    config=self.config
                )

            self.log_message.emit(f"[SUCCESS] Training complete!")
            self.log_message.emit(f"[SUCCESS] Model saved: {model_path}")
            self.training_finished.emit(model_path, True)

        except Exception as e:
            self.log_message.emit(f"[ERROR] Training failed: {str(e)}")
            import traceback
            self.log_message.emit(f"[ERROR] {traceback.format_exc()}")
            self.training_finished.emit("", False)
        finally:
            # Restore original stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def stop(self):
        """Stop training (request)."""
        self.is_running = False


class ResponsiveWidget(QWidget):
    """Base class for responsive widgets"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_width = 800
        self.min_height = 600
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_layout()
    
    def adjust_layout(self):
        pass

class WelcomeScreen(ResponsiveWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Title
        self.title = QLabel("Uni Trainer")
        self.title.setFont(QFont("Arial", 48, QFont.Light))
        self.title.setStyleSheet("color: #666;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Subtitle
        self.subtitle = QLabel("AI TRAINING DESKTOP APPLICATION")
        self.subtitle.setFont(QFont("Arial", 11))
        self.subtitle.setStyleSheet("color: #999; letter-spacing: 2px;")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Start button
        self.start_btn = QPushButton("START TRAINING")
        self.start_btn.setFont(QFont("Arial", 10))
        self.start_btn.setMinimumSize(180, 40)
        self.start_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #666;
                color: #666;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        self.start_btn.clicked.connect(self.start_training)
        
        layout.addStretch()
        layout.addWidget(self.title)
        layout.addSpacing(10)
        layout.addWidget(self.subtitle)
        layout.addSpacing(40)
        layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #e8e6e1;")
    
    def adjust_layout(self):
        width = self.width()
        if width < 600:
            self.title.setFont(QFont("Arial", 32, QFont.Light))
            self.subtitle.setFont(QFont("Arial", 9))
        elif width < 900:
            self.title.setFont(QFont("Arial", 40, QFont.Light))
            self.subtitle.setFont(QFont("Arial", 10))
        else:
            self.title.setFont(QFont("Arial", 48, QFont.Light))
            self.subtitle.setFont(QFont("Arial", 11))
    
    def start_training(self):
        self.parent().setCurrentIndex(1)

class FileDropZone(QFrame):
    def __init__(self, on_files_changed=None):
        super().__init__()
        self.init_ui()
        self.files = []
        self.dataset_path = None  # Path to dataset folder
        self.on_files_changed = on_files_changed

    def clear(self):
        self.files = []
        self.dataset_path = None
        self.text.setText("Drop dataset folder here or click to upload")
        if self.on_files_changed:
            self.on_files_changed(self.files)
        
    def init_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Folder icon
        self.icon_label = QLabel("📁")
        self.icon_label.setFont(QFont("Arial", 48))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Instructions
        self.text = QLabel("Drop dataset folder here or click to upload")
        self.text.setFont(QFont("Arial", 11))
        self.text.setStyleSheet("color: #666;")
        self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text.setWordWrap(True)
        
        # File types
        self.types = QLabel("Folder with class subfolders (e.g., cats/, dogs/)")
        self.types.setFont(QFont("Arial", 9))
        self.types.setStyleSheet("color: #999;")
        self.types.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.types)
        
        self.setLayout(self.layout)
        self.setFrameShape(QFrame.Box)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #ccc;
                border-radius: 5px;
                background-color: #fafafa;
            }
        """)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        # Show dialog to select folder (dataset) or files
        msg = QMessageBox()
        msg.setWindowTitle("Select Input Type")
        msg.setText("What would you like to upload?")
        folder_btn = msg.addButton("Dataset Folder", QMessageBox.ActionRole)
        csv_btn = msg.addButton("CSV File (Regression)", QMessageBox.ActionRole)
        files_btn = msg.addButton("Individual Files", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec_()

        if msg.clickedButton() == folder_btn:
            # Select folder (dataset)
            folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
            if folder:
                self.dataset_path = folder
                self.text.setText(f"Dataset: {os.path.basename(folder)}")
                self.icon_label.setText("📂")
                if self.on_files_changed:
                    self.on_files_changed([folder])
        elif msg.clickedButton() == csv_btn:
            # Select CSV file for regression
            csv_file, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
            if csv_file:
                self.dataset_path = csv_file
                self.text.setText(f"CSV: {os.path.basename(csv_file)}")
                self.icon_label.setText("📊")
                if self.on_files_changed:
                    self.on_files_changed([csv_file])
        elif msg.clickedButton() == files_btn:
            # Select individual files
            files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*)")
            if files:
                self.files.extend(files)
                self.update_display()
                if self.on_files_changed:
                    self.on_files_changed(self.files)

    def update_display(self):
        if self.files:
            self.text.setText(f"{len(self.files)} files selected")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.width()
        if width < 300:
            self.icon_label.setFont(QFont("Arial", 32))
            self.text.setFont(QFont("Arial", 9))
            self.types.setFont(QFont("Arial", 7))
        else:
            self.icon_label.setFont(QFont("Arial", 48))
            self.text.setFont(QFont("Arial", 11))
            self.types.setFont(QFont("Arial", 9))

class MainScreen(ResponsiveWidget):

    MODEL_ZOO = {
    "Image Classification": {
        "YOLO-Cls": {
            "versions": {
                "YOLOv8n-cls (nano - fastest)": "yolov8n-cls.pt",
                "YOLOv8s-cls (small)": "yolov8s-cls.pt",
                "YOLOv8m-cls (medium)": "yolov8m-cls.pt",
                "YOLOv8l-cls (large)": "yolov8l-cls.pt",
                "YOLOv8x-cls (extra large - most accurate)": "yolov8x-cls.pt",
                "YOLO11n-cls (YOLO11 nano)": "yolo11n-cls.pt",
                "YOLO11s-cls (YOLO11 small)": "yolo11s-cls.pt",
            },
            "default_config": {
                "epochs": 50,
                "imgsz": 224,
                "batch": 16,
            }
        },
        "ResNet": {
            "versions": {
                "ResNet18 (smallest, fastest)": "resnet18",
                "ResNet34": "resnet34",
                "ResNet50 (good balance)": "resnet50",
                "ResNet101": "resnet101",
                "ResNet152 (largest, most accurate)": "resnet152",
            },
            "default_config": {
                "epochs": 50,
                "imgsz": 224,
                "batch": 32,
            }
        }
    },
    "Object Detection": {
        "YOLO-OBB": {
            "versions": {
                "YOLO11n-obb (nano - fastest)": "yolo11n-obb.pt",
                "YOLO11s-obb (small)": "yolo11s-obb.pt",
                "YOLO11m-obb (medium)": "yolo11m-obb.pt",
                "YOLO11l-obb (large)": "yolo11l-obb.pt",
                "YOLO11x-obb (extra large - most accurate)": "yolo11x-obb.pt",
                "YOLOv8n-obb (nano)": "yolov8n-obb.pt",
                "YOLOv8s-obb (small)": "yolov8s-obb.pt",
                "YOLOv8m-obb (medium)": "yolov8m-obb.pt",
                "YOLOv8l-obb (large)": "yolov8l-obb.pt",
                "YOLOv8x-obb (extra large)": "yolov8x-obb.pt",

            },
            "default_config": {
                "epochs": 100,
                "imgsz": 640,
                "batch": 16,
            }
        },
        "RT-DETR": {
            "versions": {
                "RT-DETR-L": "rtdetr-l.pt",
                "RT-DETR-X": "rtdetr-x.pt",
            },
            "default_config": {
                "epochs": 100,
                "imgsz": 640,
                "batch": 8,
            }
        }
    },
    "Segmentation": {
        "YOLO-Seg": {
            "versions": {
                "YOLO11n-seg (nano)": "yolo11n-seg.pt",
                "YOLO11s-seg (small)": "yolo11s-seg.pt",
                "YOLO11m-seg (medium)": "yolo11m-seg.pt",
                "YOLOv8n-seg (nano)": "yolov8n-seg.pt",
                "YOLOv8s-seg (small)": "yolov8s-seg.pt",
                "YOLOv8m-seg (medium)": "yolov8m-seg.pt",
            },
            "default_config": {
                "epochs": 100,
                "imgsz": 640,
                "batch": 16,
            }
        },
        "SAM": {
            "versions": {
                "SAM-b (base)": "sam_b.pt",
                "SAM-l (large)": "sam_l.pt",
                "SAM 2 (tiny)": "sam2_t.pt",
                "SAM 2 (small)": "sam2_s.pt",
                "SAM 2 (base)": "sam2_b.pt",
            },
            "default_config": {
                "epochs": 50,
                "imgsz": 1024,
                "batch": 4,
            }
        }
    },
    "Regression": {
        "XGBoost": {
            "versions": {
                "XGBoost (default)": "xgboost",
                "XGBoost (light - fast)": "xgboost_light",
                "XGBoost (deep - accurate)": "xgboost_deep",
            },
            "default_config": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            }
        },
        "LightGBM": {
            "versions": {
                "LightGBM (default)": "lightgbm",
                "LightGBM (light - fast)": "lightgbm_light",
                "LightGBM (deep - accurate)": "lightgbm_deep",
            },
            "default_config": {
                "n_estimators": 100,
                "max_depth": -1,
                "learning_rate": 0.1,
            }
        },
        "RandomForest": {
            "versions": {
                "RandomForest (default)": "randomforest",
                "RandomForest (light - 50 trees)": "randomforest_light",
                "RandomForest (heavy - 200 trees)": "randomforest_heavy",
            },
            "default_config": {
                "n_estimators": 100,
                "max_depth": -1,
            }
        },
        "NeuralNetwork": {
            "versions": {
                "Neural Net (small - 64,32)": "nn_small",
                "Neural Net (medium - 128,64,32)": "nn_medium",
                "Neural Net (large - 256,128,64,32)": "nn_large",
            },
            "default_config": {
                "epochs": 100,
                "batch": 32,
                "learning_rate": 0.001,
            }
        }
    },
}


    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_horizontal = True
        self.init_ui()
        self.start_detection()
        
    def init_ui(self):
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Left sidebar with scroll
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_scroll.setStyleSheet("QScrollArea { border: none; }")
        self.left_panel = self.create_left_panel()
        self.left_scroll.setWidget(self.left_panel)
        
        # Right panel with scroll
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.right_scroll.setStyleSheet("QScrollArea { border: none; }")
        self.right_panel = self.create_right_panel()
        self.right_scroll.setWidget(self.right_panel)
        
        self.main_layout.addWidget(self.left_scroll, 1)
        self.main_layout.addWidget(self.right_scroll, 2)
        
        self.setLayout(self.main_layout)
        self.setStyleSheet("background-color: #f5f5f5;")
    
    def create_left_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background-color: #e8e6e1;")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)
        
        # Header
        self.header_title = QLabel("UNI TRAINER")
        self.header_title.setFont(QFont("Arial", 16, QFont.Bold))
        self.header_title.setStyleSheet("color: #333;")
        
        self.header_subtitle = QLabel("AI TRAINING CONTROL")
        self.header_subtitle.setFont(QFont("Arial", 9))
        self.header_subtitle.setStyleSheet("color: #999;")
        
        layout.addWidget(self.header_title)
        layout.addWidget(self.header_subtitle)
        layout.addSpacing(15)
        
        # System section
        self.system_frame = self.create_system_section()
        layout.addWidget(self.system_frame)
        
        # Training data section
        self.training_frame = self.create_training_data_section()
        layout.addWidget(self.training_frame)
        
        # Resources section
        self.resources_frame = self.create_resources_section()
        layout.addWidget(self.resources_frame)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_system_section(self):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        title = QLabel("SYSTEM")
        title.setFont(QFont("Arial", 9, QFont.Bold))
        title.setStyleSheet("color: #999;")
        layout.addWidget(title)
        
        # System info rows
        self.cpu_layout = self.create_info_row("CPU:", "Detecting...")
        self.gpu_layout = self.create_info_row("GPU:", "Detecting...")
        self.memory_layout = self.create_info_row("MEMORY:", "Detecting...")
        self.status_layout = self.create_info_row("STATUS:", "Ready", "#4a9d5f")
        
        layout.addLayout(self.cpu_layout)
        layout.addLayout(self.gpu_layout)
        layout.addLayout(self.memory_layout)
        layout.addLayout(self.status_layout)
        
        # Refresh button
        self.refresh_btn = QPushButton("↻ REFRESH")
        self.refresh_btn.setFont(QFont("Arial", 9))
        self.refresh_btn.setMinimumHeight(32)
        self.refresh_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #999;
                color: #666;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        self.refresh_btn.clicked.connect(self.start_detection)
        layout.addWidget(self.refresh_btn)
        
        frame.setLayout(layout)
        return frame
    
    def create_info_row(self, label_text, value_text, color="#666"):
        h_layout = QHBoxLayout()
        h_layout.setSpacing(5)
        
        label = QLabel(label_text)
        label.setFont(QFont("Arial", 9))
        label.setStyleSheet("color: #999;")
        
        value = QLabel(value_text)
        value.setFont(QFont("Arial", 9))
        value.setStyleSheet(f"color: {color};")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        value.setWordWrap(True)
        
        h_layout.addWidget(label, 0)
        h_layout.addWidget(value, 1)
        
        return h_layout
    
    def create_training_data_section(self):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        title = QLabel("TRAINING DATA")
        title.setFont(QFont("Arial", 9, QFont.Bold))
        title.setStyleSheet("color: #999;")
        layout.addWidget(title)
        
        # File drop zone
        self.drop_zone = FileDropZone(on_files_changed=self.update_file_info)
        layout.addWidget(self.drop_zone)
        
        # File info
        self.info1 = QLabel()
        self.info1.setFont(QFont("Arial", 8))
        self.info1.setStyleSheet("color: #666;")
        self.info1.setWordWrap(True)
        
        self.info1_size = QLabel()
        self.info1_size.setFont(QFont("Arial", 8))
        self.info1_size.setStyleSheet("color: #999;")
        self.info1_size.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        info1_layout = QHBoxLayout()
        info1_layout.addWidget(self.info1, 1)
        info1_layout.addWidget(self.info1_size, 0)
        
        self.info2 = QLabel()
        self.info2.setFont(QFont("Arial", 8))
        self.info2.setStyleSheet("color: #666;")
        self.info2.setWordWrap(True)
        
        self.info2_size = QLabel()
        self.info2_size.setFont(QFont("Arial", 8))
        self.info2_size.setStyleSheet("color: #999;")
        self.info2_size.setAlignment(Qt.AlignmentFlag.AlignCenter)

        info2_layout = QHBoxLayout()
        info2_layout.addWidget(self.info2, 1)
        info2_layout.addWidget(self.info2_size, 0)

        self.info3 = QLabel("files name.......")
        self.info3.setFont(QFont("Arial", 7))
        self.info3.setStyleSheet("color: #999;")
        self.info3.setWordWrap(True)
        
        layout.addLayout(info1_layout)
        layout.addLayout(info2_layout)
        layout.addWidget(self.info3)
        
        # Clear button
        self.clear_btn = QPushButton("CLEAR ALL FILES")
        self.clear_btn.setFont(QFont("Arial", 9))
        self.clear_btn.setMinimumHeight(32)
        self.clear_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_btn.setStyleSheet(""" 
               QPushButton {
                  background-color: transparent;
                   border: 1px solid #999;
                   color: #666;
                   border-radius: 3px;
               }
               QPushButton:hover {
                   background-color: #f0f0f0;
               }
           """)
        self.clear_btn.clicked.connect(self.clear_files)
        layout.addWidget(self.clear_btn)
        
        frame.setLayout(layout)
        return frame
    
    def create_resources_section(self):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        title = QLabel("RESOURCES")
        title.setFont(QFont("Arial", 9, QFont.Bold))
        title.setStyleSheet("color: #999;")
        layout.addWidget(title)
        
        # Resource rows
        self.cpu_res_label = QLabel("CPU")
        self.cpu_res_label.setFont(QFont("Arial", 8))
        self.cpu_res_label.setStyleSheet("color: #999;")
        
        self.cpu_res_value = QLabel("0%")
        self.cpu_res_value.setFont(QFont("Arial", 10))
        self.cpu_res_value.setStyleSheet("color: #666;")
        
        self.gpu_res_label = QLabel("GPU")
        self.gpu_res_label.setFont(QFont("Arial", 8))
        self.gpu_res_label.setStyleSheet("color: #999;")
        
        self.gpu_res_value = QLabel("0%")
        self.gpu_res_value.setFont(QFont("Arial", 10))
        self.gpu_res_value.setStyleSheet("color: #666;")
        
        self.mem_res_label = QLabel("MEMORY")
        self.mem_res_label.setFont(QFont("Arial", 8))
        self.mem_res_label.setStyleSheet("color: #999;")
        
        self.mem_res_value = QLabel("0%")
        self.mem_res_value.setFont(QFont("Arial", 10))
        self.mem_res_value.setStyleSheet("color: #666;")
        
        layout.addWidget(self.cpu_res_label)
        layout.addWidget(self.cpu_res_value)
        layout.addWidget(self.gpu_res_label)
        layout.addWidget(self.gpu_res_value)
        layout.addWidget(self.mem_res_label)
        layout.addWidget(self.mem_res_value)
        
        frame.setLayout(layout)
        return frame
    
    def create_right_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background-color: white;")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Neural Connections header
        header = QLabel("NEURAL CONNECTIONS")
        header.setFont(QFont("Arial", 10))
        header.setStyleSheet("color: #999; letter-spacing: 1px;")
        layout.addWidget(header)
        
        # Metrics - make responsive
        self.metrics_container = QWidget()
        self.metrics_layout = QHBoxLayout()
        self.metrics_layout.setSpacing(10)
        
        self.metrics_widgets = []
        metrics = [
            ("TRAINING PROGRESS", "14%", "#333"),
            ("ACCURACY", "12.27%", "#e74c3c"),
            ("LOSS", "0.8292", "#e74c3c"),
            ("EPOCH", "9/73", "#3498db"),
            ("PARAMETERS", "1.5K", "#2ecc71")
        ]

        for label, value, color in metrics:
            metric_widget = self.create_metric_widget(label, value, color)
            self.metrics_widgets.append(metric_widget)
            self.metrics_layout.addWidget(metric_widget)
        
        self.metrics_container.setLayout(self.metrics_layout)
        layout.addWidget(self.metrics_container)
        
        # Training controls section
        self.controls_frame = self.create_training_controls()
        layout.addWidget(self.controls_frame)
        
        # Model configuration
        self.config_frame = self.create_model_config()
        layout.addWidget(self.config_frame)
        
        # Training settings
        self.settings_frame = self.create_training_settings()
        layout.addWidget(self.settings_frame)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_metric_widget(self, label, value, color):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        label_widget = QLabel(label)
        label_widget.setFont(QFont("Arial", 7))
        label_widget.setStyleSheet("color: #999;")
        label_widget.setWordWrap(True)

        value_widget = QLabel(value)
        value_widget.setFont(QFont("Arial", 16, QFont.Bold))
        value_widget.setStyleSheet(f"color: {color};")

        # Mini chart placeholder
        chart = QFrame()
        chart.setFixedHeight(25)
        chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        chart.setStyleSheet(f"background-color: {color}20; border-radius: 3px;")

        layout.addWidget(label_widget)
        layout.addWidget(value_widget)
        layout.addWidget(chart)

        widget.setLayout(layout)
        return widget
    
    def create_training_controls(self):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        frame.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        title = QLabel("TRAINING")
        title.setFont(QFont("Arial", 9, QFont.Bold))
        title.setStyleSheet("color: #999;")
        layout.addWidget(title)
        
        # Buttons
        buttons = ["TEST GPU", "TEST CPU", "START TRAINING", "STOP", "TRAIN NEW MODEL"]
        for btn_text in buttons:
            btn = QPushButton(btn_text)
            btn.setFont(QFont("Arial", 9))
            btn.setMinimumHeight(36)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if btn_text == "START TRAINING":
                btn.clicked.connect(self.start_training_logic)
            layout.addWidget(btn)


        # Console output title with resize hint
        console_header = QHBoxLayout()
        console_title = QLabel("CONSOLE OUTPUT")
        console_title.setFont(QFont("Arial", 8, QFont.Bold))
        console_title.setStyleSheet("color: #999; margin-top: 5px;")

        resize_hint = QLabel("⋮⋮ drag to resize")
        resize_hint.setFont(QFont("Arial", 7))
        resize_hint.setStyleSheet("color: #666;")

        console_header.addWidget(console_title)
        console_header.addStretch()
        console_header.addWidget(resize_hint)
        layout.addLayout(console_header)

        # Resizable console container
        self.console_container = QFrame()
        self.console_container.setMinimumHeight(100)
        self.console_container.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)

        console_container_layout = QVBoxLayout(self.console_container)
        console_container_layout.setContentsMargins(0, 0, 0, 0)
        console_container_layout.setSpacing(0)

        # Scrollable console area
        self.console_scroll = QScrollArea()
        self.console_scroll.setWidgetResizable(True)
        self.console_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Console text label inside scroll area
        self.console_text = QLabel("Ready to start training...\nSelect a task and dataset, then click START TRAINING")
        self.console_text.setFont(QFont("Courier", 9))
        self.console_text.setStyleSheet("""
            color: #ccc; 
            background-color: #1e1e1e; 
            padding: 10px;
        """)
        self.console_text.setWordWrap(True)
        self.console_text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.console_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.console_scroll.setWidget(self.console_text)
        console_container_layout.addWidget(self.console_scroll)

        # Resize grip at bottom
        self.console_resize_grip = QFrame()
        self.console_resize_grip.setFixedHeight(8)
        self.console_resize_grip.setCursor(Qt.CursorShape.SizeVerCursor)
        self.console_resize_grip.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QFrame:hover {
                background-color: #555;
            }
        """)
        self.console_resize_grip.mousePressEvent = self.console_resize_start
        self.console_resize_grip.mouseMoveEvent = self.console_resize_move
        console_container_layout.addWidget(self.console_resize_grip)

        layout.addWidget(self.console_container, 1)  # stretch factor 1 to allow growth

        frame.setLayout(layout)
        return frame

    def console_resize_start(self, event):
        """Start resizing the console"""
        self._console_resize_start_y = event.globalPos().y()
        self._console_resize_start_height = self.console_container.height()

    def console_resize_move(self, event):
        """Handle console resize drag"""
        if hasattr(self, '_console_resize_start_y'):
            delta = event.globalPos().y() - self._console_resize_start_y
            new_height = max(100, min(600, self._console_resize_start_height + delta))
            self.console_container.setFixedHeight(new_height)

    def create_model_config(self):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        frame.setStyleSheet("background-color: #fafafa; border: 1px solid #e0e0e0; border-radius: 5px;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Create the 4 Dropdown Sections
        self.task_section, self.combo_task = self.create_combo_section("TASK", "Classification", "Select Computer Vision task")
        self.frame_section, self.combo_framework = self.create_combo_section("FRAMEWORK", "", "Select engine for the task")
        self.version_section, self.combo_version = self.create_combo_section("MODEL VERSION", "", "Select specific architecture")


        # Target column selector (for regression - hidden by default)
        self.target_section, self.combo_target = self.create_combo_section("TARGET COLUMN", "", "Select target column for regression")
        self.target_section.setVisible(False)

        # Populate Task (Level 1) with placeholder
        self.combo_task.clear()
        self.combo_task.addItem("-- Select Task --")  # Placeholder
        self.combo_task.addItems(self.MODEL_ZOO.keys())

        # Connect the "Cascade" logic
        self.combo_task.currentIndexChanged.connect(self.update_framework_menu)
        self.combo_task.currentIndexChanged.connect(self.toggle_regression_options)
        self.combo_framework.currentIndexChanged.connect(self.update_version_menu)
        self.combo_version.currentIndexChanged.connect(self.update_config_summary)

        layout.addWidget(self.task_section)
        layout.addWidget(self.frame_section)
        layout.addWidget(self.version_section)
        layout.addWidget(self.target_section)

        # Initial trigger to populate everything
        self.update_framework_menu()

        frame.setLayout(layout)
        return frame

    def toggle_regression_options(self):
        """Show/hide regression-specific options based on task selection"""
        selected_task = self.combo_task.currentText()
        is_regression = selected_task == "Regression"
        self.target_section.setVisible(is_regression)

        # Update the drop zone hint text based on task
        if is_regression:
            self.drop_zone.types.setText("Upload a CSV file with features and target column")
        elif selected_task == "Object Detection":
            self.drop_zone.types.setText("Folder with images/, labels/, and data.yaml")
        elif selected_task == "Segmentation":
            self.drop_zone.types.setText("Folder with images/ and labels/ subfolders")
        else:
            self.drop_zone.types.setText("Folder with class subfolders (e.g., cats/, dogs/)")

    def update_target_columns(self, csv_path):
        """Load CSV and populate target column dropdown"""
        try:
            df = pd.read_csv(csv_path, nrows=5)  # Read only first 5 rows for speed
            columns = list(df.columns)
            self.combo_target.clear()
            self.combo_target.addItems(columns)
            # Try to auto-select common target column names
            common_targets = ['target', 'y', 'label', 'price', 'value', 'output']
            for col in columns:
                if col.lower() in common_targets:
                    self.combo_target.setCurrentText(col)
                    break
        except Exception as e:
            print(f"Error reading CSV: {e}")

    # --- Cascading Logic Methods ---

    def update_framework_menu(self):
        """Level 2: Updates when Task changes"""
        selected_task = self.combo_task.currentText()
        self.combo_framework.blockSignals(True) # Prevent triggering Level 3 prematurely
        self.combo_framework.clear()
        
        if selected_task in self.MODEL_ZOO:
            frameworks = list(self.MODEL_ZOO[selected_task].keys())
            self.combo_framework.addItems(frameworks)
            
        self.combo_framework.blockSignals(False)
        self.update_version_menu() # Now trigger Level 3

    def update_version_menu(self):
        """Level 3: Updates when Framework changes"""
        selected_task = self.combo_task.currentText()
        selected_framework = self.combo_framework.currentText()
        self.combo_version.clear()
        
        if selected_task in self.MODEL_ZOO and selected_framework in self.MODEL_ZOO[selected_task]:
            versions_dict = self.MODEL_ZOO[selected_task][selected_framework]["versions"]
            # Add display names (keys) to combo box
            self.combo_version.addItems(versions_dict.keys())
            # Update training settings based on framework defaults
            self.update_training_settings_from_config()

    # Commented out - MODEL FORMAT export feature removed
    # def update_format_menu(self):
    #     """Level 4: Updates when Version changes"""
    #     selected_task = self.combo_task.currentText()
    #     selected_framework = self.combo_framework.currentText()
    #     self.combo_format.clear()
    #
    #     if selected_task in self.MODEL_ZOO and selected_framework in self.MODEL_ZOO[selected_task]:
    #         formats = self.MODEL_ZOO[selected_task][selected_framework]["formats"]
    #         self.combo_format.addItems(formats)
    #
    #     # Update config summary to show the new format
    #     self.update_config_summary()

    def create_combo_section(self, title, value, help_text):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 2, 0, 2) # Tighter margins
        layout.setSpacing(2)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 7, QFont.Bold)) # Smaller font for label
        title_label.setStyleSheet("color: #aaa;")
        
        combo = NoWheelComboBox()
        combo.setMinimumHeight(28) # Slightly shorter
        combo.setStyleSheet("""
            QComboBox { 
                background-color: white; border: 1px solid #ddd; 
                border-radius: 4px; padding: 4px; color: #555; 
            }
        """)
        
        layout.addWidget(title_label)
        layout.addWidget(combo)
        return widget, combo
    
    def create_training_settings(self):
        """Creates training settings frame with preset/manual options"""
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        frame.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        title = QLabel("TRAINING SETTINGS")
        title.setFont(QFont("Arial", 9, QFont.Bold))
        title.setStyleSheet("color: #999;")
        layout.addWidget(title)
        
        # Preset / Manual buttons
        btn_layout = QHBoxLayout()
        self.preset_btn = QPushButton("PRESET")
        self.preset_btn.setFont(QFont("Arial", 9))
        self.preset_btn.setMinimumHeight(32)
        self.preset_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.preset_btn.setCheckable(True)
        self.preset_btn.setChecked(True)
        self.preset_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                border: 1px solid #999;
                color: #666;
                padding: 6px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #3498db;
                color: white;
                border: 1px solid #2980b9;
            }
        """)
        self.preset_btn.clicked.connect(self.toggle_preset_mode)

        self.manual_btn = QPushButton("MANUAL")
        self.manual_btn.setFont(QFont("Arial", 9))
        self.manual_btn.setMinimumHeight(32)
        self.manual_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.manual_btn.setCheckable(True)
        self.manual_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #ccc;
                color: #999;
                padding: 6px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #3498db;
                color: white;
                border: 1px solid #2980b9;
            }
        """)
        self.manual_btn.clicked.connect(self.toggle_manual_mode)

        btn_layout.addWidget(self.preset_btn)
        btn_layout.addWidget(self.manual_btn)
        layout.addLayout(btn_layout)

        # Manual settings container (hidden by default)
        self.manual_settings_container = QWidget()
        manual_layout = QVBoxLayout(self.manual_settings_container)
        manual_layout.setContentsMargins(0, 10, 0, 0)
        manual_layout.setSpacing(8)

        # Epochs setting
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("EPOCHS:")
        epochs_label.setFont(QFont("Arial", 8))
        epochs_label.setStyleSheet("color: #999;")
        self.epochs_spinbox = NoWheelComboBox()
        self.epochs_spinbox.addItems(["10", "25", "50", "100", "150", "200", "300"])
        self.epochs_spinbox.setCurrentText("50")
        self.epochs_spinbox.setEditable(True)
        self.epochs_spinbox.setMinimumHeight(28)
        self.epochs_spinbox.setStyleSheet("QComboBox { background-color: white; border: 1px solid #ddd; border-radius: 4px; padding: 4px; }")
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_spinbox, 1)
        manual_layout.addLayout(epochs_layout)

        # Batch size setting
        batch_layout = QHBoxLayout()
        batch_label = QLabel("BATCH SIZE:")
        batch_label.setFont(QFont("Arial", 8))
        batch_label.setStyleSheet("color: #999;")
        self.batch_spinbox = NoWheelComboBox()
        self.batch_spinbox.addItems(["4", "8", "16", "32", "64", "128"])
        self.batch_spinbox.setCurrentText("16")
        self.batch_spinbox.setEditable(True)
        self.batch_spinbox.setMinimumHeight(28)
        self.batch_spinbox.setStyleSheet("QComboBox { background-color: white; border: 1px solid #ddd; border-radius: 4px; padding: 4px; }")
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_spinbox, 1)
        manual_layout.addLayout(batch_layout)

        # Image size setting
        imgsz_layout = QHBoxLayout()
        imgsz_label = QLabel("IMAGE SIZE:")
        imgsz_label.setFont(QFont("Arial", 8))
        imgsz_label.setStyleSheet("color: #999;")
        self.imgsz_spinbox = NoWheelComboBox()
        self.imgsz_spinbox.addItems(["64", "128", "224", "320", "416", "512", "640", "1024"])
        self.imgsz_spinbox.setCurrentText("224")
        self.imgsz_spinbox.setEditable(True)
        self.imgsz_spinbox.setMinimumHeight(28)
        self.imgsz_spinbox.setStyleSheet("QComboBox { background-color: white; border: 1px solid #ddd; border-radius: 4px; padding: 4px; }")
        imgsz_layout.addWidget(imgsz_label)
        imgsz_layout.addWidget(self.imgsz_spinbox, 1)
        manual_layout.addLayout(imgsz_layout)

        self.manual_settings_container.setVisible(False)
        layout.addWidget(self.manual_settings_container)

        # Speed/Quality slider (for preset mode)
        self.preset_slider_container = QWidget()
        preset_slider_layout = QVBoxLayout(self.preset_slider_container)
        preset_slider_layout.setContentsMargins(0, 0, 0, 0)
        preset_slider_layout.setSpacing(5)

        slider_layout = QHBoxLayout()
        speed_label = QLabel("SPEED")
        speed_label.setFont(QFont("Arial", 8))
        speed_label.setStyleSheet("color: #999;")

        quality_label = QLabel("QUALITY")
        quality_label.setFont(QFont("Arial", 8))
        quality_label.setStyleSheet("color: #999;")

        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setMinimum(0)
        self.quality_slider.setMaximum(100)
        self.quality_slider.setValue(70)
        self.quality_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #e0e0e0;
            }
            QSlider::handle:horizontal {
                background: #666;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)
        self.quality_slider.valueChanged.connect(self.update_quality_description)

        slider_layout.addWidget(speed_label)
        slider_layout.addWidget(self.quality_slider, 1)
        slider_layout.addWidget(quality_label)
        preset_slider_layout.addLayout(slider_layout)

        self.quality_desc = QLabel("High Quality")
        self.quality_desc.setFont(QFont("Arial", 10))
        self.quality_desc.setStyleSheet("color: #666;")

        self.quality_subdesc = QLabel("Great results, balanced time")
        self.quality_subdesc.setFont(QFont("Arial", 8))
        self.quality_subdesc.setStyleSheet("color: #999;")

        preset_slider_layout.addWidget(self.quality_desc)
        preset_slider_layout.addWidget(self.quality_subdesc)

        layout.addWidget(self.preset_slider_container)

        # Config summary
        self.config_summary = QLabel("Config: epochs=50, batch=16, imgsz=224")
        self.config_summary.setFont(QFont("Courier", 8))
        self.config_summary.setStyleSheet("color: #666; background-color: white; padding: 6px; border: 1px solid #e0e0e0; border-radius: 3px;")
        self.config_summary.setWordWrap(True)
        layout.addWidget(self.config_summary)

        # Apply button
        apply_btn = QPushButton("APPLY SETTINGS")
        apply_btn.setFont(QFont("Arial", 9))
        apply_btn.setMinimumHeight(36)
        apply_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #999;
                color: #666;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        apply_btn.clicked.connect(self.apply_training_settings)
        layout.addWidget(apply_btn)

        frame.setLayout(layout)
        return frame

    def toggle_preset_mode(self):
        """Switch to preset mode"""
        self.preset_btn.setChecked(True)
        self.manual_btn.setChecked(False)
        self.manual_settings_container.setVisible(False)
        self.preset_slider_container.setVisible(True)
        self.update_quality_description(self.quality_slider.value())

    def toggle_manual_mode(self):
        """Switch to manual mode"""
        self.manual_btn.setChecked(True)
        self.preset_btn.setChecked(False)
        self.manual_settings_container.setVisible(True)
        self.preset_slider_container.setVisible(False)
        self.update_config_summary()

    def update_quality_description(self, value):
        """Update quality description based on slider value"""
        if value < 25:
            self.quality_desc.setText("Fast Training")
            self.quality_subdesc.setText("Quick results, lower accuracy")
            epochs, batch, imgsz = 25, 32, 128
        elif value < 50:
            self.quality_desc.setText("Balanced")
            self.quality_subdesc.setText("Good balance of speed and quality")
            epochs, batch, imgsz = 50, 16, 224
        elif value < 75:
            self.quality_desc.setText("High Quality")
            self.quality_subdesc.setText("Great results, balanced time")
            epochs, batch, imgsz = 100, 16, 320
        else:
            self.quality_desc.setText("Maximum Quality")
            self.quality_subdesc.setText("Best results, longer training time")
            epochs, batch, imgsz = 200, 8, 640

        # Update UI if in preset mode
        if self.preset_btn.isChecked():
            self.epochs_spinbox.setCurrentText(str(epochs))
            self.batch_spinbox.setCurrentText(str(batch))
            self.imgsz_spinbox.setCurrentText(str(imgsz))

        self.update_config_summary()

    def update_config_summary(self):
        """Update the config summary label"""
        try:
            epochs = int(self.epochs_spinbox.currentText())
            batch = int(self.batch_spinbox.currentText())
            imgsz = int(self.imgsz_spinbox.currentText())

            selected_task = self.combo_task.currentText()
            selected_framework = self.combo_framework.currentText()
            selected_version = self.combo_version.currentText()

            model_name = "N/A"
            if selected_task in self.MODEL_ZOO and selected_framework in self.MODEL_ZOO[selected_task]:
                versions_dict = self.MODEL_ZOO[selected_task][selected_framework]["versions"]
                if selected_version in versions_dict:
                    model_name = versions_dict[selected_version]

            self.config_summary.setText(f"Model: {model_name}\nEpochs: {epochs}, Batch: {batch}, ImgSize: {imgsz}")
        except:
            self.config_summary.setText("Config: Invalid values")

    def update_training_settings_from_config(self):
        """Update training settings based on selected framework's default config"""
        selected_task = self.combo_task.currentText()
        selected_framework = self.combo_framework.currentText()

        if selected_task in self.MODEL_ZOO and selected_framework in self.MODEL_ZOO[selected_task]:
            default_config = self.MODEL_ZOO[selected_task][selected_framework].get("default_config", {})

            if "epochs" in default_config:
                self.epochs_spinbox.setCurrentText(str(default_config["epochs"]))
            if "batch" in default_config:
                self.batch_spinbox.setCurrentText(str(default_config["batch"]))
            if "imgsz" in default_config:
                self.imgsz_spinbox.setCurrentText(str(default_config["imgsz"]))

        self.update_config_summary()

    def apply_training_settings(self):
        """Apply training settings and show confirmation"""
        self.update_config_summary()
        QMessageBox.information(self, "Settings Applied",
            f"Training settings applied:\n\n"
            f"Epochs: {self.epochs_spinbox.currentText()}\n"
            f"Batch Size: {self.batch_spinbox.currentText()}\n"
            f"Image Size: {self.imgsz_spinbox.currentText()}")

    def get_current_training_config(self):
        """Get current training configuration from UI settings"""
        selected_task = self.combo_task.currentText()
        selected_framework = self.combo_framework.currentText()
        selected_version = self.combo_version.currentText()

        # Get model file name from versions dictionary
        # Set default based on task and framework
        if selected_task == "Regression":
            model_name = selected_framework.lower()
            project = "runs/regression"
        elif selected_task == "Object Detection":
            model_name = "yolo11n-obb.pt"  # Default for Object Detection
            project = "runs/detect"
        elif selected_task == "Segmentation":
            model_name = "yolo11n-seg.pt"  # Default for Segmentation
            project = "runs/segment"
        elif selected_framework == "ResNet":
            model_name = "resnet18"  # Default for ResNet
            project = "runs/classify"
        else:
            model_name = "yolov8n-cls.pt"  # Default for YOLO Classification
            project = "runs/classify"

        if selected_task in self.MODEL_ZOO and selected_framework in self.MODEL_ZOO[selected_task]:
            versions_dict = self.MODEL_ZOO[selected_task][selected_framework]["versions"]
            if selected_version in versions_dict:
                model_name = versions_dict[selected_version]

        try:
            epochs = int(self.epochs_spinbox.currentText())
            batch = int(self.batch_spinbox.currentText())
            imgsz = int(self.imgsz_spinbox.currentText())
        except ValueError:
            # Fallback defaults based on task
            if selected_task == "Object Detection":
                epochs, batch, imgsz = 100, 16, 640
            elif selected_task == "Segmentation":
                epochs, batch, imgsz = 100, 16, 640
            elif selected_task == "Regression":
                epochs, batch, imgsz = 100, 32, 0  # imgsz not used for regression
            else:
                epochs, batch, imgsz = 50, 16, 224

        # Set model name for output directory
        name_suffix = model_name.replace(".pt", "").replace("-cls", "").replace("-obb", "").replace("-seg", "")

        config = {
            "model": model_name,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": project,
            "name": f"{name_suffix}_trained",
        }

        # Add regression-specific config
        if selected_task == "Regression":
            config["target_column"] = self.combo_target.currentText()
            config["n_estimators"] = epochs  # Use epochs as n_estimators for tree-based models
            config["learning_rate"] = 0.1
            config["max_depth"] = 6
            config["batch_size"] = batch

            # Adjust config based on model version
            if "light" in model_name.lower():
                config["n_estimators"] = 50
                config["max_depth"] = 4
            elif "deep" in model_name.lower() or "heavy" in model_name.lower():
                config["n_estimators"] = 200
                config["max_depth"] = 10
            elif "small" in model_name.lower():
                config["hidden_dims"] = [64, 32]
            elif "medium" in model_name.lower():
                config["hidden_dims"] = [128, 64, 32]
            elif "large" in model_name.lower():
                config["hidden_dims"] = [256, 128, 64, 32]

        return config

    def start_detection(self):

        self.cpu_layout.itemAt(1).widget().setText(
            f"{psutil.cpu_count()} Cores Detected"
        )

        self.memory_layout.itemAt(1).widget().setText(
            f"{round(psutil.virtual_memory().total / (1024 ** 3), 1)} GB Total"
        )

        self.gpu_layout.itemAt(1).widget().setText("Detecting GPU...")

        self.gpu_worker = GPUDetectWorker()
        self.gpu_worker.gpu_ready.connect(
            lambda txt: self.gpu_layout.itemAt(1).widget().setText(txt)
        )
        self.gpu_worker.start()

        if hasattr(self, "stats_timer"):
            self.stats_timer.stop()
            self.stats_timer.deleteLater()

        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_live_stats)
        self.stats_timer.start(2000)

    def update_live_stats(self):
        """Updates the Resources section with real usage data"""
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().percent

        # Update the Resource labels
        self.cpu_res_value.setText(f"{cpu_usage}%")
        self.mem_res_value.setText(f"{ram_usage}%")

        # Simple color warning logic
        if cpu_usage > 80:
            self.cpu_res_value.setStyleSheet("color: #e74c3c; font-weight: bold;")
        else:
            self.cpu_res_value.setStyleSheet("color: #666;")

    def start_training_logic(self):
        """Starts the actual training process based on selected task."""
        # Check which task is selected
        selected_task = self.combo_task.currentText()
        selected_framework = self.combo_framework.currentText()

        # Check if user has selected a task
        if selected_task == "-- Select Task --" or not selected_task:
            QMessageBox.warning(self, "No Task Selected",
                "Please select a task from the TASK dropdown.\n\n"
                "Available tasks:\n"
                "• Image Classification\n"
                "• Object Detection\n"
                "• Segmentation\n"
                "• Regression")
            return

        # Check if task is supported
        supported_tasks = ["Image Classification", "Object Detection", "Segmentation", "Regression"]
        if selected_task not in supported_tasks:
            QMessageBox.warning(self, "Task Not Supported",
                f"'{selected_task}' is not yet implemented.\n\n"
                "Currently supported tasks:\n"
                "• Image Classification (YOLO-Cls, ResNet)\n"
                "• Object Detection (YOLO-OBB, RT-DETR)\n"
                "• Segmentation (YOLO-Seg)\n"
                "• Regression (XGBoost, LightGBM, RandomForest, NeuralNetwork)\n"
                "Please select a supported task from the dropdown.")
            return

        # Check if framework is supported for the selected task
        if selected_task == "Image Classification":
            supported_frameworks = ["YOLO-Cls", "ResNet"]
        elif selected_task == "Object Detection":
            supported_frameworks = ["YOLO-OBB", "RT-DETR"]
        elif selected_task == "Segmentation":
            supported_frameworks = ["YOLO-Seg"]
        elif selected_task == "Regression":
            supported_frameworks = ["XGBoost", "LightGBM", "RandomForest", "NeuralNetwork"]
        else:
            supported_frameworks = []

        if selected_framework not in supported_frameworks:
            QMessageBox.warning(self, "Framework Not Supported",
                f"'{selected_framework}' is not yet implemented for {selected_task}.\n\n"
                f"Supported frameworks for {selected_task}:\n"
                + "\n".join(f"• {fw}" for fw in supported_frameworks))
            return

        # Check if model version is selected
        selected_version = self.combo_version.currentText()
        if not selected_version:
            QMessageBox.warning(self, "No Model Version Selected",
                "Please select a model version from the MODEL VERSION dropdown.")
            return

        # Check if dataset is selected
        if not self.drop_zone.dataset_path:
            if selected_task == "Regression":
                QMessageBox.warning(self, "No Dataset",
                    "Please select a CSV file first.\n\n"
                    "CSV file should contain:\n"
                    "• Feature columns (numeric or categorical)\n"
                    "• One target column for prediction\n\n"
                    "Click the upload area and select 'CSV File (Regression)'")
            elif selected_task == "Object Detection":
                QMessageBox.warning(self, "No Dataset",
                    "Please select a dataset folder first.\n\n"
                    "Dataset structure for Object Detection:\n"
                    "dataset/\n"
                    "├── images/\n"
                    "│   ├── train/\n"
                    "│   └── val/\n"
                    "├── labels/\n"
                    "│   ├── train/\n"
                    "│   └── val/\n"
                    "└── data.yaml")
            elif selected_task == "Segmentation":
                QMessageBox.warning(self, "No Dataset",
                    "Please select a dataset folder first.\n\n"
                    "Dataset structure for Segmentation:\n"
                    "dataset/\n"
                    "├── images/\n"
                    "│   ├── train/\n"
                    "│   └── val/\n"
                    "└── labels/\n"
                    "    ├── train/\n"
                    "    └── val/\n\n"
                    "Labels format: class_id x1 y1 x2 y2 ... xn yn\n"
                    "(No data.yaml needed - auto-generated!)")
            else:
                QMessageBox.warning(self, "No Dataset",
                    "Please select a dataset folder first.\n\n"
                    "Dataset structure should be:\n"
                    "dataset/\n"
                    "├── class1/ (images)\n"
                    "├── class2/ (images)\n"
                    "└── class3/ (images)\n\n"
                    "We will automatically split into train/val/test!")
            return

        # Validate dataset structure based on task
        dataset_path = self.drop_zone.dataset_path

        # For regression, validate CSV file
        if selected_task == "Regression":
            if not dataset_path.lower().endswith('.csv'):
                QMessageBox.warning(self, "Invalid File",
                    "Please select a CSV file for regression.\n\n"
                    "Click the upload area and select 'CSV File (Regression)'")
                return

            # Check if target column is selected
            target_column = self.combo_target.currentText()
            if not target_column:
                QMessageBox.warning(self, "No Target Column",
                    "Please select a target column for regression.\n\n"
                    "The target column dropdown should show available columns from your CSV.")
                return

            # Validate CSV can be read
            try:
                df = pd.read_csv(dataset_path, nrows=10)
                if target_column not in df.columns:
                    QMessageBox.warning(self, "Invalid Target Column",
                        f"Target column '{target_column}' not found in CSV.\n\n"
                        f"Available columns: {', '.join(df.columns)}")
                    return
                is_valid = True
                message = "CSV validated successfully"
                classes = list(df.columns)
            except Exception as e:
                QMessageBox.warning(self, "CSV Error", f"Error reading CSV: {str(e)}")
                return
        elif selected_task == "Object Detection":
            is_valid, message, classes = det_validate_dataset_structure(dataset_path)
        elif selected_task == "Segmentation":
            is_valid, message, classes = seg_validate_dataset_structure(dataset_path)
        else:
            is_valid, message, classes = cls_validate_dataset_structure(dataset_path)

        if not is_valid:
            QMessageBox.warning(self, "Invalid Dataset", message)
            return

        self.update_console(f"[{self.get_timestamp()}] Task: {selected_task}")
        self.update_console(f"[{self.get_timestamp()}] Framework: {selected_framework}")
        self.update_console(f"[{self.get_timestamp()}] Checking dataset structure...")

        # For Image Classification, check if we need to split
        from pathlib import Path
        if selected_task == "Image Classification":
            train_exists = (Path(dataset_path) / "train").exists()
            val_exists = (Path(dataset_path) / "val").exists()

            if not (train_exists and val_exists):
                self.update_console(f"[{self.get_timestamp()}] Auto-splitting dataset (70% train, 20% val, 10% test)...")
                dataset_path = split_dataset(dataset_path)
                self.update_console(f"[{self.get_timestamp()}] Dataset split complete!")
        elif selected_task == "Object Detection":
            # For object detection, prepare/update data.yaml for the detected format
            self.update_console(f"[{self.get_timestamp()}] Preparing data.yaml for detected format...")
            yaml_path = prepare_data_yaml(dataset_path)
            dataset_path = yaml_path
            self.update_console(f"[{self.get_timestamp()}] Using data.yaml: {yaml_path}")
        elif selected_task == "Segmentation":
            # For segmentation, prepare/update data.yaml for the detected format
            self.update_console(f"[{self.get_timestamp()}] Preparing data.yaml for segmentation...")
            yaml_path = seg_prepare_data_yaml(dataset_path)
            dataset_path = yaml_path
            self.update_console(f"[{self.get_timestamp()}] Using data.yaml: {yaml_path}")
        elif selected_task == "Regression":
            # For regression, just log the target column
            target_col = self.combo_target.currentText()
            self.update_console(f"[{self.get_timestamp()}] Target column: {target_col}")
            self.update_console(f"[{self.get_timestamp()}] CSV file validated successfully")

        # Get config dynamically from UI settings
        config = self.get_current_training_config()

        # Update UI status
        self.status_layout.itemAt(1).widget().setText("Training...")
        self.status_layout.itemAt(1).widget().setStyleSheet("color: #3498db;")
        self.training_val = 0

        # Log to console
        self.update_console(f"[{self.get_timestamp()}] Starting {selected_framework} training...")
        self.update_console(f"[{self.get_timestamp()}] Model: {config['model']}")

        # Log appropriate config based on task
        if selected_task == "Regression":
            if 'n_estimators' in config:
                self.update_console(f"[{self.get_timestamp()}] Estimators: {config.get('n_estimators', 100)}, Batch: {config['batch']}")
            if 'epochs' in config and selected_framework == "NeuralNetwork":
                self.update_console(f"[{self.get_timestamp()}] Epochs: {config['epochs']}, Batch: {config['batch']}")
        else:
            self.update_console(f"[{self.get_timestamp()}] Epochs: {config['epochs']}, Batch: {config['batch']}, ImgSize: {config['imgsz']}")

        self.update_console(f"[{self.get_timestamp()}] Dataset: {dataset_path}")
        if classes and selected_task != "Regression":
            self.update_console(f"[{self.get_timestamp()}] Classes: {', '.join(classes)}")
        elif selected_task == "Regression" and classes:
            self.update_console(f"[{self.get_timestamp()}] Features: {len(classes) - 1} columns")

        # Start training in background thread
        self.training_worker = TrainingWorker(dataset_path, config, selected_framework, selected_task)
        self.training_worker.log_message.connect(self.update_console)
        self.training_worker.training_finished.connect(self.on_training_finished)
        self.training_worker.start()

        # Start progress animation
        self.train_timer = QTimer()
        self.train_timer.timeout.connect(self.advance_progress)
        self.train_timer.start(500)

    def get_timestamp(self):
        """Returns current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%I:%M:%S %p")

    def update_console(self, message):
        """Updates the console text in training controls."""
        import re

        # Get current text
        current_text = self.console_text.text()

        # Keep last 100 lines for scrollable history
        lines = current_text.split('\n')[-99:]

        # Clean up message - remove ANSI codes if any
        clean_msg = re.sub(r'\x1b\[[0-9;]*m', '', message)

        # Add timestamp if not present
        if not clean_msg.startswith('['):
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            clean_msg = f"[{timestamp}] {clean_msg}"

        lines.append(clean_msg)
        self.console_text.setText('\n'.join(lines))

        # Auto-scroll to bottom
        QTimer.singleShot(10, lambda: self.console_scroll.verticalScrollBar().setValue(
            self.console_scroll.verticalScrollBar().maximum()
        ))

    def on_training_finished(self, model_path, success):
        """Called when training completes."""
        self.train_timer.stop()

        if success:
            self.status_layout.itemAt(1).widget().setText("Finished")
            self.status_layout.itemAt(1).widget().setStyleSheet("color: #4a9d5f;")
            self.metrics_widgets[0].findChildren(QLabel)[1].setText("100%")
            self.update_console(f"[{self.get_timestamp()}] ✓ Model saved: {model_path}")
            QMessageBox.information(self, "Training Complete",
                f"Model trained successfully!\n\nSaved at:\n{model_path}")
        else:
            self.status_layout.itemAt(1).widget().setText("Failed")
            self.status_layout.itemAt(1).widget().setStyleSheet("color: #e74c3c;")
            QMessageBox.warning(self, "Training Failed",
                "Training failed. Check console for details.")

    def advance_progress(self):
        """Animates the progress bar and metrics"""
        if self.training_val < 100:
            self.training_val += 1
            # Update the Progress Metric (the first card)
            self.metrics_widgets[0].findChildren(QLabel)[1].setText(f"{self.training_val}%")

            # Simulate shifting accuracy/loss
            new_acc = 12.27 + (self.training_val * 0.7)
            self.metrics_widgets[1].findChildren(QLabel)[1].setText(f"{new_acc:.2f}%")
        else:
            self.train_timer.stop()
            self.status_layout.itemAt(1).widget().setText("Finished")
            self.status_layout.itemAt(1).widget().setStyleSheet("color: #4a9d5f;")

    def clear_files(self):
        """Clears all files from the drop zone and resets file info labels"""
        self.drop_zone.clear()
        self.update_file_info([])
    
    def update_file_info(self, files):
        """Dynamically updates file info labels based on selected files"""
        import os
        
        if not files:
            self.info1.setText("No files loaded")
            self.info1_size.setText("0 MB")
            self.info2.setText("")
            self.info2_size.setText("")
            self.info3.setText("")
            return
        
        # Check if it's a CSV file (for regression)
        if len(files) == 1 and files[0].lower().endswith('.csv'):
            csv_file = files[0]
            try:
                df = pd.read_csv(csv_file, nrows=100)
                file_size = os.path.getsize(csv_file) / (1024 * 1024)
                self.info1.setText(f"CSV: {df.shape[0]}+ rows, {df.shape[1]} columns")
                self.info1_size.setText(f"{file_size:.2f} MB")
                self.info2.setText(f"Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                self.info2_size.setText("")
                self.info3.setText("Ready for regression training")
                # Update target column dropdown
                self.update_target_columns(csv_file)
            except Exception as e:
                self.info1.setText(f"Error reading CSV: {str(e)[:30]}")
                self.info1_size.setText("")
            return

        # Check if it's a folder (existing logic)
        if len(files) == 1 and os.path.isdir(files[0]):
            folder = files[0]
            # Count files in folder
            total_files = 0
            total_size = 0
            for root, dirs, filenames in os.walk(folder):
                for f in filenames:
                    total_files += 1
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except:
                        pass
            self.info1.setText(f"Folder: {total_files} files")
            self.info1_size.setText(f"{total_size / (1024 * 1024):.2f} MB")
            self.info2.setText(f"Path: {os.path.basename(folder)}")
            self.info2_size.setText("")
            self.info3.setText("Ready for training")
            return

        # Categorize files (original logic for individual files)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
        text_files = [f for f in files if f.lower().endswith(('.txt', '.csv', '.json', '.xml', '.yaml', '.yml'))]
        unsupported = [f for f in files if f not in image_files and f not in text_files]
        
        # Calculate total sizes
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        total_size_mb = total_size / (1024 * 1024)
        
        # Update info1 (images)
        if image_files:
            self.info1.setText(f"{len(image_files)} image files (computer vision data)")
            img_size = sum(os.path.getsize(f) for f in image_files if os.path.isfile(f)) / (1024 * 1024)
            self.info1_size.setText(f"{img_size:.2f} MB")
        else:
            self.info1.setText("No image files")
            self.info1_size.setText("0 MB")
        
        # Update info2 (text)
        if text_files:
            self.info2.setText(f"{len(text_files)} text files (NLP data)")
            text_size = sum(os.path.getsize(f) for f in text_files if os.path.isfile(f)) / (1024 * 1024)
            self.info2_size.setText(f"{text_size:.2f} MB")
        else:
            self.info2.setText("No text files")
            self.info2_size.setText("0 MB")
        
        # Update info3 (unsupported)
        if unsupported:
            unsupported_names = ", ".join(os.path.basename(f) for f in unsupported[:3])
            if len(unsupported) > 3:
                unsupported_names += "..."
            self.info3.setText(f"{len(unsupported)} unsupported files: {unsupported_names}")
        else:
            self.info3.setText("All files supported")
    
    def adjust_layout(self):
        width = self.width()
        
        # Adjust font sizes based on width
        if width < 800:
            self.header_title.setFont(QFont("Arial", 12, QFont.Bold))
            self.header_subtitle.setFont(QFont("Arial", 7))
        elif width < 1000:
            self.header_title.setFont(QFont("Arial", 14, QFont.Bold))
            self.header_subtitle.setFont(QFont("Arial", 8))
        else:
            self.header_title.setFont(QFont("Arial", 16, QFont.Bold))
            self.header_subtitle.setFont(QFont("Arial", 9))
        
        # Switch to vertical layout on small screens
        if width < 1000 and self.is_horizontal:
            self.is_horizontal = False
            # Clear current layout
            while self.main_layout.count():
                item = self.main_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            
            # Add vertical layout
            self.main_layout = QVBoxLayout()
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.main_layout.setSpacing(0)
            self.main_layout.addWidget(self.left_scroll)
            self.main_layout.addWidget(self.right_scroll)
            self.setLayout(self.main_layout)
        
        elif width >= 1000 and not self.is_horizontal:
            self.is_horizontal = True
            # Clear current layout
            while self.main_layout.count():
                item = self.main_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            
            # Add horizontal layout
            self.main_layout = QHBoxLayout()
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.main_layout.setSpacing(0)
            self.main_layout.addWidget(self.left_scroll, 1)
            self.main_layout.addWidget(self.right_scroll, 2)
            self.setLayout(self.main_layout)

    

class UniTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Uni Trainer")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(600, 500)
        
        # Create stacked widget for screens
        self.stacked_widget = QStackedWidget()
        
        # Add screens
        welcome = WelcomeScreen(self.stacked_widget)
        main_screen = MainScreen(self.stacked_widget)
        
        self.stacked_widget.addWidget(welcome)
        self.stacked_widget.addWidget(main_screen)
        
        self.setCentralWidget(self.stacked_widget)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = UniTrainerApp()
    window.show()
    sys.exit(app.exec_())