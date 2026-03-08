# Uni Trainer 💻

**Uni Trainer** is a comprehensive, responsive AI Training Desktop Application built with PyQt5. It provides a seamless drag-and-drop graphical interface for training state-of-the-art Computer Vision and Machine Learning models without writing a single line of code.

## ✨ Features

* **Multi-Task Support:** Train models for Image Classification, Object Detection, Segmentation, and Tabular Regression.
* **Extensive Model Zoo:** * *Classification:* YOLO-Cls, ResNet (18 to 152)
  * *Detection:* YOLO-OBB, RT-DETR
  * *Segmentation:* YOLO-Seg, SAM / SAM 2
  * *Regression:* XGBoost, LightGBM, Random Forest, Neural Networks
* **Hardware Auto-Detection:** Automatically detects your CPU, RAM, and GPU (`torch.cuda`) to monitor resource usage during training.
* **Smart Data Handling:** Drag-and-drop interface for folders and CSVs. Auto-splits image datasets and auto-generates `data.yaml` files.
* **Real-Time Console:** View training progress, metrics, and logs directly in the app via a background processing thread (keeps the UI from freezing).
* **Presets & Manual Tuning:** Choose between quick speed/quality presets or manually tune Epochs, Batch Size, and Image Size.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Divyajeet01/uni-trainer.git](https://github.com/YOUR_USERNAME/uni-trainer.git)
   cd uni-trainer
