from ultralytics import RTDETR
import torch
from pathlib import Path
import yaml



class RTDETRDetector:
    """RT-DETR (Real-Time Detection Transformer) Object Detection class for training and inference."""

    def __init__(self, model_name="rtdetr-l.pt"):
        """
        Initialize the RT-DETR detector.

        Args:
            model_name: Pre-trained RT-DETR model to use. Options:
                - 'rtdetr-l.pt' (large - balanced)
                - 'rtdetr-x.pt' (extra large - most accurate)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model_name = model_name

        # Load RT-DETR model (Ultralytics auto-downloads if not found, like YOLO)
        try:
            self.model = RTDETR(model_name)
        except Exception as e:
            print(f"Error loading RT-DETR model: {e}")
            raise

    def _get_unique_run_name(self, project, base_name):
        """Return a unique run name by incrementing if needed."""
        project_path = Path(project)
        run_name = base_name
        i = 1
        while (project_path / run_name).exists():
            i += 1
            run_name = f"{base_name}{i}"
        return run_name

    def train(self, data_path, epochs=100, imgsz=640, batch=16,
              project="runs/detect", name="rtdetr_trained", save_period=10):
        """
        Train the RT-DETR object detection model.

        Args:
            data_path: Path to dataset YAML file. Structure should be:
                       dataset/
                       ├── images/
                       │   ├── train/
                       │   │   ├── img1.jpg
                       │   │   └── img2.jpg
                       │   └── val/
                       │       ├── img1.jpg
                       │       └── img2.jpg
                       ├── labels/
                       │   ├── train/
                       │   │   ├── img1.txt  (YOLO format: class x_center y_center width height)
                       │   │   └── img2.txt
                       │   └── val/
                       │       ├── img1.txt
                       │       └── img2.txt
                       └── data.yaml
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Project directory for saving results
            name: Name of the training run
            save_period: Save checkpoint every N epochs

        Returns:
            Training results
        """
        # Ensure unique output folder for each run
        name = self._get_unique_run_name(project, name)

        print(f"\n{'='*50}")
        print(f"Starting RT-DETR Object Detection Training")
        print(f"{'='*50}")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {data_path}")
        print(f"Epochs: {epochs}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch}")
        print(f"Device: {self.device}")
        print(f"Output: {project}/{name}")
        print(f"{'='*50}\n")

        # Train the model (workers=0 to avoid Windows multiprocessing issues)
        results = self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name,
            save_period=save_period,
            patience=50,  # Early stopping patience
            pretrained=True,
            optimizer='AdamW',  # RT-DETR works best with AdamW
            verbose=True,
            seed=42,
            deterministic=True,
            plots=True,  # Generate training plots
            workers=0,  # Disable multiprocessing for Windows compatibility
        )

        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best model saved at: {project}/{name}/weights/best.pt")
        print(f"Last model saved at: {project}/{name}/weights/last.pt")
        print(f"{'='*50}\n")

        return results, name

    def predict(self, image_path, conf=0.25, iou=0.45, save=True):
        """
        Run inference on an image.

        Args:
            image_path: Path to image or directory of images
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Save results to disk

        Returns:
            Prediction results
        """
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=True,
            save=save
        )

        for result in results:
            boxes = result.boxes
            print(f"\nImage: {result.path}")
            print(f"Detected {len(boxes)} objects:")

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                confidence = box.conf[0].item()
                xyxy = box.xyxy[0].tolist()
                print(f"  {i+1}. {cls_name}: {confidence*100:.2f}% at [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

        return results

    def load_trained_model(self, model_path):
        """
        Load a trained model from .pt file.

        Args:
            model_path: Path to the trained model .pt file
        """
        self.model = RTDETR(model_path)
        print(f"Loaded model from: {model_path}")

    def validate(self, data_path):
        """
        Validate the model on a dataset.

        Args:
            data_path: Path to validation dataset YAML

        Returns:
            Validation metrics
        """
        results = self.model.val(data=data_path, device=self.device)
        return results

    def export(self, format='onnx'):
        """
        Export the model to different formats.

        Args:
            format: Export format ('onnx', 'torchscript', 'openvino', etc.)

        Returns:
            Path to exported model
        """
        return self.model.export(format=format)


def create_sample_dataset_structure(base_path="dataset_detect"):
    """
    Create sample dataset folder structure for object detection.

    Args:
        base_path: Base path for the dataset
    """
    folders = [
        f"{base_path}/images/train",
        f"{base_path}/images/val",
        f"{base_path}/labels/train",
        f"{base_path}/labels/val",
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Create sample data.yaml
    data_yaml = {
        'path': str(Path(base_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'class1',
            1: 'class2',
            # Add more classes as needed
        }
    }

    yaml_path = Path(base_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Created dataset structure at: {base_path}")
    print(f"Data YAML created at: {yaml_path}")
    print("\nLabel format (YOLO format - one .txt per image):")
    print("  class_id x_center y_center width height")
    print("  (all values normalized 0-1)")
    print("\nExample label file content:")
    print("  0 0.5 0.5 0.2 0.3")
    print("  1 0.3 0.7 0.1 0.15")


# Available RT-DETR models
RTDETR_MODELS = {
    'rtdetr-l': 'rtdetr-l.pt',   # Large model - balanced speed/accuracy
    'rtdetr-x': 'rtdetr-x.pt',   # Extra large - highest accuracy
}

# Default configuration for RT-DETR training
DEFAULT_CONFIG = {
    "model": "rtdetr-l.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "project": "runs/detect",
    "name": "rtdetr_trained",
}


def train_with_user_data(data_path, config=None, progress_callback=None):
    """
    Train RT-DETR detector with user-uploaded data.

    Args:
        data_path: Path to user's dataset folder or YAML file (from UI upload)
        config: Optional config dict to override defaults
        progress_callback: Optional callback function(epoch, total_epochs, metrics)

    Returns:
        tuple: (Path to trained model, training results)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Initialize detector
    detector = RTDETRDetector(model_name=config.get("model", "rtdetr-l.pt"))

    # Train
    results, actual_name = detector.train(
        data_path=data_path,
        epochs=config.get("epochs", 100),
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 8),
        project=config.get("project", "runs/detect"),
        name=config.get("name", "rtdetr_trained")
    )

    # Path to best PyTorch model (use actual_name which may have been incremented)
    model_path = f"{config.get('project', 'runs/detect')}/{actual_name}/weights/best.pt"

    return model_path, results


def predict_with_model(model_path, image_path, conf=0.25):
    """
    Run prediction using trained RT-DETR model.

    Args:
        model_path: Path to trained .pt model
        image_path: Path to image for prediction
        conf: Confidence threshold

    Returns:
        Prediction results
    """
    detector = RTDETRDetector()
    detector.load_trained_model(model_path)
    return detector.predict(image_path, conf=conf)


def get_available_models():
    """Return list of available RT-DETR models."""
    return list(RTDETR_MODELS.keys())


if __name__ == "__main__":
    # Example usage
    print("RT-DETR Object Detection Module")
    print("=" * 40)
    print("\nAvailable models:")
    for name, path in RTDETR_MODELS.items():
        print(f"  - {name}: {path}")

    print("\nExample usage:")
    print("  from object_detection.rt_detr_obj import RTDETRDetector")
    print("  detector = RTDETRDetector('rtdetr-l.pt')")
    print("  detector.train('path/to/data.yaml', epochs=100)")
    print("  results = detector.predict('path/to/image.jpg')")
