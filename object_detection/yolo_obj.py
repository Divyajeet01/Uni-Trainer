from ultralytics import YOLO
import torch
from pathlib import Path
import yaml


class YOLODetector:
    """YOLO-based Object Detection (OBB - Oriented Bounding Box) class for training and inference."""

    def __init__(self, model_name="yolo11n-obb.pt"):
        """
        Initialize the YOLO OBB detector.

        Args:
            model_name: Pre-trained OBB model to use. Options:
                - 'yolo11n-obb.pt' (nano - fastest, smallest)
                - 'yolo11s-obb.pt' (small)
                - 'yolo11m-obb.pt' (medium)
                - 'yolo11l-obb.pt' (large)
                - 'yolo11x-obb.pt' (extra large - most accurate)
                - 'yolov8n-obb.pt' (nano)
                - 'yolov8s-obb.pt' (small)
                - 'yolov8m-obb.pt' (medium)
                - 'yolov8l-obb.pt' (large)
                - 'yolov8x-obb.pt' (extra large)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load pre-trained YOLO OBB detection model
        self.model = YOLO(model_name)
        self.model_name = model_name

    def _get_unique_run_name(self, project, base_name):
        """Return a unique run name by incrementing if needed (like YOLO default behavior)."""
        project_path = Path(project)
        run_name = base_name
        i = 1
        while (project_path / run_name).exists():
            i += 1
            run_name = f"{base_name}{i}"
        return run_name

    def train(self, data_path, epochs=100, imgsz=640, batch=16,
              project="runs/detect", name="train", save_period=10):
        """
        Train the YOLO object detection model.

        Args:
            data_path: Path to dataset YAML file or folder. Structure should be:
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
                       │   │   ├── img1.txt
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
        print(f"Starting YOLO Object Detection Training")
        print(f"{'='*50}")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {data_path}")
        print(f"Epochs: {epochs}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch}")
        print(f"Device: {self.device}")
        print(f"Output: {project}/{name}")
        print(f"{'='*50}\n")

        # Train the model
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
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            plots=True,  # Generate training plots
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
        self.model = YOLO(model_path)
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
    yaml_content = {
        'path': base_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'class1',
            1: 'class2',
        }
    }

    yaml_path = Path(base_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\nCreated dataset structure at: {base_path}/")
    print("Structure:")
    print(f"  {base_path}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/  (add training images here)")
    print(f"  │   └── val/    (add validation images here)")
    print(f"  ├── labels/")
    print(f"  │   ├── train/  (add training labels here)")
    print(f"  │   └── val/    (add validation labels here)")
    print(f"  └── data.yaml   (dataset configuration)")
    print("\nLabel format (YOLO): class_id x_center y_center width height")
    print("All values normalized to [0, 1]")


# ============================================================
# CONFIGURATION - DEFAULT VALUES
# ============================================================

DEFAULT_CONFIG = {
    "model": "yolo11n-obb.pt",  # Model: yolo11n/s/m/l/x-obb.pt or yolov8n/s/m/l/x-obb.pt
    "epochs": 100,              # Number of training epochs
    "imgsz": 640,               # Image size
    "batch": 16,                # Batch size
    "project": "runs/detect",   # Output directory
    "name": "trained_model",    # Model name
}


def train_with_user_data(data_path, config=None, progress_callback=None):
    """
    Train YOLO detector with user-uploaded data.

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
    detector = YOLODetector(model_name=config["model"])

    # Train
    results, actual_name = detector.train(
        data_path=data_path,
        epochs=config["epochs"],
        imgsz=config.get("imgsz", 640),
        batch=config["batch"],
        project=config["project"],
        name=config["name"]
    )

    # Path to best PyTorch model (use actual_name which may have been incremented)
    model_path = f"{config['project']}/{actual_name}/weights/best.pt"

    return model_path, results


def predict_with_model(model_path, image_path, conf=0.25):
    """
    Run prediction using trained model.

    Args:
        model_path: Path to trained .pt model
        image_path: Path to image for prediction
        conf: Confidence threshold

    Returns:
        Prediction results
    """
    detector = YOLODetector()
    detector.load_trained_model(model_path)
    return detector.predict(image_path, conf=conf)


def get_available_models():
    """Returns list of available YOLO OBB object detection models."""
    return [
        "yolo11n-obb.pt",
        "yolo11s-obb.pt",
        "yolo11m-obb.pt",
        "yolo11l-obb.pt",
        "yolo11x-obb.pt",
        "yolov8n-obb.pt",
        "yolov8s-obb.pt",
        "yolov8m-obb.pt",
        "yolov8l-obb.pt",
        "yolov8x-obb.pt",
    ]


def validate_obb_label_format(label_path):
    """
    Validate that a label file is in OBB format (9 values per line).

    OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    Standard format: class_id x_center y_center width height (5 values)

    Args:
        label_path: Path to a label .txt file

    Returns:
        tuple: (is_obb, message)
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            return True, "Empty label file"

        for line in lines:
            values = line.strip().split()
            if len(values) == 0:
                continue

            num_values = len(values)

            if num_values == 5:
                return False, f"Label has 5 values (standard YOLO format). OBB requires 9 values.\nFormat: class_id x1 y1 x2 y2 x3 y3 x4 y4"
            elif num_values == 9:
                return True, "Valid OBB format"
            else:
                return False, f"Label has {num_values} values. OBB requires 9 values.\nFormat: class_id x1 y1 x2 y2 x3 y3 x4 y4"

        return True, "Valid OBB format"
    except Exception as e:
        return False, f"Error reading label: {str(e)}"


def validate_dataset_structure(data_path):
    """
    Validate that the dataset has correct structure for OBB object detection.

    Accepts multiple formats:
    1. Path to data.yaml file
    2. Path to folder containing data.yaml
    3. Standard YOLO structure: images/ and labels/ with train/val subfolders
    4. User format: train/valid/test folders, each containing images/ and labels/
    5. Simple format: train/valid/test folders with images and labels mixed

    Labels must be in OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 values)

    Args:
        data_path: Path to dataset folder or YAML file

    Returns:
        tuple: (is_valid, message, class_names)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        return False, f"Dataset path does not exist: {data_path}", []

    def check_obb_labels(labels_folder):
        """Check if labels in folder are OBB format"""
        label_files = [f for f in labels_folder.iterdir() if f.suffix == '.txt']
        if len(label_files) > 0:
            # Check first non-empty label file
            for lf in label_files[:5]:  # Check up to 5 files
                is_obb, msg = validate_obb_label_format(lf)
                if not is_obb:
                    return False, msg
            return True, "OBB format validated"
        return True, "No labels to check"

    # Check if it's a YAML file
    if data_path.suffix in ['.yaml', '.yml']:
        try:
            with open(data_path, 'r') as f:
                data_yaml = yaml.safe_load(f)

            if 'names' not in data_yaml:
                return False, "data.yaml missing 'names' field", []

            class_names = list(data_yaml['names'].values()) if isinstance(data_yaml['names'], dict) else data_yaml['names']
            return True, f"Valid YAML dataset with {len(class_names)} classes", class_names
        except Exception as e:
            return False, f"Error reading YAML: {str(e)}", []

    # Check if folder contains data.yaml
    yaml_path = data_path / 'data.yaml'
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)

            if 'names' not in data_yaml:
                return False, "data.yaml missing 'names' field", []

            class_names = list(data_yaml['names'].values()) if isinstance(data_yaml['names'], dict) else data_yaml['names']

            # Try to find and validate labels
            train_labels = None
            if (data_path / 'train' / 'labels').exists():
                train_labels = data_path / 'train' / 'labels'
            elif (data_path / 'labels' / 'train').exists():
                train_labels = data_path / 'labels' / 'train'

            if train_labels:
                is_obb, msg = check_obb_labels(train_labels)
                if not is_obb:
                    return False, f"Invalid label format!\n\n{msg}\n\nOBB format required: class_id x1 y1 x2 y2 x3 y3 x4 y4", class_names

            return True, f"Valid OBB dataset with {len(class_names)} classes", class_names
        except Exception as e:
            return False, f"Error reading data.yaml: {str(e)}", []

    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    # Format 1: Standard YOLO structure (images/ and labels/ at root)
    images_path = data_path / 'images'
    labels_path = data_path / 'labels'

    if images_path.exists() and labels_path.exists():
        train_images = images_path / 'train'
        train_labels = labels_path / 'train'

        if train_images.exists() and train_labels.exists():
            images = [f for f in train_images.iterdir() if f.suffix.lower() in image_exts]
            labels = [f for f in train_labels.iterdir() if f.suffix == '.txt']

            if len(images) == 0:
                return False, "No images found in images/train/", []
            if len(labels) == 0:
                return False, "No labels found in labels/train/", []

            # Validate OBB format
            is_obb, msg = check_obb_labels(train_labels)
            if not is_obb:
                return False, f"Invalid label format!\n\n{msg}\n\nOBB format required: class_id x1 y1 x2 y2 x3 y3 x4 y4", []

            return True, f"Valid OBB dataset ({len(images)} images, {len(labels)} labels)", []

    # Format 2: User format - train/valid/test folders with images/ and labels/ inside each
    train_path = data_path / 'train'
    valid_path = data_path / 'valid'

    # Also check for 'val' instead of 'valid'
    if not valid_path.exists():
        valid_path = data_path / 'val'

    if train_path.exists():
        # Check if train has images/ and labels/ subfolders
        train_images_path = train_path / 'images'
        train_labels_path = train_path / 'labels'

        if train_images_path.exists() and train_labels_path.exists():
            images = [f for f in train_images_path.iterdir() if f.suffix.lower() in image_exts]
            labels = [f for f in train_labels_path.iterdir() if f.suffix == '.txt']

            if len(images) == 0:
                return False, "No images found in train/images/", []
            if len(labels) == 0:
                return False, "No labels found in train/labels/", []

            # Validate OBB format
            is_obb, msg = check_obb_labels(train_labels_path)
            if not is_obb:
                return False, f"Invalid label format!\n\n{msg}\n\nOBB format required: class_id x1 y1 x2 y2 x3 y3 x4 y4", []

            return True, f"Valid OBB dataset ({len(images)} train images, {len(labels)} train labels)", []

        # Format 3: train/valid/test with images and labels mixed in same folder
        images = [f for f in train_path.iterdir() if f.suffix.lower() in image_exts]
        labels = [f for f in train_path.iterdir() if f.suffix == '.txt']

        if len(images) > 0 and len(labels) > 0:
            # Validate OBB format
            is_obb, msg = check_obb_labels(train_path)
            if not is_obb:
                return False, f"Invalid label format!\n\n{msg}\n\nOBB format required: class_id x1 y1 x2 y2 x3 y3 x4 y4", []
            return True, f"Valid OBB dataset ({len(images)} train images, {len(labels)} train labels)", []

        # Check if there are subfolders with images
        subfolders = [d for d in train_path.iterdir() if d.is_dir()]
        total_images = 0
        total_labels = 0
        for folder in subfolders:
            total_images += len([f for f in folder.iterdir() if f.suffix.lower() in image_exts])
            total_labels += len([f for f in folder.iterdir() if f.suffix == '.txt'])

        if total_images > 0:
            return True, f"Valid dataset ({total_images} images found in train subfolders)", []

    return False, """Invalid dataset structure for OBB object detection.

Supported formats:

1. Standard YOLO:
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml

2. Split folders:
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── data.yaml

3. Simple format:
   dataset/
   ├── train/ (images + labels)
   ├── valid/ (images + labels)
   └── data.yaml

IMPORTANT: Labels must be in OBB format (9 values per line):
class_id x1 y1 x2 y2 x3 y3 x4 y4

(NOT standard YOLO format which has 5 values)

Please ensure data.yaml exists with class names.""", []


def prepare_data_yaml(data_path):
    """
    Prepare/update data.yaml to work with the detected dataset structure.

    Args:
        data_path: Path to dataset folder

    Returns:
        Path to data.yaml file
    """
    data_path = Path(data_path)
    yaml_path = data_path / 'data.yaml'

    # Read existing data.yaml if it exists
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
    else:
        data_yaml = {}

    # Detect dataset structure and update paths
    train_path = data_path / 'train'
    valid_path = data_path / 'valid'
    test_path = data_path / 'test'

    if not valid_path.exists():
        valid_path = data_path / 'val'

    # Check for Format 2: train/valid/test with images/ and labels/ inside
    if train_path.exists() and (train_path / 'images').exists():
        data_yaml['path'] = str(data_path.absolute())
        data_yaml['train'] = 'train/images'
        if valid_path.exists():
            data_yaml['val'] = f'{valid_path.name}/images'
        if test_path.exists():
            data_yaml['test'] = 'test/images'

    # Check for Format 3: train/valid/test with mixed files
    elif train_path.exists():
        data_yaml['path'] = str(data_path.absolute())
        data_yaml['train'] = 'train'
        if valid_path.exists():
            data_yaml['val'] = valid_path.name
        if test_path.exists():
            data_yaml['test'] = 'test'

    # Check for standard YOLO structure
    elif (data_path / 'images').exists():
        data_yaml['path'] = str(data_path.absolute())
        data_yaml['train'] = 'images/train'
        data_yaml['val'] = 'images/val'
        if (data_path / 'images' / 'test').exists():
            data_yaml['test'] = 'images/test'

    # Write updated data.yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Updated data.yaml at: {yaml_path}")
    return str(yaml_path)


def create_data_yaml(data_path, class_names, output_path=None):
    """
    Create a data.yaml file for the dataset.

    Args:
        data_path: Path to dataset folder
        class_names: List of class names
        output_path: Path for output YAML (default: data_path/data.yaml)

    Returns:
        Path to created YAML file
    """
    data_path = Path(data_path)
    if output_path is None:
        output_path = data_path / 'data.yaml'

    yaml_content = {
        'path': str(data_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }

    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Created data.yaml at: {output_path}")
    return str(output_path)


# ============================================================
# MAIN EXECUTION (for testing)
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO Object Detection Module")
    print("="*60)
    print("\nAvailable models:", get_available_models())
    print("\nTo use in UI:")
    print("  from object_detection.yolo_obj import train_with_user_data")
    print("  model_path, results = train_with_user_data(data_path, config)")
    print("\nDataset structure:")
    print("  dataset/")
    print("  ├── images/")
    print("  │   ├── train/")
    print("  │   └── val/")
    print("  ├── labels/")
    print("  │   ├── train/")
    print("  │   └── val/")
    print("  └── data.yaml")
    print("="*60)

