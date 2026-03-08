from ultralytics import YOLO
import torch
from pathlib import Path
import yaml


class YOLOSegmentor:
    """YOLO-based Instance Segmentation class for training and inference."""

    def __init__(self, model_name="yolo11n-seg.pt"):
        """
        Initialize the YOLO Segmentation model.

        Args:
            model_name: Pre-trained segmentation model to use. Options:
                - 'yolo11n-seg.pt' (nano - fastest, smallest)
                - 'yolo11s-seg.pt' (small)
                - 'yolo11m-seg.pt' (medium)
                - 'yolo11l-seg.pt' (large)
                - 'yolo11x-seg.pt' (extra large - most accurate)
                - 'yolov8n-seg.pt' (nano)
                - 'yolov8s-seg.pt' (small)
                - 'yolov8m-seg.pt' (medium)
                - 'yolov8l-seg.pt' (large)
                - 'yolov8x-seg.pt' (extra large)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load pre-trained YOLO segmentation model
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
              project="runs/segment", name="train", save_period=10):
        """
        Train the YOLO segmentation model.

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
                       │   │   ├── img1.txt  (YOLO segmentation format)
                       │   │   └── img2.txt
                       │   └── val/
                       │       ├── img1.txt
                       │       └── img2.txt
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
        print(f"Starting YOLO Segmentation Training")
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
            optimizer='auto',
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
            masks = result.masks
            print(f"\nImage: {result.path}")
            print(f"Detected {len(boxes)} objects with segmentation masks:")

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


def create_sample_dataset_structure(base_path="dataset_segment"):
    """
    Create sample dataset folder structure for segmentation.
    No data.yaml needed - it will be auto-generated during training.

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

    print(f"\nCreated dataset structure at: {base_path}/")
    print("Structure:")
    print(f"  {base_path}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/  (add training images here)")
    print(f"  │   └── val/    (add validation images here)")
    print(f"  └── labels/")
    print(f"      ├── train/  (add training labels here)")
    print(f"      └── val/    (add validation labels here)")
    print("\nNote: data.yaml will be auto-generated during training!")
    print("\nLabel format (YOLO Segmentation):")
    print("  class_id x1 y1 x2 y2 x3 y3 ... xn yn")
    print("  (Polygon coordinates, normalized to [0, 1])")


# ============================================================
# CONFIGURATION - DEFAULT VALUES
# ============================================================

DEFAULT_CONFIG = {
    "model": "yolo11n-seg.pt",  # Model: yolo11n/s/m/l/x-seg.pt or yolov8n/s/m/l/x-seg.pt
    "epochs": 100,              # Number of training epochs
    "imgsz": 640,               # Image size
    "batch": 16,                # Batch size
    "project": "runs/segment",  # Output directory
    "name": "trained_model",
    "lr": 0.01# Model name
}


def train_with_user_data(data_path, config=None, progress_callback=None):
    """
    Train YOLO segmentor with user-uploaded data.

    Args:
        data_path: Path to user's dataset folder or YAML file (from UI upload)
        config: Optional config dict to override defaults
        progress_callback: Optional callback function(epoch, total_epochs, metrics)

    Returns:
        tuple: (Path to trained model, training results)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Initialize segmentor
    segmentor = YOLOSegmentor(model_name=config["model"])

    # Train
    results, actual_name = segmentor.train(
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
    segmentor = YOLOSegmentor()
    segmentor.load_trained_model(model_path)
    return segmentor.predict(image_path, conf=conf)


def get_available_models():
    """Returns list of available YOLO segmentation models."""
    return [
        "yolo11n-seg.pt",
        "yolo11s-seg.pt",
        "yolo11m-seg.pt",
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolov8n-seg.pt",
        "yolov8s-seg.pt",
        "yolov8m-seg.pt",
        "yolov8l-seg.pt",
        "yolov8x-seg.pt",
    ]


def validate_segmentation_label_format(label_path):
    """
    Validate that a label file is in segmentation format (polygon coordinates).

    Segmentation format: class_id x1 y1 x2 y2 x3 y3 ... xn yn (at least 6 coordinate values for a triangle)
    Standard detection format: class_id x_center y_center width height (5 values)

    Args:
        label_path: Path to a label .txt file

    Returns:
        tuple: (is_valid, message)
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
                return False, f"Label has 5 values (standard detection format). Segmentation requires polygon coordinates.\nFormat: class_id x1 y1 x2 y2 x3 y3 ... xn yn"
            elif num_values >= 7:
                # At least class_id + 3 points (6 coordinates)
                return True, "Valid segmentation format"
            else:
                return False, f"Label has {num_values} values. Segmentation requires at least 7 values.\nFormat: class_id x1 y1 x2 y2 x3 y3 ... xn yn"

        return True, "Valid segmentation format"
    except Exception as e:
        return False, f"Error reading label: {str(e)}"


def validate_dataset_structure(data_path):
    """
    Validate that the dataset has correct structure for instance segmentation.

    No data.yaml required - will be auto-generated if needed.

    Accepts multiple formats:
    1. Standard YOLO structure: images/ and labels/ with train/val subfolders
    2. User format: train/valid/test folders, each containing images/ and labels/
    3. Path to data.yaml file (optional)

    Labels must be in segmentation format: class_id x1 y1 x2 y2 x3 y3 ... xn yn

    Args:
        data_path: Path to dataset folder or YAML file

    Returns:
        tuple: (is_valid, message, class_names)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        return False, f"Dataset path does not exist: {data_path}", []

    def check_seg_labels(labels_folder):
        """Check if labels in folder are segmentation format"""
        label_files = [f for f in labels_folder.iterdir() if f.suffix == '.txt']
        if len(label_files) > 0:
            # Check first non-empty label file
            for lf in label_files[:5]:  # Check up to 5 files
                is_valid, msg = validate_segmentation_label_format(lf)
                if not is_valid:
                    return False, msg
            return True, "Segmentation format validated"
        return True, "No labels to check"

    def extract_classes_from_labels(labels_folder):
        """Extract unique class IDs from label files"""
        class_ids = set()
        label_files = [f for f in labels_folder.iterdir() if f.suffix == '.txt']
        for lf in label_files[:100]:  # Check up to 100 files
            try:
                with open(lf, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) >= 1:
                            class_ids.add(int(values[0]))
            except:
                continue
        return sorted(list(class_ids))

    # Check if it's a YAML file (optional support)
    if data_path.suffix in ['.yaml', '.yml']:
        try:
            with open(data_path, 'r') as f:
                data_yaml = yaml.safe_load(f)

            class_names = []
            if 'names' in data_yaml:
                class_names = list(data_yaml['names'].values()) if isinstance(data_yaml['names'], dict) else data_yaml['names']
            return True, f"Valid YAML dataset with {len(class_names)} classes", class_names
        except Exception as e:
            return False, f"Error reading YAML: {str(e)}", []

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

            # Validate segmentation format
            is_valid, msg = check_seg_labels(train_labels)
            if not is_valid:
                return False, f"Invalid label format!\n\n{msg}\n\nSegmentation format required: class_id x1 y1 x2 y2 x3 y3 ... xn yn", []

            # Extract class names from labels
            class_ids = extract_classes_from_labels(train_labels)
            class_names = [f"class_{i}" for i in class_ids] if class_ids else []

            return True, f"Valid segmentation dataset ({len(images)} images, {len(labels)} labels)", class_names

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

            # Validate segmentation format
            is_valid, msg = check_seg_labels(train_labels_path)
            if not is_valid:
                return False, f"Invalid label format!\n\n{msg}\n\nSegmentation format required: class_id x1 y1 x2 y2 x3 y3 ... xn yn", []

            # Extract class names from labels
            class_ids = extract_classes_from_labels(train_labels_path)
            class_names = [f"class_{i}" for i in class_ids] if class_ids else []

            return True, f"Valid segmentation dataset ({len(images)} train images, {len(labels)} train labels)", class_names

    return False, """Invalid dataset structure for instance segmentation.

Supported formats:

1. Standard YOLO:
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/

2. Split folders:
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── valid/
       ├── images/
       └── labels/

Note: data.yaml will be auto-generated if not present.

IMPORTANT: Labels must be in segmentation format (polygon coordinates):
class_id x1 y1 x2 y2 x3 y3 ... xn yn

(NOT standard YOLO detection format which has 5 values)""", []


def prepare_data_yaml(data_path):
    """
    Auto-generate data.yaml for segmentation dataset.
    Extracts class IDs from label files and creates appropriate YAML.

    Args:
        data_path: Path to dataset folder

    Returns:
        Path to data.yaml file
    """
    data_path = Path(data_path)
    yaml_path = data_path / 'data.yaml'

    # Try to extract class IDs from labels
    def extract_classes_from_labels(labels_folder):
        """Extract unique class IDs from label files"""
        class_ids = set()
        if labels_folder.exists():
            label_files = [f for f in labels_folder.iterdir() if f.suffix == '.txt']
            for lf in label_files[:200]:  # Check up to 200 files
                try:
                    with open(lf, 'r') as f:
                        for line in f:
                            values = line.strip().split()
                            if len(values) >= 1:
                                class_ids.add(int(values[0]))
                except:
                    continue
        return sorted(list(class_ids))

    # Find labels folder
    train_labels = None
    if (data_path / 'train' / 'labels').exists():
        train_labels = data_path / 'train' / 'labels'
    elif (data_path / 'labels' / 'train').exists():
        train_labels = data_path / 'labels' / 'train'

    # Extract class IDs
    class_ids = extract_classes_from_labels(train_labels) if train_labels else []

    # Create class names dict
    names_dict = {i: f"class_{i}" for i in class_ids} if class_ids else {0: "object"}

    # Read existing data.yaml if it exists (to preserve custom class names)
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                existing_yaml = yaml.safe_load(f)
            if existing_yaml and 'names' in existing_yaml:
                names_dict = existing_yaml['names']
        except:
            pass

    # Detect dataset structure and create YAML
    data_yaml = {'names': names_dict}

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

    # Check for standard YOLO structure
    elif (data_path / 'images').exists():
        data_yaml['path'] = str(data_path.absolute())
        data_yaml['train'] = 'images/train'
        data_yaml['val'] = 'images/val'
        if (data_path / 'images' / 'test').exists():
            data_yaml['test'] = 'images/test'

    # Write data.yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Auto-generated data.yaml at: {yaml_path}")
    print(f"  Classes detected: {len(names_dict)}")
    return str(yaml_path)


# ============================================================
# MAIN EXECUTION (for testing)
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO Segmentation Module")
    print("="*60)
    print("\nAvailable models:", get_available_models())
    print("\nTo use in UI:")
    print("  from segmentation.yolo_seg import train_with_user_data")
    print("  model_path, results = train_with_user_data(data_path, config)")
    print("\nDataset structure (no data.yaml required!):")
    print("  dataset/")
    print("  ├── images/")
    print("  │   ├── train/")
    print("  │   └── val/")
    print("  └── labels/")
    print("      ├── train/")
    print("      └── val/")
    print("\nLabel format (Segmentation - polygon coordinates):")
    print("  class_id x1 y1 x2 y2 x3 y3 ... xn yn")
    print("  (all coordinate values normalized to [0, 1])")
    print("\nNote: data.yaml will be auto-generated during training!")
    print("="*60)

