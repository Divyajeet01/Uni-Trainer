from ultralytics import YOLO
import torch
from pathlib import Path


class YOLOClassifier:
    """YOLO-based Image Classification class for training and inference."""

    def __init__(self, model_name="yolov8n-cls.pt"):
        """
        Initialize the YOLO classifier.

        Args:
            model_name: Pre-trained model to use. Options:
                - 'yolov8n-cls.pt' (nano - fastest, smallest)
                - 'yolov8s-cls.pt' (small)
                - 'yolov8m-cls.pt' (medium)
                - 'yolov8l-cls.pt' (large)
                - 'yolov8x-cls.pt' (extra large - most accurate)
                - 'yolo11n-cls.pt' (YOLO11 nano)
                - 'yolo11s-cls.pt' (YOLO11 small)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load pre-trained YOLO classification model
        self.model = YOLO(model_name)
        self.model_name = model_name

    def train(self, data_path, epochs=100, imgsz=224, batch=16,
              project="runs/classify", name="train", save_period=10):
        """
        Train the YOLO classification model.

        Args:
            data_path: Path to dataset folder. Structure should be:
                       dataset/
                       ├── train/
                       │   ├── class1/
                       │   │   ├── img1.jpg
                       │   │   └── img2.jpg
                       │   └── class2/
                       │       ├── img1.jpg
                       │       └── img2.jpg
                       └── val/
                           ├── class1/
                           └── class2/
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Project directory for saving results
            name: Name of the training run
            save_period: Save checkpoint every N epochs

        Returns:
            Training results
        """
        print(f"\n{'='*2}")
        print(f"Starting YOLO Classification Training")
        print(f"{'='*2}")
        print(f"Dataset: {data_path}")
        print(f"Epochs: {epochs}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch}")
        print(f"Device: {self.device}")
        print(f"{'='*2}\n")

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
            patience=20,  # Early stopping patience
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            plots=True,  # Generate training plots
        )

        print(f"\n{'='*2}")
        print(f"Training Complete!")
        print(f"Best model saved at: {project}/{name}/weights/best.pt")
        print(f"Last model saved at: {project}/{name}/weights/last.pt")
        print(f"{'='*2}\n")

        return results

    def predict(self, image_path, conf=0.25):
        """
        Run inference on an image.

        Args:
            image_path: Path to image or directory of images
            conf: Confidence threshold

        Returns:
            Prediction results
        """
        results = self.model.predict(
            source=image_path,
            conf=conf,
            device=self.device,
            verbose=True
        )

        for result in results:
            probs = result.probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            top5_idx = probs.top5
            top5_conf = probs.top5conf.tolist()

            print(f"\nImage: {result.path}")
            print(f"Top-1 Prediction: {result.names[top1_idx]} ({top1_conf*100:.2f}%)")
            print(f"Top-5 Predictions:")
            for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
                print(f"  {i+1}. {result.names[idx]}: {conf*100:.2f}%")

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
            data_path: Path to validation dataset

        Returns:
            Validation metrics
        """
        results = self.model.val(data=data_path, device=self.device)
        return results


def create_sample_dataset_structure(base_path="dataset"):
    """
    Create sample dataset folder structure for reference.

    Args:
        base_path: Base path for the dataset
    """
    folders = [
        f"{base_path}/train/class1",
        f"{base_path}/train/class2",
        f"{base_path}/val/class1",
        f"{base_path}/val/class2",
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    print(f"\nCreated dataset structure at: {base_path}/")
    print("Structure:")
    print(f"  {base_path}/")
    print(f"  ├── train/")
    print(f"  │   ├── class1/  (add training images here)")
    print(f"  │   └── class2/  (add training images here)")
    print(f"  └── val/")
    print(f"      ├── class1/  (add validation images here)")
    print(f"      └── class2/  (add validation images here)")
    print("\nReplace 'class1', 'class2' with your actual class names.")
    print("Add more class folders as needed.")


# ============================================================
# CONFIGURATION - DEFAULT VALUES
# ============================================================

DEFAULT_CONFIG = {
    "model": "yolov8n-cls.pt",  # Model: yolov8n/s/m/l/x-cls.pt or yolo11n/s-cls.pt
    "epochs": 50,               # Number of training epochs
    "imgsz": 224,               # Image size
    "batch": 16,                # Batch size
    "project": "runs/classify", # Output directory
    "name": "trained_model",    # Model name
}


def train_with_user_data(data_path, config=None, progress_callback=None):
    """
    Train YOLO classifier with user-uploaded data.

    Args:
        data_path: Path to user's dataset folder (from UI upload)
        config: Optional config dict to override defaults
        progress_callback: Optional callback function(epoch, total_epochs, metrics)

    Returns:
        tuple: (Path to trained model, training results)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Initialize classifier
    classifier = YOLOClassifier(model_name=config["model"])

    # Train
    results = classifier.train(
        data_path=data_path,
        epochs=config["epochs"],
        imgsz=config.get("imgsz", 224),
        batch=config["batch"],
        project=config["project"],
        name=config["name"]
    )

    # Path to best PyTorch model
    model_path = f"{config['project']}/{config['name']}/weights/best.pt"


    return model_path, results


def predict_with_model(model_path, image_path):
    """
    Run prediction using trained model.

    Args:
        model_path: Path to trained .pt model
        image_path: Path to image for prediction

    Returns:
        Prediction results
    """
    classifier = YOLOClassifier()
    classifier.load_trained_model(model_path)
    return classifier.predict(image_path)


def get_available_models():
    """Returns list of available YOLO classification models."""
    return [
        "yolov8n-cls.pt",
        "yolov8s-cls.pt",
        "yolov8m-cls.pt",
        "yolov8l-cls.pt",
        "yolov8x-cls.pt",
        "yolov11n-cls.pt",
        "yolov11s-cls.pt",
    ]


def validate_dataset_structure(data_path):
    """
    Validate that the dataset has correct structure for classification.
    Accepts simple structure: dataset/class1/, dataset/class2/, etc.
    No train/val split required - we'll do it automatically.

    Args:
        data_path: Path to dataset folder

    Returns:
        tuple: (is_valid, message, class_names)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        return False, f"Dataset path does not exist: {data_path}", []

    # Check for simple structure (class folders directly in dataset)
    class_folders = [d for d in data_path.iterdir() if d.is_dir() and d.name not in ['train', 'val', 'test']]

    # Check if already has train/val structure
    train_path = data_path / "train"
    val_path = data_path / "val"

    if train_path.exists() and val_path.exists():
        # Already has train/val structure
        class_names = [d.name for d in train_path.iterdir() if d.is_dir()]
        if len(class_names) < 2:
            return False, "Need at least 2 class folders", []
        return True, f"Valid dataset with {len(class_names)} classes (pre-split)", class_names

    # Simple structure: class folders directly
    if len(class_folders) >= 2:
        class_names = [d.name for d in class_folders]
        # Check if folders have images
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        for folder in class_folders:
            images = [f for f in folder.iterdir() if f.suffix.lower() in image_exts]
            if len(images) < 2:
                return False, f"Class '{folder.name}' needs at least 2 images", class_names
        return True, f"Valid dataset with {len(class_names)} classes (will auto-split)", class_names

    return False, "Need at least 2 class folders with images.\n\nExpected structure:\ndataset/\n├── class1/ (images)\n├── class2/ (images)\n└── class3/ (images)", []


def split_dataset(source_path, output_path=None, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split a simple class-folder dataset into train/val/test sets.

    Args:
        source_path: Path to source dataset with class folders
        output_path: Path for output split dataset (default: source_path + '_split')
        train_ratio: Ratio of training data (default: 0.7)
        val_ratio: Ratio of validation data (default: 0.2)
        test_ratio: Ratio of test data (default: 0.1)

    Returns:
        Path to the split dataset
    """
    import shutil
    import random

    source_path = Path(source_path)
    if output_path is None:
        output_path = source_path.parent / f"{source_path.name}_split"
    else:
        output_path = Path(output_path)

    # Check if already has train/val structure
    if (source_path / "train").exists() and (source_path / "val").exists():
        print("Dataset already has train/val structure, using as-is")
        return str(source_path)

    # Create output directories
    train_path = output_path / "train"
    val_path = output_path / "val"
    test_path = output_path / "test"

    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    # Get class folders
    class_folders = [d for d in source_path.iterdir() if d.is_dir() and d.name not in ['train', 'val', 'test']]

    for class_folder in class_folders:
        class_name = class_folder.name

        # Get all images in this class
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in image_exts]
        random.shuffle(images)

        # Calculate split indices
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Create class directories
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (val_path / class_name).mkdir(parents=True, exist_ok=True)
        if test_images:
            (test_path / class_name).mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy2(img, train_path / class_name / img.name)
        for img in val_images:
            shutil.copy2(img, val_path / class_name / img.name)
        for img in test_images:
            shutil.copy2(img, test_path / class_name / img.name)

        print(f"Class '{class_name}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    print(f"\nDataset split complete! Output: {output_path}")
    return str(output_path)


# ============================================================
# MAIN EXECUTION (for testing)
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO Image Classification Module")
    print("="*60)
    print("\nAvailable models:", get_available_models())
    print("\nTo use in UI:")
    print("  from image_classification.yolo import train_with_user_data")
    print("  model_path, results = train_with_user_data(data_path, config)")
    print("="*60)

