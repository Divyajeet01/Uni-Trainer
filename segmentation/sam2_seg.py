from ultralytics import SAM
import torch
from pathlib import Path


class SAM2Segmentor:
    """SAM2 (Segment Anything Model 2) for Instance Segmentation training and inference."""

    def __init__(self, model_name="sam2_b.pt"):
        """
        Initialize the SAM2 Segmentation model.

        Args:
            model_name: Pre-trained SAM model to use. Options:
                - 'sam_b.pt' (SAM base)
                - 'sam_l.pt' (SAM large)
                - 'sam2_t.pt' (SAM2 tiny - fastest)
                - 'sam2_s.pt' (SAM2 small)
                - 'sam2_b.pt' (SAM2 base - balanced)
                - 'sam2_l.pt' (SAM2 large - most accurate)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model_name = model_name

        # Load SAM model (Ultralytics auto-downloads if not found)
        try:
            self.model = SAM(model_name)
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            raise

    def _get_unique_run_name(self, project, base_name):
        """Return a unique run name by incrementing if needed (like YOLO default behavior)."""
        project_path = Path(project)
        run_name = base_name
        i = 1
        while (project_path / run_name).exists():
            i += 1
            run_name = f"{base_name}{i}"
        return run_name

    def train(self, data_path, epochs=50, imgsz=1024, batch=4,
              project="runs/segment", name="sam2_trained", save_period=10):
        """
        Train/Fine-tune the SAM2 segmentation model.

        ⚠️ IMPORTANT: SAM2 is primarily a ZERO-SHOT model designed for prompt-based
        segmentation. Training/fine-tuning is limited compared to YOLO-Seg.

        For most use cases, use predict() with prompts (points/boxes) instead of training.

        Dataset Format for SAM2 Fine-tuning (YOLO Segmentation Format):
        ================================================================

        Structure:
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
            └── data.yaml (auto-generated if missing)

        Label Format (Polygon coordinates, normalized 0-1):
            class_id x1 y1 x2 y2 x3 y3 ... xn yn

            Example (triangle with class 0):
            0 0.25 0.1 0.75 0.1 0.5 0.9

            Example (quadrilateral with class 1):
            1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9

        Args:
            data_path: Path to dataset YAML file or folder
            epochs: Number of training epochs
            imgsz: Image size for training (SAM uses larger images, default 1024)
            batch: Batch size (smaller due to memory requirements)
            project: Project directory for saving results
            name: Name of the training run
            save_period: Save checkpoint every N epochs

        Returns:
            Training results
        """
        # Ensure unique output folder for each run
        name = self._get_unique_run_name(project, name)

        print(f"\n{'='*50}")
        print(f"Starting SAM2 Segmentation Training")
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

    def predict(self, image_path, points=None, labels=None, boxes=None, conf=0.25, save=True):
        """
        Run inference on an image using SAM2.

        SAM supports multiple prompt types:
        - Point prompts: Click on specific locations
        - Box prompts: Draw bounding boxes around objects
        - Automatic mode: Segment everything

        Args:
            image_path: Path to image or directory of images
            points: Point prompts [[x1, y1], [x2, y2], ...] (optional)
            labels: Labels for points [1, 0, ...] (1=foreground, 0=background)
            boxes: Bounding box prompts [[x1, y1, x2, y2], ...] (optional)
            conf: Confidence threshold
            save: Save results to disk

        Returns:
            Prediction results
        """
        predict_args = {
            'source': image_path,
            'conf': conf,
            'device': self.device,
            'verbose': True,
            'save': save
        }

        # Add prompts if provided
        if points is not None:
            predict_args['points'] = points
        if labels is not None:
            predict_args['labels'] = labels
        if boxes is not None:
            predict_args['bboxes'] = boxes

        results = self.model.predict(**predict_args)

        for result in results:
            masks = result.masks
            if masks is not None:
                print(f"\nImage: {result.path}")
                print(f"Detected {len(masks)} segmentation masks")
            else:
                print(f"\nImage: {result.path}")
                print("No masks detected")

        return results

    def segment_everything(self, image_path, save=True):
        """
        Automatic mask generation - segment everything in the image.

        Args:
            image_path: Path to image
            save: Save results to disk

        Returns:
            Prediction results with all detected masks
        """
        print(f"Running automatic segmentation on: {image_path}")

        results = self.model.predict(
            source=image_path,
            device=self.device,
            verbose=True,
            save=save
        )

        for result in results:
            masks = result.masks
            if masks is not None:
                print(f"Found {len(masks)} masks")

        return results

    def load_trained_model(self, model_path):
        """
        Load a trained model from .pt file.

        Args:
            model_path: Path to the trained model .pt file
        """
        self.model = SAM(model_path)
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
    "model": "sam2_b.pt",       # Model: sam_b/l.pt or sam2_t/s/b/l.pt
    "epochs": 50,               # Number of training epochs
    "imgsz": 1024,              # Image size (SAM uses larger images)
    "batch": 4,                 # Batch size (smaller due to memory)
    "project": "runs/segment",  # Output directory
    "name": "sam2_trained",     # Model name
}


def train_with_user_data(data_path, config=None, progress_callback=None):
    """
    Train SAM2 segmentor with user-uploaded data.

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
    segmentor = SAM2Segmentor(model_name=config["model"])

    # Train
    results, actual_name = segmentor.train(
        data_path=data_path,
        epochs=config["epochs"],
        imgsz=config.get("imgsz", 1024),
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
    segmentor = SAM2Segmentor()
    segmentor.load_trained_model(model_path)
    return segmentor.predict(image_path, conf=conf)


def get_available_models():
    """Returns list of available SAM segmentation models."""
    return [
        "sam_b.pt",     # SAM base
        "sam_l.pt",     # SAM large
        "sam2_t.pt",    # SAM2 tiny
        "sam2_s.pt",    # SAM2 small
        "sam2_b.pt",    # SAM2 base
        "sam2_l.pt",    # SAM2 large
    ]


def validate_dataset_structure(data_path):
    """
    Validate that the dataset has correct structure for SAM2 segmentation.
    Uses the same format as YOLO segmentation.

    Args:
        data_path: Path to dataset folder or YAML file

    Returns:
        tuple: (is_valid, message, class_names)
    """
    # Import from yolo_seg to reuse validation logic
    from segmentation.yolo_seg import validate_dataset_structure as yolo_validate
    return yolo_validate(data_path)


def prepare_data_yaml(data_path):
    """
    Auto-generate data.yaml for SAM2 segmentation dataset.
    Uses the same format as YOLO segmentation.

    Args:
        data_path: Path to dataset folder

    Returns:
        Path to data.yaml file
    """
    # Import from yolo_seg to reuse data.yaml generation
    from segmentation.yolo_seg import prepare_data_yaml as yolo_prepare_yaml
    return yolo_prepare_yaml(data_path)


# ============================================================
# MAIN EXECUTION (for testing)
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SAM2 Segmentation Module")
    print("="*60)
    print("\nAvailable models:", get_available_models())
    print("\nTo use in UI:")
    print("  from segmentation.sam2_seg import train_with_user_data")
    print("  model_path, results = train_with_user_data(data_path, config)")
    print("\nDataset structure (same as YOLO Segmentation):")
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
    print("\nSAM2 Prompting options:")
    print("  - Point prompts: Click on specific locations")
    print("  - Box prompts: Draw bounding boxes around objects")
    print("  - Automatic mode: Segment everything")
    print("="*60)

