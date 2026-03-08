import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import yaml
from datetime import datetime


class ResNetClassifier:
    """ResNet-based Image Classification class for training and inference."""

    AVAILABLE_MODELS = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    def __init__(self, model_name="resnet18", num_classes=None, pretrained=True):
        """
        Initialize the ResNet classifier.

        Args:
            model_name: Pre-trained model to use. Options:
                - 'resnet18' (smallest, fastest)
                - 'resnet34'
                - 'resnet50' (good balance)
                - 'resnet101'
                - 'resnet152' (largest, most accurate)
            num_classes: Number of output classes (set during training)
            pretrained: Whether to use pretrained ImageNet weights
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None
        self.class_names = None

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}")

    def _build_model(self, num_classes):
        """Build the ResNet model with custom classifier head."""
        # Load pretrained model
        weights = "IMAGENET1K_V1" if self.pretrained else None
        model = self.AVAILABLE_MODELS[self.model_name](weights=weights)

        # Replace the final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model.to(self.device)

    def _get_transforms(self, imgsz, is_training=True):
        """Get data transforms for training/validation."""
        if is_training:
            return transforms.Compose([
                transforms.RandomResizedCrop(imgsz),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(imgsz * 1.14)),
                transforms.CenterCrop(imgsz),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def _get_unique_run_name(self, project, base_name):
        """Return a unique run name by incrementing if needed (like YOLO)."""
        project_path = Path(project)
        run_name = base_name
        i = 1
        while (project_path / run_name).exists():
            i += 1
            run_name = f"{base_name}{i}"
        return run_name

    def train(self, data_path, epochs=50, imgsz=224, batch=32,
              project="runs/classify", name="resnet_model", learning_rate=0.001):
        """
        Train the ResNet classification model.

        Args:
            data_path: Path to dataset folder. Structure should be:
                       dataset/
                       ├── train/
                       │   ├── class1/
                       │   └── class2/
                       └── val/
                           ├── class1/
                           └── class2/
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Project directory for saving results
            name: Name of the training run
            learning_rate: Learning rate for optimizer

        Returns:
            Training results dict
        """
        print(f"\n{'='*50}")
        print(f"Starting ResNet Classification Training")
        print(f"{'='*50}")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {data_path}")
        print(f"Epochs: {epochs}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"{'='*50}\n")

        # Ensure unique output folder for each run
        name = self._get_unique_run_name(project, name)

        # Prepare data loaders
        train_transform = self._get_transforms(imgsz, is_training=True)
        val_transform = self._get_transforms(imgsz, is_training=False)

        train_path = Path(data_path) / "train"
        val_path = Path(data_path) / "val"

        train_dataset = datasets.ImageFolder(str(train_path), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(val_path), transform=val_transform)

        self.class_names = train_dataset.classes
        num_classes = len(self.class_names)

        print(f"Found {num_classes} classes: {self.class_names}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                  num_workers=0, pin_memory=True if self.device == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True if self.device == 'cuda' else False)

        # Build model
        self.model = self._build_model(num_classes)
        self.num_classes = num_classes

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Training loop
        best_val_acc = 0.0
        results = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_acc': 0.0,
        }

        # Create output directory
        output_dir = Path(project) / name / "weights"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save args.yaml with training configuration
        args_dict = {
            'task': 'classify',
            'mode': 'train',
            'model': self.model_name,
            'data': str(data_path),
            'epochs': epochs,
            'batch': batch,
            'imgsz': imgsz,
            'device': self.device,
            'project': project,
            'name': name,
            'pretrained': self.pretrained,
            'optimizer': 'Adam',
            'learning_rate': learning_rate,
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 5,
            'num_classes': num_classes,
            'class_names': self.class_names,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'seed': 42,
            'deterministic': True,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            # Data augmentation settings
            'augmentation': {
                'random_resized_crop': imgsz,
                'random_horizontal_flip': True,
                'random_rotation': 15,
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2,
                },
            },
            # Normalization (ImageNet stats)
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
            },
        }

        args_path = Path(project) / name / 'args.yaml'
        with open(args_path, 'w') as f:
            yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Training args saved to: {args_path}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100. * correct / total

            # Update scheduler
            scheduler.step(val_loss)

            # Save results
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                results['best_acc'] = best_val_acc
                self._save_model(output_dir / "best.pt")
                print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")

            # Save last model
            self._save_model(output_dir / "last.pt")

        # Save results.csv
        results_path = Path(project) / name / 'results.csv'
        with open(results_path, 'w') as f:
            f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')
            for i in range(len(results['train_loss'])):
                f.write(f"{i+1},{results['train_loss'][i]:.6f},{results['train_acc'][i]:.4f},"
                        f"{results['val_loss'][i]:.6f},{results['val_acc'][i]:.4f}\n")
        print(f"Training results saved to: {results_path}")

        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Best model saved at: {output_dir}/best.pt")
        print(f"Last model saved at: {output_dir}/last.pt")
        print(f"Args saved at: {Path(project) / name / 'args.yaml'}")
        print(f"Results saved at: {results_path}")
        print(f"{'='*50}\n")

        return results

    def _save_model(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
        }
        torch.save(checkpoint, path)

    def load_trained_model(self, model_path):
        """
        Load a trained model from .pt file.

        Args:
            model_path: Path to the trained model .pt file
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model_name = checkpoint['model_name']
        self.num_classes = checkpoint['num_classes']
        self.class_names = checkpoint['class_names']

        self.model = self._build_model(self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded {self.model_name} model from: {model_path}")
        print(f"Classes: {self.class_names}")

    def predict(self, image_path, conf=0.25):
        """
        Run inference on an image.

        Args:
            image_path: Path to image
            conf: Confidence threshold

        Returns:
            Prediction results dict
        """
        from PIL import Image

        if self.model is None:
            raise ValueError("Model not loaded. Call load_trained_model() first.")

        self.model.eval()

        # Load and transform image
        transform = self._get_transforms(224, is_training=False)
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get top-5 predictions
            top5_prob, top5_idx = torch.topk(probabilities, min(5, self.num_classes))
            top5_prob = top5_prob.squeeze().cpu().tolist()
            top5_idx = top5_idx.squeeze().cpu().tolist()

            if isinstance(top5_prob, float):
                top5_prob = [top5_prob]
                top5_idx = [top5_idx]

        # Format results
        top1_class = self.class_names[top5_idx[0]]
        top1_conf = top5_prob[0]

        print(f"\nImage: {image_path}")
        print(f"Top-1 Prediction: {top1_class} ({top1_conf*100:.2f}%)")
        print(f"Top-5 Predictions:")
        for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob)):
            print(f"  {i+1}. {self.class_names[idx]}: {prob*100:.2f}%")

        return {
            'top1_class': top1_class,
            'top1_conf': top1_conf,
            'top5': [(self.class_names[idx], prob) for idx, prob in zip(top5_idx, top5_prob)],
            'path': image_path
        }


# ============================================================
# CONFIGURATION - DEFAULT VALUES
# ============================================================

DEFAULT_CONFIG = {
    "model": "resnet18",
    "epochs": 50,
    "imgsz": 224,
    "batch": 32,
    "learning_rate": 0.001,
    "project": "runs/classify",
    "name": "resnet_model",
}


def train_with_user_data(data_path, config=None, progress_callback=None):
    """
    Train ResNet classifier with user-uploaded data.

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
    classifier = ResNetClassifier(model_name=config["model"])

    # Train
    results = classifier.train(
        data_path=data_path,
        epochs=config["epochs"],
        imgsz=config.get("imgsz", 224),
        batch=config["batch"],
        project=config["project"],
        name=config["name"],
        learning_rate=config.get("learning_rate", 0.001)
    )

    # Path to best PyTorch model
    model_path = f"{config['project']}/{config['name']}/weights/best.pt"

    return model_path, results


def predict_with_model(model_path, image_path):
    """
    Run prediction using trained ResNet model.

    Args:
        model_path: Path to trained .pt model
        image_path: Path to image for prediction

    Returns:
        Prediction results
    """
    classifier = ResNetClassifier()
    classifier.load_trained_model(model_path)
    return classifier.predict(image_path)


def get_available_models():
    """Returns list of available ResNet classification models."""
    return [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ]


def validate_dataset_structure(data_path):
    """
    Validate that the dataset has correct structure for classification.
    Same as YOLO validation for consistency.

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


# ============================================================
# MAIN EXECUTION (for testing)
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ResNet Image Classification Module")
    print("="*60)
    print("\nAvailable models:", get_available_models())
    print("\nTo use in UI:")
    print("  from image_classification.resnet import train_with_user_data")
    print("  model_path, results = train_with_user_data(data_path, config)")
    print("="*60)

