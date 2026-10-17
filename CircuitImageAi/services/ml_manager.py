import os
import torch
import torch.nn as nn
from torchvision import datasets

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class SimpleCNN(nn.Module):
    # Initializes the CNN layers for image classification.
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    # Runs an image batch through the network and returns class scores.
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class MLManager:
    # Initializes model state, class names, and the compute device.
    def __init__(self):
        self.model = None
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reads class names from the training dataset directory.
    def discover_classes(self, data_dir: str = os.path.join("dataset", "train")):
        if os.path.exists(data_dir):
            try:
                ds = datasets.ImageFolder(data_dir)
                self.class_names = ds.classes
            except Exception:
                pass

    # Loads the saved model weights and switches the model to evaluation mode.
    def load_model(self, path: str = "best_model.pth"):
        self.discover_classes()
        if os.path.exists(path) and self.class_names:
            try:
                self.model = SimpleCNN(num_classes=len(self.class_names)).to(self.device)
                state_dict = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                print(f"Failed to load model: {e}")

    # Releases the model and clears unused CUDA memory.
    def unload_model(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

ml_manager = MLManager()