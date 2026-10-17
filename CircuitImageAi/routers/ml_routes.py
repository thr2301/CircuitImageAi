import os
import traceback
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from services.ml_manager import ml_manager, SimpleCNN, MODELS_DIR

router = APIRouter()

@router.get("/train_stream")
# Streams progress while training a new image-classification model.
def train_stream():
    # Yields server-sent training progress and error events.
    def event_generator():
        try:
            data_dir = os.path.join("dataset", "train")
            if not os.path.exists(data_dir):
                yield "data: ERROR: dataset/train not found.\n\n"
                return

            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            train_dataset = datasets.ImageFolder(data_dir, transform=transform)
            if not train_dataset.classes:
                yield "data: ERROR: No classes found.\n\n"
                return

            ml_manager.class_names = train_dataset.classes
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

            ml_manager.model = SimpleCNN(num_classes=len(ml_manager.class_names)).to(ml_manager.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(ml_manager.model.parameters(), lr=0.001)
            num_epochs = 30

            yield f"data: Classes: {', '.join(ml_manager.class_names)}\n\n"

            for epoch in range(num_epochs):
                ml_manager.model.train()
                running_loss, correct, total = 0.0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(ml_manager.device), labels.to(ml_manager.device)
                    optimizer.zero_grad()
                    outputs = ml_manager.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                avg_loss = running_loss / max(1, len(train_loader))
                acc = 100.0 * correct / max(1, total)
                progress = int(((epoch + 1) / num_epochs) * 100)
                msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Progress: {progress}%"
                yield f"data: {msg}\n\n"

            torch.save(ml_manager.model.state_dict(), "best_model.pth")
            yield "data: Training completed. Model saved.\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
            print(traceback.format_exc())

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/train_stream_resume")
# Streams progress while resuming training of the loaded model.
def train_stream_resume():
    # Yields server-sent resumed-training progress and error events.
    def event_generator():
        try:
            if ml_manager.model is None:
                yield "data: ERROR: No model loaded to resume.\n\n"
                return

            data_dir = os.path.join("dataset", "train")
            if not os.path.exists(data_dir):
                yield "data: ERROR: dataset/train not found.\n\n"
                return

            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            train_dataset = datasets.ImageFolder(data_dir, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            
            ml_manager.model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(ml_manager.model.parameters(), lr=0.0005) 
            num_epochs = 10

            yield f"data: Resuming training for {num_epochs} epochs...\n\n"

            for epoch in range(num_epochs):
                running_loss, correct, total = 0.0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(ml_manager.device), labels.to(ml_manager.device)
                    optimizer.zero_grad()
                    outputs = ml_manager.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                avg_loss = running_loss / max(1, len(train_loader))
                acc = 100.0 * correct / max(1, total)
                progress = int(((epoch + 1) / num_epochs) * 100)
                msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Progress: {progress}%"
                yield f"data: {msg}\n\n"

            torch.save(ml_manager.model.state_dict(), "best_model.pth")
            yield "data: Training completed. Model saved.\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
            print(traceback.format_exc())

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/predict")
# Classifies an uploaded image and returns its label and confidence.
async def predict(file: UploadFile = File(...)):
    if ml_manager.model is None: 
        return {"error": "Model not loaded"}
    
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    tensor_img = t(img).unsqueeze(0).to(ml_manager.device)
    
    ml_manager.model.eval()
    with torch.no_grad():
        out = ml_manager.model(tensor_img)
        conf, idx = torch.max(torch.softmax(out, 1), 1)
    
    label = ml_manager.class_names[idx.item()]
    colors = {"amplifier": "#f39c12", "rc_lp": "#2ecc71", "rc_hp": "#1abc9c", "resonator": "#9b59b6", "other": "#95a5a6"}
    friendly = {"rc_lp": "RC Lowpass", "rc_hp": "RC Highpass", "resonator": "Resonator"}
    
    return {
        "class": label,
        "friendly": friendly.get(label, label),
        "color": colors.get(label, "#000"),
        "confidence": round(conf.item() * 100, 2)
    }

@router.post("/load_model_default")
# Loads the default saved model weights into the current model.
def load_def():
    path = "best_model.pth"
    if not os.path.exists(path): 
        return {"status": "error", "message": "best_model.pth not found"}
    try:
        state = torch.load(path, map_location=ml_manager.device)
        ml_manager.model.load_state_dict(state)
        return {"status": "ok", "classes": ml_manager.class_names}
    except Exception:
        return {"status": "error", "message": "Failed to load default model"}

@router.post("/load_model_file")
# Loads model weights from an uploaded model file.
async def load_f(file: UploadFile = File(...)):
    path = os.path.join(MODELS_DIR, file.filename)
    with open(path, "wb") as f: 
        f.write(await file.read())
    try:
        state = torch.load(path, map_location=ml_manager.device)
        ml_manager.model.load_state_dict(state)
        return {"status": "ok", "classes": ml_manager.class_names}
    except Exception:
        return {"status": "error", "message": "Invalid model file"}