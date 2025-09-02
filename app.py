from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from io import BytesIO
import os, time, json

# --- FastAPI setup ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Users ---
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({"admin":"password"}, f, indent=2)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

# --- CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*56*56,128)
        self.fc2 = nn.Linear(128,num_classes)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,32*56*56)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = None
class_names = []

# --- Label map ---
label_map = {
    "amplifier": {"friendly": "Amplifier", "color": "#f39c12"},
    "rc_lp": {"friendly": "RC Lowpass", "color": "#2ecc71"},
    "rc_hp": {"friendly": "RC Highpass", "color": "#1abc9c"},
    "other": {"friendly": "Other", "color": "#95a5a6"}
}

# --- Routes ---

# Login
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    users = load_users()
    if username in users and users[username] == password:
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/", status_code=303)

# Register
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    users = load_users()
    if username in users:
        return RedirectResponse("/register", status_code=303)
    users[username] = password
    save_users(users)
    return RedirectResponse("/", status_code=303)

# Dashboard
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# --- Training ---
@app.get("/train_stream")
def train_stream():
    def event_generator():
        global model, class_names
        data_dir = "dataset"
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=transform)
        class_names = train_dataset.classes
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        model = SimpleCNN(num_classes=len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 30

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Accuracy calculation
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            progress = int(((epoch+1)/num_epochs)*100)

            msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Progress: {progress}%"
            yield f"data: {msg}\n\n"
            time.sleep(0.2)

        torch.save(model.state_dict(), "best_model.pth")
        yield "data: Training completed!\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Prediction ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, class_names
    if model is None:
        return {"error": "Model not trained"}

    image = Image.open(BytesIO(await file.read())).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    label = class_names[predicted.item()]
    friendly = label_map.get(label, {"friendly": label, "color": "#000000"})
    return {"class": label, "friendly": friendly["friendly"], "color": friendly["color"]}
