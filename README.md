[README.md](https://github.com/user-attachments/files/22097510/README.md)
# 🔌 Circuit AI – Circuit Classification Web App  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?logo=pytorch)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

A deep learning web application built with **PyTorch** + **FastAPI** that can **train, evaluate, and predict electronic circuits** (Amplifier, RC Highpass, RC Lowpass, Other) directly from a **user-friendly dashboard**.  

---

## 🚀 Features  

- 📊 **Dashboard-style UI** with plain HTML/CSS/JS  
- 🔐 **User Authentication** (Login & Register)  
- 🖼️ **Upload Images** for circuit prediction  
- 🎯 **Live Training** with real-time **loss & accuracy per epoch**  
- 📂 Organized dataset structure for easy training/testing  
- 💾 Model automatically saved as `best_model.pth`  

---

## 📂 Project Structure  

```
CircuitImageAi/
│── app.py                # FastAPI backend
│── model.py              # CNN model definition
│── static/
│   ├── index.html        # Dashboard UI
│   ├── login.html        # Login & Register UI
│   ├── style.css         # Styling
│   └── script.js         # Frontend logic
│── dataset/
│   ├── train/
│   │   ├── amplifier/
│   │   ├── rc_lp/
│   │   ├── rc_hp/
│   │   └── other/
│   └── test/
│       ├── amplifier/
│       ├── rc_lp/
│       ├── rc_hp/
│       └── other/
│── best_model.pth        # Saved model (after training)
│── users.json            # User credentials (auto-created on register)
│── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

## ⚙️ Installation  

1. **Clone the repo**  

```bash
git clone https://github.com/your-username/CircuitImageAi.git
cd CircuitImageAi
```

2. **Install dependencies**  

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App  

Start the FastAPI server:  

```bash
uvicorn app:app --reload
```

Open your browser:  
👉 [http://127.0.0.1:8000/static/login.html](http://127.0.0.1:8000/static/login.html)  

---

## 📊 Dataset  

Organize your dataset as follows:  

```
dataset/
├── train/
│   ├── amplifier/   # training images
│   ├── rc_lp/
│   ├── rc_hp/
│   └── other/
└── test/
    ├── amplifier/   # testing images
    ├── rc_lp/
    ├── rc_hp/
    └── other/
```

⚠️ Place **at least 20–30 images per class** in `train/` and **2-5 images per class** in `test/` for decent results.  

---

## 🧑‍💻 Usage  

1. **Register/Login**  
   - Register a new account or login with existing credentials.  

2. **Training**  
   - Start training from the dashboard.  
   - Monitor **loss & accuracy per epoch** in real time.  

3. **Prediction**  
   - Upload an image of a circuit.  
   - Get the **predicted class + confidence percentage** instantly.  

---

## 📌 Requirements  

Add these to **requirements.txt**:  

```
fastapi
uvicorn
torch
torchvision
pillow
python-multipart
```

---

## ✅ To-Do  

- [ ] Add more circuits 
- [ ] Add prediction for the values of the circuit components 

---

## 🖼️ Demo Screenshots  

- **Login Page:** `screenshots/login.png`  
- **Training in Progress:** `screenshots/training.png`  
- **Prediction Result:** `screenshots/prediction.png`  

*(Create a `screenshots/` folder in your repo and add your PNG images there)*  
