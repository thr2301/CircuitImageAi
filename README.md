[README.md](https://github.com/user-attachments/files/22097510/README.md)
# 🔌 Circuit AI – Circuit Classification Web App  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?logo=pytorch)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

A deep learning web application built with **PyTorch** + **FastAPI** that can **train, evaluate, and predict electronic circuits** (Amplifier, RC Highpass, RC Lowpass, Resonator, Other) directly from a **user-friendly dashboard** , calculates the values of the passive elements and creates a netlist.  

---

## 🚀 Features  

-  **Dashboard-style UI** with plain HTML/CSS/JS  
-  **User Authentication** (Login & Register)  
-  **Upload Images** for circuit prediction  
-  **Live Training** with real-time **loss & accuracy per epoch**  
-  Organized dataset structure for easy training/testing
-  **Passive elements values calculation**
-  **Netlist creation**

---

## 📂 Project Structure  

```
CircuitImageAi/
│── app.py                # FastAPI backend
│── static/
│   ├── images.jpg        # Background image
│   ├── login.html        # Login & Register UI
│   ├── style.css         # Styling
│   └── script.js         # Frontend logic
│── templates/
│   ├── login.html        # Login UI
│   ├── dashboard.html    # Dashboard UI
│   └── register.html     # Register UI
│── dataset/
│   ├── train/
│      ├── amplifier/
│      ├── rc_lp/
│      ├── rc_hp/
│      ├── resonator/
│      └── other/
│── models/               # Here are saved some previous trainings
│── images/               # Some images for testing 
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
   ├── amplifier/   # training images
   ├── rc_lp/
   ├── rc_hp/
   ├── resonator/
   └── other/

```

⚠️ Place **at least 20–30 images per class** in `train/` and **2-5 images per class** in `test/` for decent results.
Insisde the helpers folder there are to subfolders with some datasets , one big and one small.

---

### Configure the Database
The credentilas are stored in a json file that is called users.json .

## 🧑‍💻 Usage  

1. **Register/Login**  
   - Register a new account or login with existing credentials.
   - <img width="1920" height="1041" alt="Image" src="https://github.com/user-attachments/assets/e0d334a6-8d8a-4b2d-a67e-7bde5030dab5" />
   - Or login with existing credentials.
   - <img width="1920" height="1043" alt="Image" src="https://github.com/user-attachments/assets/b235ebe1-f45a-40f9-8283-c0f6900661e0" />

2. **Training**  
   - Start training or continue a previous one from the dashboard.
   - <img width="1920" height="913" alt="Image" src="https://github.com/user-attachments/assets/df13d68a-2253-47af-950b-78438c638c82" />
   - Monitor **loss & accuracy per epoch** in real time.
   - Or you can load a previous training.

3. **Prediction**  
   - Upload an image of a circuit.
   - Get the **predicted class + confidence percentage** instantly.
   - In case of filter :
   - Enter the cutoff frequency.
   - Click Compute & plot
   - Fill the Library and cell name
   - Click Download Netlist
   - <img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/dc77afee-6a49-4f2a-aa05-4172d82a2ba6" />

   - In case of resonator
   - Choose vowel
   - Click Load & plot
   - Fill the Library & Cell name
   - Click DownLoad Netlist
   - <img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/a79bd09d-fb58-41dc-8579-96dc753604c9" /> 
     
5. **Settings**
   - Here the User can change his email, phone number or add a new password
   - <img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/35afeb26-29dc-49fe-8580-0af216eb5550" />
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
passlib[bcrypt]
cryptography
```

---

## ✅ To-Do  

- [ ] Add more circuits 



---

