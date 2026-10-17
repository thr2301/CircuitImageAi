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
│── main.py
│── core/
│   │── __init__.py
│   └── utils.py
│── routers/
│   │── __init__.py
│   ├── auth.py
│   ├── engineering.py 
│   └── ml_routes.py
│── services/
│   │── __init__.py
│   ├── ml_manager.py
│   ├── netlist_exporter.py 
│   └── user_repo.py      
│── static/
│   ├── images.jpg        
│   ├── login.html        
│   ├── style.css         
│   └── script.js       
│── templates/
│   ├── calculator.html         
│   ├── forgot_password.html    
│   ├── login.html             
│   ├── dashboard.html         
│   └── register.html         
│── dataset/
│   ├── train/
│      ├── amplifier/
│      ├── rc_lp/
│      ├── rc_hp/
│      ├── resonator/
│      └── other/
│── models/               
│── images/             
│── requirements.txt      
└── README.md            
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
uvicorn main:app --reload
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
   - <img width="1901" height="902" alt="Image" src="https://github.com/user-attachments/assets/37e947c3-3994-4726-84f4-d684a7fe75a9" />
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
   - <img width="1905" height="900" alt="Image" src="https://github.com/user-attachments/assets/28078761-e8b3-4f95-9630-cf4775e971fd" />

   - In case of resonator
   - Choose vowel
   - Click Load & plot
   - Fill the Library & Cell name
   - Click DownLoad Netlist
   - <img width="1904" height="896" alt="Image" src="https://github.com/user-attachments/assets/853d2b41-a1b0-4e14-a287-8e3ec0a06808" />
   - <img width="1905" height="903" alt="Image" src="https://github.com/user-attachments/assets/02cf0a9c-01ae-4024-83bb-13e76bf110a4" />
     
5. **Settings**
   - Here the User can change his email, phone number or add a new password
   - <img width="1914" height="888" alt="Image" src="https://github.com/user-attachments/assets/9445c1f5-b14b-4ade-8016-92ee48b245b2" />
   - <img width="1896" height="895" alt="Image" src="https://github.com/user-attachments/assets/c53bbb2e-524f-4640-8521-89fcabcff0a7" />

6. **Filter Calculator**
   - <img width="1900" height="895" alt="Image" src="https://github.com/user-attachments/assets/9ed6846c-ef79-4e39-8453-50b35c5b64d3" />
   - <img width="1882" height="893" alt="Image" src="https://github.com/user-attachments/assets/b2318ba3-d88b-4125-94ad-5747c615f166" />
   - <img width="1912" height="902" alt="Image" src="https://github.com/user-attachments/assets/8dfd7643-b8dc-4e4f-be96-6f28e281fd3a" />
---

## 📌 Requirements  

Add these to **requirements.txt**:  

```
fastapi
fastapi-mail
Its dangerous
uvicorn
torch
torchvision
pillow
python-multipart
sqlalchemy
pymysql
passlib[bcrypt]
cryptography
```

---

## ✅ To-Do  

- [ ] Add more circuits
- [ ] Fix netlist production



---

