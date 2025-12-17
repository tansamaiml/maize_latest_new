

# 🌽 Maize Detection Flask API (YOLOv8)

This project exposes **Flask-based REST APIs (POST)** for maize-related computer vision tasks using **YOLOv8** models.

The APIs support:
- 🌱 Plant population counting (image)
- 🌽 Tassel + branch detection & segmentation (image)
- 🎥 Tassel counting from video

All models are loaded **once at startup** for high performance.

---

## 📂 PROJECT STRUCTURE

project/
├── app.py
├── requirements.txt
├── README.md
├── Models/
│   ├── Plant_Population.pt
│   ├── tassel.pt
│   ├── branch.pt
│   └── Tassel_count.pt
├── uploads/
└── outputs/

---

## ⚙️ SYSTEM REQUIREMENTS

- **Python 3.9 (MANDATORY)**
- Windows / Linux / macOS
- NVIDIA GPU (optional)

---

## 🧩 INSTALLATION

### Create virtual environment
```
py -3.11 -m venv maize

```

Activate:
```
maize\Scripts\activate
```

---

### Install dependencies
```
pip install -r requirements.txt
```

---

### (Optional) GPU support
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ▶️ RUN SERVER
```
python app.py
```

Server:
http://localhost:5000

---

## 🔗 API ENDPOINTS (POST)

### Plant Population
POST /plant_population  
Form-Data: image

### Tassel + Branch
POST /tassel_branch_image  
Form-Data: image

### Tassel Video
POST /tassel_video  
Form-Data: video

---

## 🧪 CURL TEST
```
curl -X POST http://localhost:5000/plant_population -F image=@image.jpg
```

---

✅ Flask YOLOv8 API ready for production
