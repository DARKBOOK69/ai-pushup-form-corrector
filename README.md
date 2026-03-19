# AI Pushup Form Corrector

A real-time AI fitness system that detects pushup form correctness and counts repetitions using Computer Vision and Machine Learning.

---

## 🚀 Features

* Real-time human pose detection using MediaPipe
* Machine Learning-based form classification (Random Forest)
* Rep counting system using joint angle tracking
* Smart feedback system (Good form / Fix form / Go lower / Keep body straight)
* Voice feedback for each rep
* Temporal smoothing for stable predictions
* Position gating to avoid false detections

---

## 🧠 Key Innovation

Unlike traditional rule-based systems, this project uses a trained Machine Learning model to classify pushup form based on biomechanical features such as:

* Arm angle
* Body alignment angle
* Hip offset

The model achieves ~88% accuracy and performs reliably in real-time conditions with noisy pose data.

---

## ⚙️ Challenges Solved

* **Noisy pose detection** → handled using moving average smoothing
* **Flickering predictions** → solved using temporal buffering (deque)
* **False detections (e.g., sitting)** → fixed using posture gating
* **Inconsistent rep counting** → stabilized with threshold tuning

---

## 🧪 How It Works

1. Capture video using OpenCV
2. Detect body landmarks using MediaPipe
3. Calculate biomechanical features:

   * Arm angle
   * Body angle
   * Hip offset
4. Feed features into trained ML model
5. Apply smoothing to stabilize predictions
6. Generate real-time feedback and count reps

---

## 🛠 Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* Scikit-learn
* Joblib
* pyttsx3 (for voice feedback)

---

## 📊 Model Performance

* Accuracy: ~88%
* Balanced classification for both correct and incorrect form

### Confusion Matrix:

[[138  30]
[ 23 247]]

---

## 🎥 Demo



## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-pushup-form-corrector.git
cd ai-pushup-form-corrector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

---

## 📁 Project Structure

```
ai-pushup-form-corrector/
│── main.py
│── pushup_model.pkl
│── requirements.txt
│── README.md
```

---

## 🚀 Future Improvements

* Support for multiple exercises (squats, curls, etc.)
* Exercise auto-detection using ML
* Performance dashboard (tracking progress over time)
* Mobile/web deployment

---

## 📌 Author

Developed by Shivansh Joshi

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
