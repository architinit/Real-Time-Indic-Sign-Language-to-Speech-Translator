# 🤟 Real Time Indic Sign Language to Speech Translator

> A B.Tech Final Year Project that translates **Indian Sign Language (ISL)** gestures into spoken sentences in **7 Indian languages** — in real time, using just a webcam.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?logo=google)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🌟 What It Does

The system captures live webcam video, extracts body, hand, and face landmarks using **MediaPipe Holistic**, and feeds 30-frame sequences into a **Bidirectional LSTM** model to classify Indian Sign Language gestures. Recognised signs are assembled into a sentence, refined by **Groq LLaMA** for grammar, and spoken aloud via **gTTS** in the user's chosen language.

```
Webcam → MediaPipe Landmarks → BiLSTM → Vote Confirmation → LLM Grammar → gTTS Speech
```

---

## ✨ Features

- 🎯 **50 ISL word classes** — pronouns, family, professions, emotions, places, greetings, and more
- 🗣 **7 Indian languages** — English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati
- ⚡ **Real-time recognition** — MediaPipe Holistic at up to 30 fps, server-side
- 🧠 **BiLSTM deep learning** — trained on personal ISL recordings with augmentation
- 🗳 **Vote-based stability** — majority vote over 6 frames before confirming a word
- ✍️ **AI grammar correction** — Groq LLaMA 3.1 converts ISL gloss into natural sentences
- 🌐 **Browser UI** — full web interface served by Flask, no plugins needed

---

## 🗂️ Project Structure

```
Major_Project_ISL/
├── app.py                          # Flask backend — camera, inference, MJPEG stream
├── requirements.txt
├── Frontend/
│   ├── index.html                  # Web UI
│   ├── styles.css
│   └── app.js
├── scripts/
│   ├── sentence_builder.py         # Standalone desktop ISL sentence builder
│   ├── word_tester.py              # Test all 50 signs with reference video panel
│   ├── personal_collector.py       # Record your own ISL training data
│   ├── data_collection.py          # Landmark extraction from raw videos
│   ├── video_processor.py          # Video preprocessing utilities
│   ├── model_training.py           # Train the BiLSTM model
│   ├── train_personal_only.py      # Train on personal recordings only
│   ├── augment_personal_data.py    # Data augmentation
│   ├── fine_tune.py                # Fine-tune on personal data
│   └── evaluation_report.py       # Model evaluation
├── models/
│   └── isl_model_solo.keras        # Trained BiLSTM model (active)
├── mediapipe_models/
│   ├── holistic_landmarker.task    # MediaPipe Holistic model
│   └── *.tflite                    # Component models
├── data/
│   └── extracted_features/         # Label folders (50 word classes)
└── reports/
    └── ISL_Dataset_Exploration_Report.pdf
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.12
- A webcam
- Windows / macOS / Linux

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/real-time-indic-sign-language-to-speech-translator.git
cd real-time-indic-sign-language-to-speech-translator
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option A — Web App (Recommended)

Runs the full browser-based interface with a live camera feed.

```bash
python app.py
```

Open **`http://localhost:5000`** in your browser, then click **Start Camera** to begin signing.

- Detected sign appears live in the right panel
- Words accumulate into a sentence automatically
- Click **Confirm** to finalise and **Speak** to hear it in your chosen language

### Option B — Desktop Script

A standalone OpenCV window with the same recognition pipeline.

```bash
python scripts/sentence_builder.py
```

| Key | Action |
|---|---|
| `ENTER` | Finalise sentence and speak |
| `B` | Undo last word |
| `C` | Clear sentence |
| `1–7` | Switch output language (En/Hi/Bn/Ta/Te/Mr/Gu) |
| `Q` | Quit |

### Option C — Word Tester

Test all 50 signs one by one with a reference video side-by-side.

```bash
python scripts/word_tester.py
```

---

## 🔑 API Key (Groq)

The project uses **Groq LLaMA** for AI grammar correction.

1. Get a free key at [console.groq.com](https://console.groq.com)
2. Replace `GROQ_API_KEY` in `app.py` and `scripts/sentence_builder.py`

> If Groq is unavailable the system automatically falls back to a rule-based grammar engine — the app still works fully.

---

## 🧠 Model & Training

The active model (`models/isl_model_solo.keras`) is a **Bidirectional LSTM** trained on personal ISL recordings of 50 word classes.

**Architecture:**
- Input: `(30 frames × 378 features)` — pose + face keypoints + both hand landmarks
- BiLSTM → Dropout → BiLSTM → Dense → Softmax (50 classes)

**To train your own model on your own recordings:**

```bash
# 1. Record your own ISL signs
python scripts/personal_collector.py

# 2. Augment the recorded data
python scripts/augment_personal_data.py

# 3. Train the BiLSTM
python scripts/train_personal_only.py
```

> **Note:** The pre-trained model was trained on the signer's own recordings. For best accuracy, training on your own hand gestures is recommended.

---

## 📊 Supported Signs — 50 Classes

| Category | Signs |
|---|---|
| Pronouns | I, You, He, She, They, We, It |
| Family | Mother, Father, Brother, Sister, Family, Friend |
| Professions | Teacher, Doctor, Police, Student, Man, Woman, Patient |
| Emotions | Happy, Sad, Good, Bad, Strong, Weak, Healthy, Sick, Alive |
| Age | Old, Young, Deaf |
| Greetings | Hello, How are you, Thank you, Good Morning |
| Places | House, Hospital, School |
| Time | Today, Tomorrow, Yesterday, Morning, Night, Time |
| Others | Exercise, Sign, Dream, Sport, Medicine |

---

## 👥 Team

**B.Tech Final Year Project — Computer Science & Engineering**

| Name |
|---|
| Archit Bali |
| Namyaa Sarin |
| Kartik Singh |
| Ansh Patyal |

---

## 🙏 Acknowledgements

- [INCLUDE Dataset](https://zenodo.org/record/4010759) — Indian Sign Language dataset (Zenodo)
- [MediaPipe](https://mediapipe.dev) — Holistic landmark extraction by Google
- [Groq](https://groq.com) — LLaMA 3.1 inference for grammar correction
- [gTTS](https://gtts.readthedocs.io) — Google Text-to-Speech for multilingual output

---

## 📄 License

This project is licensed under the MIT License.
