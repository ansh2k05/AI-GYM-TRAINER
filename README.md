# 🏋️ AI Gym Trainer & Biomechanics Tracker

An intelligent, real-time computer vision fitness coach built with **MediaPipe**, **OpenCV**, **Streamlit**, and **Google Gemini 2.5 Flash**. Tracks repetitions, analyzes movement mechanics, validates exercise form, and provides live biomechanical feedback.

---

## ✨ Features

- 🎯 **Multi-Exercise Tracking**:
  - **Dumbbell Bicep Curls**: Real-time elbow angle kinematics, concentric/eccentric detection, and elbow flare prevention.
  - **Pec Dec Flyes**: Horizontal adduction tracking, peak contraction hold validation, and arm crossover analysis.
  - **Overhead Shoulder Press**: Vertical deltoid trajectory, lockout angle calculation, and symmetry checks.
- 🔋 **Daily Goal Battery Tracker**: Visual rep battery meter tracking cumulative progress toward daily workout goals.
- ⚡ **Goku Ultra Instinct Celebration**: Custom audiovisual celebration sequence triggered upon reaching daily exercise milestones.
- 🤖 **Gemini-Powered AI Coach**:
  - Biomechanical form analysis and cadence critiques.
  - Interactive fitness and recovery Q&A chat.
  - Intelligent multi-model cascade with offline fallback when no API key is provided.
- 🖥️ **Dual Modes**:
  - **Desktop App (OpenCV HUD)**: High-speed, low-latency glassmorphic HUD overlays.
  - **Web App (Streamlit)**: Modern, browser-accessible dashboard with live webcam processing and chat.

---

## 📂 Project Structure

```text
Gym_Trainer/
├── streamlit_app.py     # Browser-based Streamlit application
├── main.py              # Desktop OpenCV dashboard and exercise selector
├── pose_engine.py       # Core pose calculation and drawing engine
├── dumbbell_curl.py     # Dumbbell curl kinematics and tracker
├── pec_dec_fly.py       # Pec dec fly kinematics and tracker
├── shoulder_press.py    # Shoulder press kinematics and tracker
├── daily_tracker.py     # Cumulative workout session state manager
├── hud_ui.py            # Glassmorphic HUD rendering components
├── llm_coach.py         # Google Gemini AI fitness coach integration
├── goku_effect.py       # Milestone video & sound effect handler
├── assets/              # Audio assets for milestone effects
├── requirements.txt     # Python package dependencies
└── .env.example         # Environment template for Gemini API key
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd Gym_Trainer
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Gemini AI Coach
Copy `.env.example` to `.env` and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> *Note: If no API key is provided, the application automatically runs in offline rule-based coaching mode.*

---

## 🎮 Running the Application

### Option A: Streamlit Web App (Recommended)
Launch the browser-based dashboard:
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option B: Desktop OpenCV Mode
Run the standalone desktop HUD:
```bash
python main.py
```
**Controls:**
- `1` – Dumbbell Curl
- `2` – Pec Dec Fly
- `3` – Shoulder Press
- `4` / `C` – AI Coach Console
- `R` – Reset today's stats
- `Q` / `ESC` – Quit

---

## 🛠️ Tech Stack

- **Computer Vision**: [MediaPipe](https://developers.google.com/mediapipe), [OpenCV](https://opencv.org/)
- **Frontend / UI**: [Streamlit](https://streamlit.io/), Custom Glassmorphic Canvas Engine
- **Generative AI**: [Google Gemini API](https://ai.google.dev/) (`google-genai`)
- **Scientific Computing**: [NumPy](https://numpy.org/)

---

## 📄 License

This project is licensed under the MIT License.
