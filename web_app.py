"""
web_app.py — Ultra-Fast Real-Time AI Gym Trainer Web Application
================================================================
A high-performance modern web application built on Starlette + Uvicorn + OpenCV + MediaPipe.
Delivers 30-60 FPS live webcam streaming via standard MJPEG (no WebRTC, no plugins, zero permission hassle).
"""

import os
import sys
import time
import json
import threading
import asyncio
import cv2
import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import core workout & tracker modules
from daily_tracker import (
    load_daily_stats,
    record_exercise_reps,
    reset_daily_stats,
    get_daily_summary,
    DEFAULT_TARGET,
)
from pose_engine import (
    process_curl_frame,
    process_pec_dec_frame,
    process_shoulder_press_frame,
    _MP,
)
from dumbbell_curl import ArmCurlTracker
from pec_dec_fly import PecDecTracker
from shoulder_press import ShoulderPressTracker
from llm_coach import get_coach_status, ask_fitness_coach, generate_daily_brief

# ─────────────────────────────────────────────
#  Global Streamer & State Manager
# ─────────────────────────────────────────────

class CameraStreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera_index = 0
        self.cap = None
        self.is_running = True
        self.active_exercise = "curl"
        self.target_reps = 10
        self.latest_frame_bytes = None
        self.stop_requested = False

        # MediaPipe Pose instance (model_complexity=1 is pre-downloaded)
        self.pose = _MP.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55, model_complexity=1)

        # Trackers
        self.left_curl_tracker = ArmCurlTracker("LEFT", self.target_reps)
        self.right_curl_tracker = ArmCurlTracker("RIGHT", self.target_reps)
        self.pec_tracker = PecDecTracker(self.target_reps)
        self.shoulder_tracker = ShoulderPressTracker(self.target_reps)
        self.prev_elbows = {"left": None, "right": None}

        self._open_camera()

        # Dedicated background capture worker thread
        self.worker = threading.Thread(target=self._capture_worker, daemon=True)
        self.worker.start()

    def _open_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

    def set_camera_index(self, index: int):
        with self.lock:
            self.camera_index = index
            self._open_camera()

    def set_exercise(self, ex_id: str):
        with self.lock:
            if ex_id in ["curl", "pec_dec", "shoulder_press"]:
                self.active_exercise = ex_id
                self.prev_elbows = {"left": None, "right": None}

    def get_current_reps(self) -> int:
        if self.active_exercise == "curl":
            return max(self.left_curl_tracker.rep_count, self.right_curl_tracker.rep_count)
        elif self.active_exercise == "pec_dec":
            return self.pec_tracker.rep_count
        else:
            return self.shoulder_tracker.rep_count

    def reset_trackers(self):
        with self.lock:
            self.left_curl_tracker = ArmCurlTracker("LEFT", self.target_reps)
            self.right_curl_tracker = ArmCurlTracker("RIGHT", self.target_reps)
            self.pec_tracker = PecDecTracker(self.target_reps)
            self.shoulder_tracker = ShoulderPressTracker(self.target_reps)
            self.prev_elbows = {"left": None, "right": None}

    def _capture_worker(self):
        """Dedicated background thread continuously capturing frames at native webcam rate."""
        while not self.stop_requested:
            if not self.is_running or self.cap is None or not self.cap.isOpened():
                standby = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(standby, "CAMERA PAUSED", (180, 240),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 200, 255), 2)
                _, buffer = cv2.imencode('.jpg', standby)
                with self.lock:
                    self.latest_frame_bytes = buffer.tobytes()
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            with self.lock:
                ex = self.active_exercise
                try:
                    if ex == "curl":
                        rgb_frame, self.prev_elbows = process_curl_frame(
                            self.pose, frame,
                            self.left_curl_tracker,
                            self.right_curl_tracker,
                            self.prev_elbows
                        )
                    elif ex == "pec_dec":
                        rgb_frame = process_pec_dec_frame(self.pose, frame, self.pec_tracker)
                    else:
                        rgb_frame = process_shoulder_press_frame(self.pose, frame, self.shoulder_tracker)

                    bgr_out = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    bgr_out = frame

                _, buffer = cv2.imencode('.jpg', bgr_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.latest_frame_bytes = buffer.tobytes()

            time.sleep(0.005)  # Lightweight yield

    def get_latest_frame(self) -> bytes | None:
        with self.lock:
            return self.latest_frame_bytes

streamer = CameraStreamManager()

# ─────────────────────────────────────────────
#  API Endpoints
# ─────────────────────────────────────────────

async def endpoint_video_feed(request):
    """Streams live MJPEG frames asynchronously without blocking the event loop."""
    async def frame_stream():
        while True:
            frame = streamer.get_latest_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.033)  # Solid 30 FPS non-blocking!

    return StreamingResponse(
        frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

async def endpoint_stats(request):
    """Returns real-time session reps and today's cumulative workout stats."""
    daily_stats = load_daily_stats()
    summary = get_daily_summary(daily_stats)
    live_reps = streamer.get_current_reps()
    coach_info = get_coach_status()

    return JSONResponse({
        "active_exercise": streamer.active_exercise,
        "live_reps": live_reps,
        "is_camera_running": streamer.is_running,
        "camera_index": streamer.camera_index,
        "summary": summary,
        "daily_stats": daily_stats,
        "coach": coach_info,
    })

async def endpoint_set_exercise(request):
    """Switch active exercise on the fly."""
    data = await request.json()
    ex_id = data.get("exercise")
    if ex_id in ["curl", "pec_dec", "shoulder_press"]:
        streamer.set_exercise(ex_id)
        return JSONResponse({"success": True, "active_exercise": streamer.active_exercise})
    return JSONResponse({"success": False, "error": "Invalid exercise ID"}, status_code=400)

async def endpoint_save_reps(request):
    """Save currently counted reps to today's cumulative workout log."""
    reps = streamer.get_current_reps()
    ex_id = streamer.active_exercise
    if reps > 0:
        record_exercise_reps(ex_id, reps)
        streamer.reset_trackers()
        return JSONResponse({"success": True, "saved_reps": reps, "exercise": ex_id})
    return JSONResponse({"success": False, "message": "No reps counted yet to save."})

async def endpoint_reset_stats(request):
    """Reset all cumulative reps for today."""
    reset_daily_stats()
    streamer.reset_trackers()
    return JSONResponse({"success": True})

async def endpoint_camera_control(request):
    """Toggle camera pause/resume or change camera index."""
    data = await request.json()
    action = data.get("action")
    if action == "toggle":
        streamer.is_running = not streamer.is_running
    elif action == "set_device":
        dev_idx = int(data.get("index", 0))
        streamer.set_camera_index(dev_idx)
    return JSONResponse({"success": True, "is_running": streamer.is_running, "index": streamer.camera_index})

async def endpoint_chat(request):
    """Chat with the Gemini AI Fitness & Nutrition Coach."""
    data = await request.json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return JSONResponse({"reply": "Please ask a question!"}, status_code=400)

    try:
        reply = ask_fitness_coach(user_msg)
    except Exception as e:
        reply = f"AI Coach temporarily unavailable: {e}"

    return JSONResponse({"reply": reply})

async def endpoint_daily_brief(request):
    """Fetch AI personalized daily brief."""
    stats = load_daily_stats()
    brief = generate_daily_brief(stats)
    return JSONResponse({"brief": brief})

# ─────────────────────────────────────────────
#  Modern HTML5 Dashboard
# ─────────────────────────────────────────────

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Gym Trainer — Live Vision Suite</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #07090e;
            --bg-surface: #0e121b;
            --bg-glass: rgba(18, 23, 36, 0.72);
            --border-glass: rgba(255, 255, 255, 0.08);
            --accent-green: #00ff88;
            --accent-cyan: #00e5ff;
            --accent-gold: #ffd700;
            --accent-orange: #ff9100;
            --accent-red: #ff3366;
            --text-primary: #f0f4fc;
            --text-secondary: #8a96aa;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 10% 15%, rgba(0, 229, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 85%, rgba(0, 255, 136, 0.05) 0%, transparent 40%);
            display: flex;
            flex-direction: column;
        }

        /* ── Header ── */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.1rem 2rem;
            background: var(--bg-glass);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-glass);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .brand-icon {
            font-size: 1.8rem;
            filter: drop-shadow(0 0 12px rgba(0, 255, 136, 0.4));
        }

        .brand-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #fff 30%, #8a96aa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-badges {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .badge {
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border: 1px solid transparent;
        }

        .badge-live {
            background: rgba(255, 51, 102, 0.15);
            color: #ff5277;
            border-color: rgba(255, 51, 102, 0.3);
        }

        .badge-pulse {
            width: 8px;
            height: 8px;
            background: #ff3366;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(255, 51, 102, 0.7); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(255, 51, 102, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(255, 51, 102, 0); }
        }

        .badge-ai {
            background: rgba(0, 229, 255, 0.12);
            color: var(--accent-cyan);
            border-color: rgba(0, 229, 255, 0.25);
        }

        /* ── Main Layout ── */
        .app-container {
            display: grid;
            grid-template-columns: 1fr 390px;
            gap: 1.5rem;
            padding: 1.5rem 2rem;
            flex: 1;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        @media (max-width: 1080px) {
            .app-container {
                grid-template-columns: 1fr;
            }
        }

        /* ── Left Section: Video Room ── */
        .video-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-glass);
            border-radius: 20px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }

        .video-header {
            padding: 1rem 1.4rem;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-glass);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .exercise-tabs {
            display: flex;
            gap: 0.5rem;
            background: rgba(0, 0, 0, 0.35);
            padding: 0.3rem;
            border-radius: 12px;
            border: 1px solid var(--border-glass);
        }

        .ex-tab {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .ex-tab:hover {
            color: var(--text-primary);
        }

        .ex-tab.active {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 229, 255, 0.2));
            color: #fff;
            border: 1px solid rgba(0, 255, 136, 0.4);
            box-shadow: 0 0 12px rgba(0, 255, 136, 0.2);
        }

        .cam-controls {
            display: flex;
            gap: 0.6rem;
            align-items: center;
        }

        select.cam-select {
            background: #141926;
            color: #d0d7e5;
            border: 1px solid var(--border-glass);
            padding: 0.45rem 0.8rem;
            border-radius: 8px;
            font-size: 0.8rem;
            outline: none;
            cursor: pointer;
        }

        /* ── Live Video Feed ── */
        .video-viewport {
            position: relative;
            background: #030407;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 480px;
        }

        .video-viewport img {
            width: 100%;
            height: auto;
            max-height: 640px;
            object-fit: contain;
            display: block;
        }

        /* ── Video Footer Action Bar ── */
        .video-footer {
            padding: 1.1rem 1.4rem;
            background: rgba(255, 255, 255, 0.02);
            border-top: 1px solid var(--border-glass);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }

        .live-counter-pill {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
        }

        .live-counter-pill .number {
            font-family: 'Outfit', sans-serif;
            font-size: 2.3rem;
            font-weight: 900;
            color: var(--accent-gold);
            line-height: 1;
        }

        .live-counter-pill .label {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .btn {
            padding: 0.65rem 1.4rem;
            border-radius: 10px;
            font-size: 0.88rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid transparent;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary {
            background: linear-gradient(135deg, #00c868, #00994d);
            color: #fff;
            border-color: #00ff88;
            box-shadow: 0 4px 16px rgba(0, 255, 136, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 255, 136, 0.45);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-primary);
            border-color: var(--border-glass);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        /* ── Right Section: Sidebar Hub ── */
        .side-hub {
            display: flex;
            flex-direction: column;
            gap: 1.4rem;
        }

        .panel-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-glass);
            border-radius: 18px;
            padding: 1.3rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* ── Battery HUD Stamina Widget ── */
        .battery-box {
            background: #080a10;
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .battery-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .battery-shell {
            display: flex;
            align-items: center;
            width: 100%;
        }

        .battery-body {
            flex: 1;
            height: 24px;
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid var(--accent-green);
            border-radius: 6px;
            padding: 2px;
            position: relative;
            overflow: hidden;
        }

        .battery-fill {
            height: 100%;
            background: linear-gradient(90deg, #00c878, #00ff88);
            border-radius: 3px;
            width: 0%;
            transition: width 0.4s ease;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.6);
        }

        .battery-nub {
            width: 5px;
            height: 12px;
            background: var(--accent-green);
            border-radius: 0 3px 3px 0;
            margin-left: 2px;
        }

        /* ── Exercise Status Rows ── */
        .ex-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.7rem 0.8rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-glass);
            border-radius: 10px;
            margin-bottom: 0.6rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .ex-row:hover {
            border-color: rgba(0, 229, 255, 0.3);
            background: rgba(0, 229, 255, 0.03);
        }

        .ex-row.active-row {
            border-color: var(--accent-green);
            background: rgba(0, 255, 136, 0.06);
        }

        .ex-row-name {
            font-size: 0.88rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .ex-status-badge {
            font-size: 0.75rem;
            font-weight: 800;
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
        }

        .status-done {
            background: rgba(0, 255, 136, 0.15);
            color: var(--accent-green);
        }

        .status-left {
            background: rgba(255, 160, 0, 0.15);
            color: #ffaa33;
        }

        /* ── AI Coach Drawer ── */
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 280px;
            background: #090b12;
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            overflow: hidden;
        }

        .chat-messages {
            flex: 1;
            padding: 0.8rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .chat-bubble {
            padding: 0.6rem 0.85rem;
            border-radius: 12px;
            max-width: 90%;
        }

        .chat-ai {
            background: rgba(0, 229, 255, 0.08);
            border: 1px solid rgba(0, 229, 255, 0.18);
            color: #dce5f8;
            align-self: flex-start;
        }

        .chat-user {
            background: rgba(0, 255, 136, 0.15);
            border: 1px solid rgba(0, 255, 136, 0.3);
            color: #ffffff;
            align-self: flex-end;
        }

        .chat-input-bar {
            display: flex;
            padding: 0.5rem;
            background: #111422;
            border-top: 1px solid var(--border-glass);
            gap: 0.4rem;
        }

        .chat-input-bar input {
            flex: 1;
            background: #090b12;
            border: 1px solid var(--border-glass);
            color: #fff;
            padding: 0.5rem 0.8rem;
            border-radius: 8px;
            font-size: 0.82rem;
            outline: none;
        }

        .chat-input-bar button {
            background: var(--accent-cyan);
            border: none;
            color: #040810;
            font-weight: 800;
            padding: 0.5rem 0.9rem;
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="brand">
            <span class="brand-icon">🏋️</span>
            <div>
                <div class="brand-title">AI GYM TRAINER</div>
                <div style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 600;">HIGH-FPS COMPUTER VISION SUITE</div>
            </div>
        </div>

        <div class="header-badges">
            <span class="badge badge-live">
                <span class="badge-pulse"></span>
                <span>DIRECT 30 FPS LIVE CAM</span>
            </span>
            <span class="badge badge-ai" id="badgeCoachStatus">
                ★ GEMINI AI ONLINE
            </span>
        </div>
    </header>

    <!-- App Body -->
    <div class="app-container">

        <!-- Left: Live Video Feed Section -->
        <div class="video-card">
            <div class="video-header">
                <!-- Exercise Switcher -->
                <div class="exercise-tabs">
                    <button class="ex-tab active" id="tab_curl" onclick="selectExercise('curl')">💪 Dumbbell Curl</button>
                    <button class="ex-tab" id="tab_pec_dec" onclick="selectExercise('pec_dec')">🦋 Pec Dec Fly</button>
                    <button class="ex-tab" id="tab_shoulder_press" onclick="selectExercise('shoulder_press')">🔱 Shoulder Press</button>
                </div>

                <!-- Camera Controls -->
                <div class="cam-controls">
                    <select class="cam-select" id="camSelect" onchange="changeCameraDevice(this.value)">
                        <option value="0">Camera 0 (Default)</option>
                        <option value="1">Camera 1 (USB/External)</option>
                        <option value="2">Camera 2</option>
                    </select>
                    <button class="btn btn-secondary" style="padding: 0.45rem 0.8rem; font-size: 0.8rem;" onclick="toggleCamera()">⏸ Pause</button>
                </div>
            </div>

            <!-- Video Frame Container -->
            <div class="video-viewport">
                <img id="liveFeed" src="/video_feed" alt="Live AI Gym Camera Feed" onerror="handleStreamError()">
            </div>

            <!-- Footer Action Bar -->
            <div class="video-footer">
                <div class="live-counter-pill">
                    <span class="number" id="liveRepCount">0</span>
                    <span class="label" id="liveExerciseLabel">Curls This Session</span>
                </div>

                <div style="display: flex; gap: 0.6rem;">
                    <button class="btn btn-secondary" onclick="resetTodayStats()">🔄 Reset Day</button>
                    <button class="btn btn-primary" onclick="saveReps()">💾 Save Reps to Workout</button>
                </div>
            </div>
        </div>

        <!-- Right: Daily Tracker & AI Coach Hub -->
        <div class="side-hub">

            <!-- Today's Progress Card -->
            <div class="panel-card">
                <div class="card-title">
                    <span>⚡ DAILY STAMINA</span>
                    <span id="staminaPct" style="color: var(--accent-green); font-size: 0.95rem;">0%</span>
                </div>

                <!-- Battery HUD Widget -->
                <div class="battery-box">
                    <div class="battery-header">
                        <span>REPS COMPLETED</span>
                        <span id="batteryNumbers">0 / 30 REPS</span>
                    </div>
                    <div class="battery-shell">
                        <div class="battery-body">
                            <div class="battery-fill" id="batteryFill"></div>
                        </div>
                        <div class="battery-nub"></div>
                    </div>
                </div>

                <!-- Exercise List Status -->
                <div id="exerciseList">
                    <!-- Populated dynamically -->
                </div>
            </div>

            <!-- AI Coach Interactive Drawer -->
            <div class="panel-card" style="flex: 1; display: flex; flex-direction: column;">
                <div class="card-title">
                    <span>🤖 AI FITNESS & DIET COACH</span>
                    <span style="font-size: 0.72rem; color: var(--accent-cyan); font-weight: 700;">UNLIMITED CHATS</span>
                </div>

                <div class="chat-container">
                    <div class="chat-messages" id="chatLogs">
                        <div class="chat-bubble chat-ai">
                            👋 Hi! I'm your AI Biomechanics & Nutrition Coach. Perform your reps in front of the camera, or ask me about posture, tempo, or high-protein diet plans!
                        </div>
                    </div>
                    <div class="chat-input-bar">
                        <input type="text" id="chatInput" placeholder="Ask AI coach a question..." onkeydown="if(event.key==='Enter') sendChatMessage()">
                        <button onclick="sendChatMessage()">Send</button>
                    </div>
                </div>
            </div>

        </div>

    </div>

    <!-- Frontend Logic -->
    <script>
        let currentExercise = "curl";

        const exerciseNames = {
            "curl": "Dumbbell Curl",
            "pec_dec": "Pec Dec Fly",
            "shoulder_press": "Shoulder Press"
        };

        // Switch exercise via API
        async function selectExercise(exId) {
            currentExercise = exId;
            document.querySelectorAll('.ex-tab').forEach(b => b.classList.remove('active'));
            const activeTab = document.getElementById('tab_' + exId);
            if (activeTab) activeTab.classList.add('active');

            try {
                await fetch('/api/set_exercise', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ exercise: exId })
                });
            } catch (e) {
                console.error("Failed to switch exercise:", e);
            }
            fetchStats();
        }

        // Toggle Camera
        async function toggleCamera() {
            await fetch('/api/camera/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'toggle' })
            });
        }

        // Change Camera Device
        async function changeCameraDevice(idx) {
            await fetch('/api/camera/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'set_device', index: parseInt(idx) })
            });
        }

        // Save Current Reps
        async function saveReps() {
            const res = await fetch('/api/save_reps', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                alert(`✅ Successfully saved ${data.saved_reps} reps to today's workout!`);
                fetchStats();
            } else {
                alert(data.message || "No reps recorded yet!");
            }
        }

        // Reset Today's Stats
        async function resetTodayStats() {
            if (confirm("Are you sure you want to reset today's workout reps?")) {
                await fetch('/api/reset_stats', { method: 'POST' });
                fetchStats();
            }
        }

        // Polling stats & live rep counter
        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();

                // Live session rep count
                document.getElementById('liveRepCount').innerText = data.live_reps;
                document.getElementById('liveExerciseLabel').innerText = (exerciseNames[data.active_exercise] || "Exercise") + " Reps";

                // Daily Battery Stamina
                const done = data.summary.total_done;
                const target = data.summary.total_target;
                const pct = Math.min(100, Math.round(data.summary.percentage * 100));

                document.getElementById('batteryNumbers').innerText = `${done} / ${target} REPS`;
                document.getElementById('staminaPct').innerText = `${pct}%`;
                document.getElementById('batteryFill').style.width = `${pct}%`;

                // Update Exercise rows
                const exList = document.getElementById('exerciseList');
                const exData = data.daily_stats.exercises || {};
                const targetPerEx = data.daily_stats.target_per_exercise || 10;

                const exMeta = [
                    { id: 'curl', name: 'Dumbbell Curl', icon: '💪' },
                    { id: 'pec_dec', name: 'Pec Dec Fly', icon: '🦋' },
                    { id: 'shoulder_press', name: 'Shoulder Press', icon: '🔱' }
                ];

                exList.innerHTML = exMeta.map(ex => {
                    const rowStat = exData[ex.id] || { reps: 0, completed: false };
                    const reps = rowStat.reps || 0;
                    const isDone = rowStat.completed || reps >= targetPerEx;
                    const isCurrent = data.active_exercise === ex.id;

                    const badge = isDone 
                        ? '<span class="ex-status-badge status-done">✅ DONE (10/10)</span>'
                        : `<span class="ex-status-badge status-left">⏳ ${reps}/${targetPerEx}</span>`;

                    return `
                        <div class="ex-row ${isCurrent ? 'active-row' : ''}" onclick="selectExercise('${ex.id}')">
                            <span class="ex-row-name">${ex.icon} ${ex.name}</span>
                            ${badge}
                        </div>
                    `;
                }).join('');

            } catch (e) {
                console.warn("Stats fetch failed:", e);
            }
        }

        // AI Coach Chat
        async function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const msg = input.value.trim();
            if (!msg) return;

            const chatLogs = document.getElementById('chatLogs');
            chatLogs.innerHTML += `<div class="chat-bubble chat-user">${msg}</div>`;
            input.value = '';
            chatLogs.scrollTop = chatLogs.scrollHeight;

            chatLogs.innerHTML += `<div class="chat-bubble chat-ai" id="aiLoading">Thinking...</div>`;
            chatLogs.scrollTop = chatLogs.scrollHeight;

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                document.getElementById('aiLoading').remove();
                chatLogs.innerHTML += `<div class="chat-bubble chat-ai">${data.reply.replace(/\\n/g, '<br>')}</div>`;
            } catch (e) {
                document.getElementById('aiLoading').remove();
                chatLogs.innerHTML += `<div class="chat-bubble chat-ai">⚠️ Could not connect to AI Coach.</div>`;
            }
            chatLogs.scrollTop = chatLogs.scrollHeight;
        }

        function handleStreamError() {
            console.warn("Video stream reconnecting...");
            setTimeout(() => {
                document.getElementById('liveFeed').src = '/video_feed?t=' + new Date().getTime();
            }, 1000);
        }

        // Poll stats every 1 second
        setInterval(fetchStats, 1000);
        fetchStats();
    </script>
</body>
</html>
"""

async def endpoint_home(request):
    return HTMLResponse(HTML_DASHBOARD)

# Starlette app routes
routes = [
    Route("/", endpoint_home),
    Route("/video_feed", endpoint_video_feed),
    Route("/api/stats", endpoint_stats),
    Route("/api/set_exercise", endpoint_set_exercise, methods=["POST"]),
    Route("/api/save_reps", endpoint_save_reps, methods=["POST"]),
    Route("/api/reset_stats", endpoint_reset_stats, methods=["POST"]),
    Route("/api/camera/control", endpoint_camera_control, methods=["POST"]),
    Route("/api/chat", endpoint_chat, methods=["POST"]),
    Route("/api/daily_brief", endpoint_daily_brief),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 AI Gym Trainer Live Suite running on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
