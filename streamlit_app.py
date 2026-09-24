"""
streamlit_app.py — AI Gym Trainer (Streamlit Edition)
======================================================
A browser-based interface for the AI Gym Trainer.

Pages
-----
🏠 Dashboard   – Daily progress overview, exercise status cards, Gemini AI badge
🏋️ Exercise     – Live webcam + MediaPipe pose tracking with rep counter
🤖 AI Coach    – Gemini-powered fitness Q&A chat + daily brief

Run:
    streamlit run streamlit_app.py
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
import streamlit as st

# UTF-8 fix for Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────
#  Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Gym Trainer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS — dark glassmorphism theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0b10;
    color: #e8e8f0;
}
.stApp { background: linear-gradient(135deg, #08090f 0%, #0e101a 50%, #080c14 100%); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f1c 0%, #111320 100%);
    border-right: 1px solid #1e2240;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffd700; }
[data-testid="stSidebar"] p  { color: #9098b8; font-size: 0.88rem; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(8px);
}
[data-testid="stMetricValue"] { color: #ffd700; font-size: 2.1rem; font-weight: 800; }
[data-testid="stMetricLabel"] { color: #9098b8; font-size: 0.82rem; letter-spacing: 0.04em; text-transform: uppercase; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e4d2b, #1a6635);
    border: 1px solid #2dbd6e;
    border-radius: 10px;
    color: #efffef;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.6rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a6b3c, #22874a);
    box-shadow: 0 0 20px rgba(45, 189, 110, 0.35);
    transform: translateY(-1px);
}

/* ── Progress bars ── */
.stProgress > div > div > div { background: linear-gradient(90deg, #1acd6e, #ffd700); border-radius: 99px; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    margin-bottom: 0.6rem;
}

/* ── Section headers ── */
h1 { font-family: 'Outfit', sans-serif; font-weight: 800; letter-spacing: -0.02em; }
h2 { font-family: 'Outfit', sans-serif; font-weight: 600; color: #c8cce8; }
h3 { color: #9098b8; font-size: 0.95rem; letter-spacing: 0.06em; text-transform: uppercase; }

/* ── Exercise status pill ── */
.ex-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.ex-done { border-color: rgba(45, 189, 110, 0.5); background: rgba(45,189,110,0.06); }
.ex-pending { border-color: rgba(230, 80, 60, 0.4); background: rgba(230,80,60,0.04); }

/* ── Stop button (danger) ── */
.stop-btn > button {
    background: linear-gradient(135deg, #4d1e1e, #661a1a) !important;
    border-color: #bd2d2d !important;
    color: #ffe0e0 !important;
}
.stop-btn > button:hover {
    box-shadow: 0 0 20px rgba(189, 45, 45, 0.35) !important;
}

/* ── Gemini badge ── */
.gemini-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0f3d2b, #1a6635);
    border: 1.5px solid #2dbd6e;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #2dff99;
    letter-spacing: 0.05em;
}
.offline-badge {
    display: inline-block;
    background: rgba(100,100,120,0.2);
    border: 1.5px solid #404060;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    color: #9098b8;
    letter-spacing: 0.05em;
}

/* ── Premium Markdown Table Styling (for Diet Plans & Routines) ── */
table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    margin: 1.2rem 0 !important;
    font-size: 0.92rem !important;
    background: rgba(13, 17, 30, 0.9) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
    overflow: hidden !important;
}
thead tr {
    background: linear-gradient(90deg, #152338 0%, #1c2b4c 100%) !important;
}
th {
    padding: 12px 14px !important;
    color: #ffd700 !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    text-align: left !important;
    border-bottom: 2px solid #ffd700 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}
td {
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    color: #e2e8f4 !important;
    line-height: 1.45 !important;
}
tbody tr:last-child td {
    border-bottom: none !important;
}
tbody tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.02) !important;
}
tbody tr:hover {
    background: rgba(255, 215, 0, 0.07) !important;
    transition: background 0.15s ease;
}

/* ── Unlimited Chat Badge ── */
.unlimited-badge {
    display: inline-block;
    background: linear-gradient(135deg, #162a50, #1f3d75);
    border: 1.2px solid #5a9cff;
    border-radius: 999px;
    padding: 0.22rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: #8ec0ff;
    letter-spacing: 0.04em;
    margin-left: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session-state initialisation
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "page":          "dashboard",
        "ex_choice":     "curl",     # 'curl' | 'pec_dec' | 'shoulder_press'
        "tracking":      False,
        "left_tracker":  None,
        "right_tracker": None,
        "tracker":       None,
        "prev_elbows":   {"left": None, "right": None},
        "cap":           None,
        "pose":          None,
        "chat_history":  [],
        "daily_brief":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
#  Imports (lazy to keep startup fast)
# ─────────────────────────────────────────────
import importlib
import daily_tracker
import llm_coach
importlib.reload(llm_coach)

from daily_tracker import (
    load_daily_stats, get_daily_summary, reset_daily_stats, record_exercise_reps
)
from llm_coach import (
    get_coach_status, ask_fitness_coach, generate_daily_brief, generate_diet_plan_table
)


# ─────────────────────────────────────────────
#  Battery Logo HUD Generator
# ─────────────────────────────────────────────
def render_battery_html(reps: int, target: int, label: str = "BATTERY", size: str = "large") -> str:
    """
    Renders an authentic, high-tech glowing Battery Logo widget instead of a flat progress line.
    Color dynamics:
      - Low reps (< 40% of target, e.g. 0-3 reps): Alert RED 🪫
      - Mid reps (40% - 79% of target, e.g. 4-7 reps): Energy ORANGE ⚡
      - Completed (>= 80% / 100%, e.g. 8-10 reps): Glowing Emerald GREEN 🔋
    Includes battery chassis, positive terminal nub (+), internal cell dividers,
    reps count, and charge percentage.
    """
    target = max(1, int(target))
    reps = max(0, int(reps))
    pct = min(1.0, reps / float(target))
    pct_int = int(pct * 100)
    is_met = reps >= target

    if reps < target * 0.4:
        theme_color = "#ff3b4e"
        bg_fill = "linear-gradient(90deg, #d32f2f, #ff3b4e)"
        glow = "0 0 14px rgba(255, 59, 78, 0.45)"
        icon = "🪫"
        state_label = "LOW"
    elif reps < target * 0.8:
        theme_color = "#ff9900"
        bg_fill = "linear-gradient(90deg, #e65100, #ffaa00)"
        glow = "0 0 14px rgba(255, 153, 0, 0.45)"
        icon = "⚡"
        state_label = "CHARGING"
    else:
        theme_color = "#00e676"
        bg_fill = "linear-gradient(90deg, #00b050, #00e676)"
        glow = "0 0 16px rgba(0, 230, 118, 0.55)"
        icon = "🔋"
        state_label = "FULL 100% ★" if is_met else "HIGH"

    if size == "large":
        return f"""
        <div style="background:rgba(18,22,34,0.85); border:1.5px solid {theme_color}; border-radius:14px; padding:0.95rem 1.15rem; margin:0.6rem 0 0.9rem 0; box-shadow:{glow};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span style="font-size:1.35rem;">{icon}</span>
                    <span style="font-weight:700; font-size:1.05rem; color:#f0f2f8; letter-spacing:0.03em;">{label}</span>
                    <span style="background:rgba(255,255,255,0.06); color:{theme_color}; font-size:0.75rem; font-weight:800; padding:0.18rem 0.55rem; border-radius:999px; border:1px solid {theme_color};">[{state_label}]</span>
                </div>
                <div style="text-align:right;">
                    <span style="font-weight:800; font-size:1.2rem; color:{theme_color};">{reps} / {target} Reps</span>
                    <span style="font-weight:700; font-size:0.95rem; color:#a0a6be; margin-left:0.35rem;">({pct_int}%)</span>
                </div>
            </div>
            <!-- Battery Chassis with Terminal Nub -->
            <div style="display:flex; align-items:center; width:100%;">
                <div style="flex:1; height:32px; background:rgba(8,10,16,0.92); border:2.5px solid {theme_color}; border-radius:8px; padding:3px; position:relative; overflow:hidden; box-shadow:inset 0 0 10px rgba(0,0,0,0.85);">
                    <div style="width:{pct_int}%; height:100%; border-radius:4px; background:{bg_fill}; box-shadow:{glow}; transition:width 0.4s ease;"></div>
                    <!-- Cell Divider Ticks (25%, 50%, 75%) -->
                    <div style="position:absolute; top:0; left:25%; width:2px; height:100%; background:rgba(8,10,16,0.7);"></div>
                    <div style="position:absolute; top:0; left:50%; width:2px; height:100%; background:rgba(8,10,16,0.7);"></div>
                    <div style="position:absolute; top:0; left:75%; width:2px; height:100%; background:rgba(8,10,16,0.7);"></div>
                </div>
                <!-- Battery Positive Terminal Cap -->
                <div style="width:7px; height:16px; background:{theme_color}; border-radius:0 4px 4px 0; margin-left:2px; box-shadow:{glow};"></div>
            </div>
        </div>
        """
    elif size == "card":
        return f"""
        <div style="display:flex; align-items:center; justify-content:space-between; background:rgba(12,16,26,0.75); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:0.5rem 0.85rem; margin:0.4rem 0 0.55rem 0;">
            <div style="display:flex; align-items:center; gap:0.4rem;">
                <span style="font-size:1.05rem;">{icon}</span>
                <span style="font-size:0.80rem; font-weight:700; color:{theme_color};">[{state_label}]</span>
            </div>
            <!-- Battery chassis -->
            <div style="display:flex; align-items:center; width:135px;">
                <div style="flex:1; height:19px; background:rgba(8,10,16,0.92); border:1.8px solid {theme_color}; border-radius:5px; padding:2px; position:relative; overflow:hidden;">
                    <div style="width:{pct_int}%; height:100%; border-radius:2px; background:{bg_fill}; box-shadow:{glow}; transition:width 0.4s ease;"></div>
                    <!-- Cell Ticks -->
                    <div style="position:absolute; top:0; left:33%; width:1.5px; height:100%; background:rgba(8,10,16,0.7);"></div>
                    <div style="position:absolute; top:0; left:66%; width:1.5px; height:100%; background:rgba(8,10,16,0.7);"></div>
                </div>
                <!-- Nub -->
                <div style="width:4px; height:11px; background:{theme_color}; border-radius:0 2px 2px 0; margin-left:1.5px;"></div>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.88rem; font-weight:800; color:#e2e6f4;">{reps}/{target}</span>
                <span style="font-size:0.75rem; font-weight:700; color:{theme_color}; margin-left:0.25rem;">({pct_int}%)</span>
            </div>
        </div>
        """
    else:  # size == "sidebar"
        return f"""
        <div style="margin:0.5rem 0 0.2rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                <span style="font-size:0.80rem; font-weight:700; color:{theme_color};">{icon} [{state_label}]</span>
                <span style="font-size:0.82rem; font-weight:800; color:#e2e6f4;">{reps}/{target} ({pct_int}%)</span>
            </div>
            <div style="display:flex; align-items:center; width:100%;">
                <div style="flex:1; height:20px; background:rgba(8,10,16,0.92); border:2px solid {theme_color}; border-radius:6px; padding:2px; position:relative; overflow:hidden;">
                    <div style="width:{pct_int}%; height:100%; border-radius:3px; background:{bg_fill}; box-shadow:{glow};"></div>
                    <div style="position:absolute; top:0; left:25%; width:1.5px; height:100%; background:rgba(8,10,16,0.7);"></div>
                    <div style="position:absolute; top:0; left:50%; width:1.5px; height:100%; background:rgba(8,10,16,0.7);"></div>
                    <div style="position:absolute; top:0; left:75%; width:1.5px; height:100%; background:rgba(8,10,16,0.7);"></div>
                </div>
                <div style="width:5px; height:11px; background:{theme_color}; border-radius:0 3px 3px 0; margin-left:1.5px;"></div>
            </div>
        </div>
        """


# ─────────────────────────────────────────────
#  Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏋️ AI Gym Trainer")
    st.markdown("---")

    coach_st = get_coach_status()
    if coach_st["configured"]:
        st.markdown(
            '<span class="gemini-badge">★ GEMINI AI ONLINE</span> '
            '<span class="unlimited-badge">⚡ UNLIMITED</span>',
            unsafe_allow_html=True
        )
        st.caption(f"Model: {coach_st['model']} (Cascade)")
    else:
        st.markdown(
            '<span class="offline-badge">⚪ LOCAL AI COACH</span> '
            '<span class="unlimited-badge">⚡ UNLIMITED</span>',
            unsafe_allow_html=True
        )
        st.caption("Intelligent Local Engine Active")

    st.markdown("---")

    pages = {
        "🏠  Dashboard":    "dashboard",
        "🏋️  Exercise":     "exercise",
        "🤖  AI Coach":     "coach",
    }
    for label, key in pages.items():
        active = st.session_state.page == key
        btn_style = "primary" if active else "secondary"
        if st.button(label, key=f"nav_{key}", width='stretch', type=btn_style):
            # Stop tracking if switching away from exercise page
            if st.session_state.tracking and key != "exercise":
                _stop_tracking()
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown("### Today's Progress")
    stats    = load_daily_stats()
    summary  = get_daily_summary(stats)
    pct      = summary["percentage"]
    done     = summary["total_done"]
    target   = summary["total_target"]
    st.markdown(render_battery_html(done, target, label="DAILY STAMINA", size="sidebar"), unsafe_allow_html=True)
    if summary["all_completed"]:
        st.success("🏆 All targets met today!")
    st.markdown("---")
    if st.button("🔄 Reset Today's Stats", width='stretch'):
        reset_daily_stats()
        st.rerun()


# ─────────────────────────────────────────────
#  Helper: stop webcam tracking
# ─────────────────────────────────────────────
def _stop_tracking():
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

    if st.session_state.pose is not None:
        try:
            st.session_state.pose.close()
        except Exception:
            pass
        st.session_state.pose = None

    st.session_state.tracking     = False
    st.session_state.left_tracker  = None
    st.session_state.right_tracker = None
    st.session_state.tracker       = None
    st.session_state.prev_elbows   = {"left": None, "right": None}


def _get_reps_from_state() -> int:
    """Return current rep count from whichever tracker is active."""
    if st.session_state.ex_choice == "curl":
        lt = st.session_state.left_tracker
        rt = st.session_state.right_tracker
        if lt and rt:
            return max(lt.rep_count, rt.rep_count)
    else:
        t = st.session_state.tracker
        if t:
            return t.rep_count
    return 0


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD (Clean & Mobile-Friendly)
# ═══════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("## 🏋️ AI Gym Trainer")
    st.caption(f"📅 Today's Workout — 10 Reps per Exercise")

    # ── Daily Progress Header Card with Battery Logo HUD ──
    curr_stats = load_daily_stats()
    curr_summary = get_daily_summary(curr_stats)
    total_done = curr_summary["total_done"]
    total_target = curr_summary["total_target"]
    progress_val = curr_summary["percentage"]
    all_done = curr_summary["all_completed"]

    # Authentic Battery Logo HUD replacing linear progress bar
    st.markdown(
        render_battery_html(total_done, total_target, label="WORKOUT STAMINA BATTERY", size="large"),
        unsafe_allow_html=True
    )

    if all_done:
        st.success("🏆 Daily Goal Met! All 3 exercises completed today.")

    st.markdown("")

    # ── Exercise Cards (Clean, 1-Click Touch Friendly) ──
    exercise_meta = [
        {"id": "curl",           "name": "Dumbbell Curl",   "icon": "💪", "color": "#00c878"},
        {"id": "pec_dec",        "name": "Pec Dec Fly",     "icon": "🦋", "color": "#00a5ff"},
        {"id": "shoulder_press", "name": "Shoulder Press",  "icon": "🔱", "color": "#ffb91e"},
    ]

    exercises_data = curr_stats.get("exercises", {})
    ex_target = curr_stats.get("target_per_exercise", 10)

    for ex in exercise_meta:
        ex_id = ex["id"]
        ex_stat = exercises_data.get(ex_id, {})
        reps = ex_stat.get("reps", 0)
        completed = ex_stat.get("completed", False) or reps >= ex_target
        needed = max(0, ex_target - reps)
        ex_pct = min(1.0, reps / float(ex_target))

        if completed:
            badge_html = '<span style="background:rgba(45,189,110,0.2); color:#2dff99; padding:0.2rem 0.65rem; border-radius:999px; font-size:0.82rem; font-weight:700; border:1px solid #2dbd6e;">✅ DONE</span>'
            card_border = "rgba(45,189,110,0.45)"
        elif reps > 0:
            badge_html = f'<span style="background:rgba(255,160,0,0.18); color:#ffaa33; padding:0.2rem 0.65rem; border-radius:999px; font-size:0.82rem; font-weight:700; border:1px solid #ff9900;">⏳ {reps}/{ex_target} ({needed} left)</span>'
            card_border = "rgba(255,160,0,0.35)"
        else:
            badge_html = f'<span style="background:rgba(255,255,255,0.06); color:#9098b8; padding:0.2rem 0.65rem; border-radius:999px; font-size:0.82rem; font-weight:600;">0/{ex_target}</span>'
            card_border = "rgba(255,255,255,0.08)"

        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid {card_border}; border-radius:14px; padding:0.9rem 1.1rem; margin-top:0.7rem; margin-bottom:0.2rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:0.6rem;">
                        <span style="font-size:1.4rem;">{ex["icon"]}</span>
                        <span style="font-size:1.05rem; font-weight:700; color:{ex['color']};">{ex["name"]}</span>
                    </div>
                    {badge_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Battery Logo widget replacing flat line on each card
        st.markdown(
            render_battery_html(reps, ex_target, label=ex["name"], size="card"),
            unsafe_allow_html=True
        )

        btn_label = f"🔄 Redo {ex['name']}" if completed else f"▶ Start {ex['name']}"
        btn_type = "secondary" if completed else "primary"
        if st.button(btn_label, key=f"start_{ex_id}", width='stretch', type=btn_type):
            st.session_state.ex_choice = ex_id
            st.session_state.page = "exercise"
            st.rerun()

    st.markdown("---")

    # ── Fast Action Shortcuts (Mobile friendly) ──
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🤖 AI Fitness & Diet Coach", width='stretch', key="dash_btn_coach"):
            st.session_state.page = "coach"
            st.rerun()
    with b2:
        if st.button("🔄 Reset Today's Reps", width='stretch', key="dash_btn_reset"):
            reset_daily_stats()
            st.rerun()


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — EXERCISE (Live Webcam)
# ═══════════════════════════════════════════════════════════════
def page_exercise():
    import mediapipe as mp
    from pose_engine import (
        process_curl_frame,
        process_pec_dec_frame,
        process_shoulder_press_frame,
        _MP,
    )
    from dumbbell_curl import ArmCurlTracker
    from pec_dec_fly import PecDecTracker
    from shoulder_press import ShoulderPressTracker

    exercise_info = {
        "curl":           {"name": "Dumbbell Curl",   "icon": "💪", "color": "#00c878", "key_tip": "Elbow flexion — shoulder→elbow→wrist"},
        "pec_dec":        {"name": "Pec Dec Fly",     "icon": "🦋", "color": "#00a5ff", "key_tip": "Join & cross hands in front of chest"},
        "shoulder_press": {"name": "Shoulder Press",  "icon": "🔱", "color": "#ffb91e", "key_tip": "Press overhead — elbows to lockout"},
    }

    ex_id = st.session_state.get("ex_choice") or "curl"
    if ex_id not in exercise_info:
        ex_id = "curl"
    st.session_state.ex_choice = ex_id
    ex_meta = exercise_info[ex_id]

    st.markdown(f"# {ex_meta['icon']} {ex_meta['name']}")
    st.caption(ex_meta['key_tip'])
    st.markdown("---")

    # Exercise selector (if navigated directly)
    col_sel, col_cam = st.columns([1, 2])
    with col_sel:
        choice = st.selectbox(
            "Select Exercise",
            options=list(exercise_info.keys()),
            format_func=lambda k: f"{exercise_info[k]['icon']} {exercise_info[k]['name']}",
            index=list(exercise_info.keys()).index(ex_id),
            key="ex_selector",
        )
        if choice != ex_id:
            st.session_state.ex_choice = choice
            if st.session_state.tracking:
                _stop_tracking()
            st.rerun()

        target_reps = 10

        st.markdown("---")
        tracking = st.session_state.tracking

        if not tracking:
            if st.button("▶ Start Camera Tracking", width='stretch', type="primary"):
                # Initialise trackers
                if ex_id == "curl":
                    st.session_state.left_tracker  = ArmCurlTracker("LEFT",  target_reps)
                    st.session_state.right_tracker = ArmCurlTracker("RIGHT", target_reps)
                elif ex_id == "pec_dec":
                    st.session_state.tracker = PecDecTracker(target_reps)
                else:
                    st.session_state.tracker = ShoulderPressTracker(target_reps)

                # Directly open default camera (device 0)
                cap = None
                for dev in (0, 1):
                    c = cv2.VideoCapture(dev)
                    if c.isOpened():
                        cap = c
                        break
                    c.release()

                # On Windows, try DirectShow if default backend did not open
                if (cap is None or not cap.isOpened()) and sys.platform.startswith("win"):
                    c = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if c.isOpened():
                        cap = c

                if cap is None or not cap.isOpened():
                    st.error("❌ Physical webcam not detected on the host server.")
                    st.info(
                        "ℹ️ **Running on Streamlit Cloud?**\n\n"
                        "Cloud servers (AWS) do not have a physical webcam attached. "
                        "To use your local laptop camera with real-time pose tracking, run the app locally:\n\n"
                        "```bash\nstreamlit run streamlit_app.py\n```\n\n"
                        "💡 **Or test directly right now** using the video upload option below!"
                    )
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    # Open MediaPipe pose
                    pose = _MP.Pose(
                        min_detection_confidence=0.65,
                        min_tracking_confidence=0.65,
                        model_complexity=1,
                    )
                    st.session_state.cap   = cap
                    st.session_state.pose  = pose
                    st.session_state.tracking     = True
                    st.session_state.prev_elbows  = {"left": None, "right": None}
                    st.rerun()

            # Upload video option (works both locally and in cloud)
            with st.expander("📁 Or Test with a Workout Video (Works on Streamlit Cloud)"):
                uploaded_video = st.file_uploader(
                    "Upload workout video (MP4, MOV, AVI)",
                    type=["mp4", "mov", "avi"],
                    key=f"video_upload_{ex_id}"
                )
                if uploaded_video is not None:
                    if st.button("▶ Analyze Uploaded Video", width='stretch', key=f"btn_analyze_{ex_id}"):
                        if ex_id == "curl":
                            st.session_state.left_tracker  = ArmCurlTracker("LEFT",  target_reps)
                            st.session_state.right_tracker = ArmCurlTracker("RIGHT", target_reps)
                        elif ex_id == "pec_dec":
                            st.session_state.tracker = PecDecTracker(target_reps)
                        else:
                            st.session_state.tracker = ShoulderPressTracker(target_reps)

                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(uploaded_video.read())
                        tfile.close()

                        cap = cv2.VideoCapture(tfile.name)
                        pose = _MP.Pose(
                            min_detection_confidence=0.65,
                            min_tracking_confidence=0.65,
                            model_complexity=1,
                        )
                        st.session_state.cap   = cap
                        st.session_state.pose  = pose
                        st.session_state.tracking     = True
                        st.session_state.prev_elbows  = {"left": None, "right": None}
                        st.rerun()
        else:
            # Live rep display with Battery Logo Widget
            reps = _get_reps_from_state()
            st.metric("Reps Counted", reps, delta=None)
            st.markdown(
                render_battery_html(reps, target_reps, label=ex_meta["name"], size="card"),
                unsafe_allow_html=True
            )
            if reps >= target_reps:
                st.success("🏆 Target Reached!")

            with st.container():
                st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
                stop_pressed = st.button("⏹ Stop & Save", width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)

            if stop_pressed:
                reps_to_save = _get_reps_from_state()
                _stop_tracking()
                if reps_to_save > 0:
                    record_exercise_reps(ex_id, reps_to_save)
                    st.success(f"✅ Saved {reps_to_save} reps for {ex_meta['name']}!")
                st.session_state.page = "dashboard"
                st.rerun()

    # Webcam frame display
    with col_cam:
        if st.session_state.tracking:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()

            cap  = st.session_state.cap
            pose = st.session_state.pose

            consecutive_empty = 0
            while st.session_state.tracking:
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_empty += 1
                    if consecutive_empty > 30:
                        status_placeholder.info("Playback or stream completed. Click 'Stop & Save' above.")
                        break
                    time.sleep(0.02)
                    continue
                consecutive_empty = 0

                # Process frame
                if ex_id == "curl":
                    rgb_frame, st.session_state.prev_elbows = process_curl_frame(
                        pose, frame,
                        st.session_state.left_tracker,
                        st.session_state.right_tracker,
                        st.session_state.prev_elbows,
                    )
                elif ex_id == "pec_dec":
                    rgb_frame = process_pec_dec_frame(pose, frame, st.session_state.tracker)
                else:
                    rgb_frame = process_shoulder_press_frame(pose, frame, st.session_state.tracker)

                frame_placeholder.image(rgb_frame, channels="RGB", width='stretch')
                time.sleep(0.02)   # ~30 fps cap

        else:
            st.info("👆 Select an exercise and press **Start Tracking** to begin.")
            # Show placeholder camera preview
            st.markdown(
                '<div style="text-align:center;padding:4rem;background:rgba(255,255,255,0.02);'
                'border:1.5px dashed rgba(255,255,255,0.12);border-radius:18px;">'
                '<span style="font-size:4rem">📷</span><br>'
                '<p style="color:#9098b8;margin-top:1rem">Webcam feed will appear here</p>'
                '</div>',
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — AI COACH
# ═══════════════════════════════════════════════════════════════
def page_coach():
    st.markdown("# 🤖 AI Fitness & Nutrition Coach")
    st.markdown("---")

    coach_st = get_coach_status()
    col_badge1, col_badge2 = st.columns([3, 1])
    with col_badge1:
        if coach_st["configured"]:
            st.markdown(
                f'<span class="gemini-badge">★ {coach_st["model"].upper()} ONLINE</span> '
                f'<span class="unlimited-badge">⚡ UNLIMITED CHATS ACTIVE</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="offline-badge">⚪ LOCAL AI COACH ONLINE</span> '
                '<span class="unlimited-badge">⚡ UNLIMITED CHATS ACTIVE</span>',
                unsafe_allow_html=True
            )
    with col_badge2:
        st.markdown(
            '<div style="text-align:right;"><span style="color:#ffd700;font-size:0.85rem;font-weight:600;">📋 Table Diet Plans: ON</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # Daily brief (load once per session)
    if st.session_state.daily_brief is None:
        with st.spinner("Generating today's workout brief..."):
            curr_summary = get_daily_summary(load_daily_stats())
            st.session_state.daily_brief = generate_daily_brief(curr_summary)

    with st.expander("📋 Today's AI Workout Brief", expanded=False):
        st.markdown(st.session_state.daily_brief)
        if st.button("🔄 Refresh Brief"):
            curr_summary = get_daily_summary(load_daily_stats())
            st.session_state.daily_brief = generate_daily_brief(curr_summary)
            st.rerun()

    # Instant 7-Day Weekly Diet Plan Generator Box (with Height & Weight)
    with st.expander("🥗 Instant 7-Day Weekly Diet Plan (Personalized by Height & Weight)", expanded=True):
        st.markdown("Enter your **Height & Weight** to get an **ultra-simple 7-day weekly meal plan** customized for your body metrics using everyday home-cooked foods (*Ghar Ka Khana*):")
        d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns([2, 2, 1.3, 1.3, 2])
        with d_col1:
            goal_sel = st.selectbox("Fitness Goal", ["🔥 Fat Loss & Cutting", "💪 Muscle Gain & Bulking", "⚖️ Balanced Maintenance"], key="dg_goal")
        with d_col2:
            pref_sel = st.selectbox("Diet Preference", ["Veg & Non-Veg (Flexible)", "🥦 Pure Vegetarian", "🥚 Eggetarian", "🥩 High Protein Non-Veg"], key="dg_pref")
        with d_col3:
            weight_val = st.number_input("Weight (kg)", min_value=35, max_value=200, value=75, step=1, key="dg_wt")
        with d_col4:
            height_val = st.number_input("Height (cm)", min_value=120, max_value=230, value=175, step=1, key="dg_ht")
            ht_ft = int(height_val / 30.48)
            ht_in = int(round((height_val % 30.48) / 2.54))
            st.caption(f"≈ {ht_ft} ft {ht_in} in")
        with d_col5:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            gen_diet_btn = st.button("🚀 Generate Diet Table", width='stretch', key="btn_gen_diet_table")

        # Live Body Metrics Preview & BMI calculation badge
        h_m = height_val / 100.0
        bmi_calc = round(weight_val / (h_m * h_m), 1)
        if bmi_calc < 18.5:
            bmi_badge = f'<span style="background:rgba(90,156,255,0.18); border:1px solid #5a9cff; color:#8ec0ff; padding:0.25rem 0.75rem; border-radius:999px; font-weight:700; font-size:0.85rem;">⚖️ BMI: {bmi_calc} — Underweight</span>'
        elif bmi_calc < 25.0:
            bmi_badge = f'<span style="background:rgba(0,230,118,0.18); border:1px solid #00e676; color:#2dff99; padding:0.25rem 0.75rem; border-radius:999px; font-weight:700; font-size:0.85rem;">⚖️ BMI: {bmi_calc} — Healthy / Ideal Weight</span>'
        elif bmi_calc < 30.0:
            bmi_badge = f'<span style="background:rgba(255,153,0,0.18); border:1px solid #ff9900; color:#ffaa33; padding:0.25rem 0.75rem; border-radius:999px; font-weight:700; font-size:0.85rem;">⚖️ BMI: {bmi_calc} — Overweight (Fat Loss Recommended)</span>'
        else:
            bmi_badge = f'<span style="background:rgba(255,59,78,0.18); border:1px solid #ff3b4e; color:#ff6b7d; padding:0.25rem 0.75rem; border-radius:999px; font-weight:700; font-size:0.85rem;">⚖️ BMI: {bmi_calc} — Obese (Caloric Deficit Recommended)</span>'

        st.markdown(
            f'<div style="margin: 0.35rem 0 0.8rem 0; display:flex; align-items:center; gap:0.6rem;">'
            f'{bmi_badge} '
            f'<span style="color:#a0a6be; font-size:0.86rem; font-weight:600;">Profile: {weight_val} kg • {height_val} cm ({ht_ft}\'{ht_in}")</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        if gen_diet_btn:
            gen_query = f"{goal_sel} 7-day weekly diet plan for {weight_val}kg, height {height_val}cm ({ht_ft}ft {ht_in}in) with {pref_sel} preference using simple ghar ka khana home foods in a complete weekly table format"
            st.session_state.chat_history.append({"role": "user", "content": f"Generate an ultra-simple 7-Day Weekly Diet Plan for **{weight_val} kg, {height_val} cm ({ht_ft}'{ht_in}\", BMI: {bmi_calc})** ({goal_sel}, {pref_sel}) using everyday Ghar Ka Khana"})
            with st.spinner(f"Personalizing 7-day home-cooked weekly meal plan for {weight_val}kg, {height_val}cm..."):
                curr_summary = get_daily_summary(load_daily_stats())
                try:
                    table_plan = generate_diet_plan_table(gen_query, context=curr_summary, weight_kg=weight_val, height_cm=height_val)
                except TypeError:
                    # Defensive fallback if old signature was cached
                    table_plan = generate_diet_plan_table(gen_query, context=curr_summary)
            st.session_state.chat_history.append({"role": "assistant", "content": table_plan})
            st.rerun()

    st.markdown("---")
    st.markdown("### 💬 Ask Your AI Coach (Unlimited Access)")

    # Quick-topic buttons — all diet topics explicitly demand 7-Day Weekly Table Format
    st.markdown("**Quick Weekly Diet & Fitness Topics:**")
    qcols = st.columns(4)
    quick_topics = [
        ("📅 7-Day Weekly Diet", "Provide a 7-day weekly meal plan (Monday to Sunday) in table format using simple home foods for a normal person"),
        ("🔥 Weekly Fat Loss Table", "Provide a comprehensive 7-day weekly fat loss diet plan in a table format with simple Ghar Ka Khana and household portions"),
        ("💪 Weekly Muscle Bulking", "Provide a 7-day high-protein weekly diet plan table with home-cooked meals for muscle building"),
        ("🥦 Simple Veg Weekly Table", "Create a 7-day 100% vegetarian high-protein diet plan in a structured table format with home food"),
    ]
    for i, (label, q) in enumerate(quick_topics):
        with qcols[i]:
            if st.button(label, width='stretch', key=f"quick_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Consulting AI Coach..."):
                    curr_summary = get_daily_summary(load_daily_stats())
                    ans = ask_fitness_coach(q, context=curr_summary)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()

    st.markdown("---")

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask for any diet plan (table format), exercise form, nutrition, or recovery — unlimited chats…"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                curr_summary = get_daily_summary(load_daily_stats())
                ans = ask_fitness_coach(user_input, context=curr_summary)
            st.markdown(ans)

        st.session_state.chat_history.append({"role": "assistant", "content": ans})

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════
#  Router
# ═══════════════════════════════════════════════════════════════
page = st.session_state.page

if page == "dashboard":
    page_dashboard()
elif page == "exercise":
    page_exercise()
elif page == "coach":
    page_coach()
