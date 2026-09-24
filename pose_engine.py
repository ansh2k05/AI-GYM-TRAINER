"""
pose_engine.py — AI Gym Trainer
================================
Thin frame-processing wrappers around the existing exercise tracker classes.
Each function takes a raw BGR webcam frame + a tracker object and returns:
  - The annotated RGB frame (ready for st.image)
  - Nothing else — tracker state is mutated in-place

This decouples the pose detection logic from the cv2.imshow / cv2.waitKey
display loop so it can be driven frame-by-frame from Streamlit.
"""

import copy
import time
import cv2
import numpy as np
import mediapipe as mp

from dumbbell_curl import (
    ArmCurlTracker,
    calculate_angle,
    get_landmark_coords,
    check_visibility,
    draw_angle_arc,
)
from pec_dec_fly import PecDecTracker
from shoulder_press import ShoulderPressTracker
from hud_ui import draw_battery_rep_dashboard

# ─────────────────────────────────────────────
#  MediaPipe shared setup
# ─────────────────────────────────────────────
_MP  = mp.solutions.pose
_DRW = mp.solutions.drawing_utils

# Landmark style constants
_LM_STYLE_GREEN  = _DRW.DrawingSpec(color=(0, 200, 120),  thickness=2, circle_radius=4)
_LM_STYLE_ORANGE = _DRW.DrawingSpec(color=(0, 165, 255),  thickness=2, circle_radius=4)
_LM_STYLE_CYAN   = _DRW.DrawingSpec(color=(255, 185, 30), thickness=2, circle_radius=4)
_CON_STYLE       = _DRW.DrawingSpec(color=(200, 200, 200), thickness=2)

VISIBILITY_THRESH  = 0.65
MAX_ELBOW_DRIFT_PX = 80

# Landmark indices
L_SHOULDER = _MP.PoseLandmark.LEFT_SHOULDER.value
R_SHOULDER = _MP.PoseLandmark.RIGHT_SHOULDER.value
L_ELBOW    = _MP.PoseLandmark.LEFT_ELBOW.value
R_ELBOW    = _MP.PoseLandmark.RIGHT_ELBOW.value
L_WRIST    = _MP.PoseLandmark.LEFT_WRIST.value
R_WRIST    = _MP.PoseLandmark.RIGHT_WRIST.value


# ─────────────────────────────────────────────
#  Shared helper: draw rep counter badge
# ─────────────────────────────────────────────

def _draw_rep_badge(frame, label: str, reps: int, target: int,
                    x: int, y: int, accent_color=(0, 200, 120)):
    """Draw a clean rep counter badge on the frame."""
    h, w = frame.shape[:2]
    bw, bh = 160, 70
    x = max(0, min(x, w - bw))
    y = max(0, min(y, h - bh))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    is_met = reps >= target
    border = (0, 215, 100) if is_met else (50, 50, 230)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), border, 2)

    cv2.putText(frame, label, (x + 10, y + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, accent_color, 1)
    cv2.putText(frame, str(reps), (x + 10, y + 58),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 215, 0), 2)
    cv2.putText(frame, f"/ {target}", (x + 55, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _draw_header(frame, title: str, feedback: str, color=(255, 215, 0)):
    """Draw a transparent top-bar with title and live feedback (leaves space for battery dashboard on right)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (10, 10, 14), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.putText(frame, title, (18, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, color, 2, cv2.LINE_AA)
    if feedback:
        cv2.putText(frame, feedback, (18, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 220), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  1. Dumbbell Curl
# ─────────────────────────────────────────────

def process_curl_frame(
    pose,
    frame: np.ndarray,
    left_tracker: ArmCurlTracker,
    right_tracker: ArmCurlTracker,
    prev_elbows: dict,
) -> tuple[np.ndarray, dict]:
    """
    Process one BGR webcam frame for Dumbbell Curl.

    Returns
    -------
    annotated_rgb  : np.ndarray  RGB image ready for st.image
    prev_elbows    : dict with 'left' and 'right' updated elbow positions
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    frame = cv2.flip(frame, 1)

    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark

        mirrored = copy.deepcopy(results.pose_landmarks)
        for lm in mirrored.landmark:
            lm.x = 1.0 - lm.x
        _DRW.draw_landmarks(frame, mirrored, _MP.POSE_CONNECTIONS,
                            _LM_STYLE_GREEN, _CON_STYLE)

        # Left arm
        if check_visibility(lms, [L_SHOULDER, L_ELBOW, L_WRIST], VISIBILITY_THRESH):
            left_tracker.mark_visible()
            ls = get_landmark_coords(lms, L_SHOULDER, frame.shape, mirror=True)
            le = get_landmark_coords(lms, L_ELBOW,    frame.shape, mirror=True)
            lw = get_landmark_coords(lms, L_WRIST,    frame.shape, mirror=True)
            prev_le = prev_elbows.get("left")
            drift_ok = (prev_le is None or abs(le[1] - prev_le[1]) <= MAX_ELBOW_DRIFT_PX)
            if drift_ok:
                l_angle = calculate_angle(ls, le, lw)
                left_tracker.update(l_angle)
                draw_angle_arc(frame, le, l_angle, active=True)
            prev_elbows["left"] = le
        else:
            left_tracker.mark_invisible()
            prev_elbows["left"] = None

        # Right arm
        if check_visibility(lms, [R_SHOULDER, R_ELBOW, R_WRIST], VISIBILITY_THRESH):
            right_tracker.mark_visible()
            rs = get_landmark_coords(lms, R_SHOULDER, frame.shape, mirror=True)
            re = get_landmark_coords(lms, R_ELBOW,    frame.shape, mirror=True)
            rw = get_landmark_coords(lms, R_WRIST,    frame.shape, mirror=True)
            prev_re = prev_elbows.get("right")
            drift_ok = (prev_re is None or abs(re[1] - prev_re[1]) <= MAX_ELBOW_DRIFT_PX)
            if drift_ok:
                r_angle = calculate_angle(rs, re, rw)
                right_tracker.update(r_angle)
                draw_angle_arc(frame, re, r_angle, active=True)
            prev_elbows["right"] = re
        else:
            right_tracker.mark_invisible()
            prev_elbows["right"] = None
    else:
        left_tracker.mark_invisible()
        right_tracker.mark_invisible()
        prev_elbows["left"] = prev_elbows["right"] = None
        cv2.putText(frame, "No person detected — step into frame",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 220), 2)

    # Overlay
    h, w = frame.shape[:2]
    lr = max(left_tracker.rep_count, right_tracker.rep_count)
    fb = left_tracker.feedback or right_tracker.feedback
    _draw_header(frame, "AI GYM TRAINER  |  Dumbbell Curl", fb, (0, 200, 120))
    draw_battery_rep_dashboard(frame, lr, left_tracker.target_reps, x=w - 246, y=7, label="DUMBBELL CURL")
    _draw_rep_badge(frame, "LEFT ARM",  left_tracker.rep_count,  left_tracker.target_reps,
                    w - 175, h - 80, accent_color=(0, 165, 255))
    _draw_rep_badge(frame, "RIGHT ARM", right_tracker.rep_count, right_tracker.target_reps,
                    10, h - 80, accent_color=(0, 200, 120))

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), prev_elbows


# ─────────────────────────────────────────────
#  2. Pec Dec Fly
# ─────────────────────────────────────────────

def process_pec_dec_frame(
    pose,
    frame: np.ndarray,
    tracker: PecDecTracker,
) -> np.ndarray:
    """Process one BGR frame for Pec Dec Fly. Returns annotated RGB frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    frame = cv2.flip(frame, 1)

    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark

        mirrored = copy.deepcopy(results.pose_landmarks)
        for lm in mirrored.landmark:
            lm.x = 1.0 - lm.x
        _DRW.draw_landmarks(frame, mirrored, _MP.POSE_CONNECTIONS,
                            _LM_STYLE_ORANGE, _CON_STYLE)

        needed = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]
        if check_visibility(lms, needed, 0.50):
            tracker.mark_visible()
            ls = get_landmark_coords(lms, L_SHOULDER, frame.shape, mirror=True)
            rs = get_landmark_coords(lms, R_SHOULDER, frame.shape, mirror=True)
            le = get_landmark_coords(lms, L_ELBOW,    frame.shape, mirror=True)
            re = get_landmark_coords(lms, R_ELBOW,    frame.shape, mirror=True)
            lw = get_landmark_coords(lms, L_WRIST,    frame.shape, mirror=True)
            rw = get_landmark_coords(lms, R_WRIST,    frame.shape, mirror=True)

            l_angle = calculate_angle(ls, le, lw)
            r_angle = calculate_angle(rs, re, rw)
            tracker.update(l_angle, r_angle, lw, rw, ls, rs)

            # Hand distance visualisation
            cx = int((lw[0] + rw[0]) / 2)
            cy = int((lw[1] + rw[1]) / 2)
            cv2.line(frame, tuple(lw), tuple(rw), (0, 165, 255), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 215, 120), -1)
        else:
            tracker.mark_invisible()
    else:
        tracker.mark_invisible()
        cv2.putText(frame, "No person detected — step into frame",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 220), 2)

    h, w = frame.shape[:2]
    _draw_header(frame, "AI GYM TRAINER  |  Pec Dec Fly", tracker.feedback, (0, 165, 255))
    draw_battery_rep_dashboard(frame, tracker.rep_count, tracker.target_reps, x=w - 246, y=7, label="PEC DEC FLY")
    _draw_rep_badge(frame, "PEC DEC FLY", tracker.rep_count, tracker.target_reps,
                    w // 2 - 80, h - 80, accent_color=(0, 165, 255))

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
#  3. Shoulder Press
# ─────────────────────────────────────────────

def process_shoulder_press_frame(
    pose,
    frame: np.ndarray,
    tracker: ShoulderPressTracker,
) -> np.ndarray:
    """Process one BGR frame for Shoulder Press. Returns annotated RGB frame."""
    CLR_CYAN = (255, 185, 30)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    frame = cv2.flip(frame, 1)

    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark

        mirrored = copy.deepcopy(results.pose_landmarks)
        for lm in mirrored.landmark:
            lm.x = 1.0 - lm.x
        _DRW.draw_landmarks(frame, mirrored, _MP.POSE_CONNECTIONS,
                            _LM_STYLE_CYAN, _CON_STYLE)

        needed = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]
        if check_visibility(lms, needed, VISIBILITY_THRESH):
            tracker.mark_visible()
            ls = get_landmark_coords(lms, L_SHOULDER, frame.shape, mirror=True)
            rs = get_landmark_coords(lms, R_SHOULDER, frame.shape, mirror=True)
            le = get_landmark_coords(lms, L_ELBOW,    frame.shape, mirror=True)
            re = get_landmark_coords(lms, R_ELBOW,    frame.shape, mirror=True)
            lw = get_landmark_coords(lms, L_WRIST,    frame.shape, mirror=True)
            rw = get_landmark_coords(lms, R_WRIST,    frame.shape, mirror=True)

            l_angle = calculate_angle(ls, le, lw)
            r_angle = calculate_angle(rs, re, rw)
            tracker.update(l_angle, r_angle, lw[1], rw[1], ls[1], rs[1])

            banner = f"Avg: {int(tracker.avg_angle)}°  (UP>{tracker.ANGLE_UP}° / DOWN<{tracker.ANGLE_DOWN}°)"
            cv2.putText(frame, banner, (20, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, CLR_CYAN, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{int(l_angle)}°", (le[0] + 12, le[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_CYAN, 2)
            cv2.putText(frame, f"{int(r_angle)}°", (re[0] + 12, re[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_CYAN, 2)
        else:
            tracker.mark_invisible()
    else:
        tracker.mark_invisible()
        cv2.putText(frame, "No person detected — step into frame",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 220), 2)

    h, w = frame.shape[:2]
    _draw_header(frame, "AI GYM TRAINER  |  Shoulder Press", tracker.feedback, CLR_CYAN)
    draw_battery_rep_dashboard(frame, tracker.rep_count, tracker.target_reps, x=w - 246, y=7, label="SHOULDER PRESS")
    _draw_rep_badge(frame, "SHOULDER PRESS", tracker.rep_count, tracker.target_reps,
                    w // 2 - 80, h - 80, accent_color=CLR_CYAN)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
