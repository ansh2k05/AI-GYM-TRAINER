"""
Shoulder Press (Overhead Press) exercise tracker module.
========================================================

Angle tracked: elbow angle at each arm (shoulder → elbow → wrist).
  Left:  calculate_angle(L_SHOULDER, L_ELBOW, L_WRIST)
  Right: calculate_angle(R_SHOULDER, R_ELBOW, R_WRIST)

Bilateral overhead press:
  DOWN  (weights at shoulder level, elbows bent) : avg angle < ANGLE_DOWN (~95 deg)
  UP    (arms pressed overhead, lockout)         : avg angle > ANGLE_UP   (~150 deg)
  One rep = down -> up -> down

Target: 10 Reps
  - If reps < 10: Prominent RED WARNING SIGN is displayed on HUD & exit.
  - If reps >= 10: Emerald green celebration badge and completion fanfare.
"""

import copy
import time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from dumbbell_curl import calculate_angle, get_landmark_coords, check_visibility
from hud_ui import (
    WorkoutSession,
    draw_circular_rep_gauge,
    draw_glass_panel,
    draw_battery_rep_dashboard,
    draw_metric_pill,
    draw_target_sign,
    show_exit_evaluation,
    DEFAULT_TARGET_REPS,
)
from goku_effect import play_goku_ui_video_effect

# ─────────────────────────────────────────────
#  Landmark indices
# ─────────────────────────────────────────────
try:
    _MP  = mp.solutions.pose
    _DRW = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.pose as _MP
    import mediapipe.python.solutions.drawing_utils as _DRW

L_SHOULDER = _MP.PoseLandmark.LEFT_SHOULDER.value
R_SHOULDER = _MP.PoseLandmark.RIGHT_SHOULDER.value
L_ELBOW    = _MP.PoseLandmark.LEFT_ELBOW.value
R_ELBOW    = _MP.PoseLandmark.RIGHT_ELBOW.value
L_WRIST    = _MP.PoseLandmark.LEFT_WRIST.value
R_WRIST    = _MP.PoseLandmark.RIGHT_WRIST.value

VISIBILITY_THRESH = 0.50


# ─────────────────────────────────────────────
#  Tracker
# ─────────────────────────────────────────────

class ShoulderPressTracker:
    """
    Bilateral Shoulder Press rep tracker and form evaluator with 10-rep target goal.
    """

    ANGLE_DOWN = 95    # degrees – elbows bent at rack position (shoulder level)
    ANGLE_UP   = 150   # degrees – arms extended overhead (lockout)

    SMOOTH_WINDOW   = 5
    MIN_HOLD_FRAMES = 2   # consecutive frames to confirm state transition

    def __init__(self, target_reps: int = DEFAULT_TARGET_REPS):
        self.target_reps    = target_reps
        self.rep_count      = 0
        self.stage          = "down"
        self.left_angle     = 0.0
        self.right_angle    = 0.0
        self.avg_angle      = 0.0
        self.symmetry_diff  = 0.0
        self.feedback       = "Get in position"
        self.visible        = True
        self.in_position    = False

        self._buf        = deque(maxlen=self.SMOOTH_WINDOW)
        self._down_hold  = 0
        self._up_hold    = 0

        # Session cadence & celebration engine
        self.session = WorkoutSession(target_reps=self.target_reps)

    @property
    def progress_pct(self) -> float:
        """Dynamic 0.0 -> 1.0 progress from rack (90°) to lockout (150°)."""
        pct = (self.avg_angle - self.ANGLE_DOWN) / (self.ANGLE_UP - self.ANGLE_DOWN)
        return max(0.0, min(pct, 1.0))

    def update(self, l_raw: float, r_raw: float,
               lw_y: int, rw_y: int,
               ls_y: int, rs_y: int):
        self.left_angle     = l_raw
        self.right_angle    = r_raw
        self.symmetry_diff  = abs(l_raw - r_raw)

        avg_raw = (l_raw + r_raw) / 2.0
        self._buf.append(avg_raw)
        avg = float(np.mean(self._buf))
        self.avg_angle = avg

        # Wrists position relative to shoulders
        wrists_above_shoulders = (lw_y <= ls_y + 40) and (rw_y <= rs_y + 40)
        hands_hanging_down     = (lw_y > ls_y + 90) and (rw_y > rs_y + 90)

        if hands_hanging_down:
            self.in_position = False
            self.feedback    = "Raise weights to shoulders"
            self._down_hold  = 0
            self._up_hold    = 0
            return

        self.in_position = True

        # Hold counters
        if avg < self.ANGLE_DOWN and wrists_above_shoulders:
            self._down_hold += 1
            self._up_hold    = 0
        elif avg > self.ANGLE_UP and wrists_above_shoulders:
            self._up_hold   += 1
            self._down_hold  = 0
        else:
            self._down_hold = 0
            self._up_hold   = 0

        # State transitions
        if self._up_hold >= self.MIN_HOLD_FRAMES and self.stage == "down":
            self.stage     = "up"
            self._up_hold  = 0

        if self._down_hold >= self.MIN_HOLD_FRAMES and self.stage == "up":
            self.rep_count += 1
            self.stage      = "down"
            self._down_hold = 0
            self.session.on_rep_completed(self.rep_count)

        self._evaluate_form(avg)

    def _evaluate_form(self, avg: float):
        if self.symmetry_diff > 25:
            self.feedback = "Press evenly! Keep arms balanced"
        elif self.stage == "up" and avg < self.ANGLE_UP + 10:
            self.feedback = "Push higher overhead!"
        elif self.stage == "down" and avg > self.ANGLE_DOWN + 15:
            self.feedback = "Lower to shoulder level!"
        elif self._up_hold > 0:
            self.feedback = "Pressing up..."
        elif self._down_hold > 0:
            self.feedback = "Lowering..."
        else:
            if self.rep_count >= self.target_reps:
                self.feedback = "Target 10 Reps Reached! \u2605"
            else:
                self.feedback = f"Good form \u2713 ({self.target_reps - self.rep_count} to goal)"

    def mark_invisible(self):
        self.visible     = False
        self.in_position = False
        self._down_hold  = 0
        self._up_hold    = 0
        self.feedback    = "Upper body not visible"

    def mark_visible(self):
        self.visible = True

    def reset(self):
        self.rep_count     = 0
        self.stage         = "down"
        self.feedback      = "Get in position"
        self._buf.clear()
        self._down_hold    = 0
        self._up_hold      = 0
        self.in_position   = False
        self.session.reset()


# ─────────────────────────────────────────────
#  Colour palette (BGR)
# ─────────────────────────────────────────────
CLR_GOLD    = (255, 215,   0)
CLR_GREEN   = (0,   200, 120)
CLR_ORANGE  = (0,   165, 255)
CLR_CYAN    = (255, 185,  30)
CLR_TEXT    = (230, 230, 230)
CLR_GRAY    = (120, 120, 120)
CLR_RED     = (60,   60, 220)


# ─────────────────────────────────────────────
#  Rendering helpers
# ─────────────────────────────────────────────

def render_overlay(frame, tracker: ShoulderPressTracker, fps: float = 0.0):
    """Draw the professional Shoulder Press HUD on frame in-place."""
    h, w = frame.shape[:2]

    # ── Top bar (frosted glass) ──────────────────────────────────────
    draw_glass_panel(frame, (0, 0), (w, 75), bg_color=(12, 14, 20), alpha=0.78, radius=0)
    cv2.putText(frame, "AI GYM TRAINER  |  Shoulder Press",
                (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_CYAN, 2, cv2.LINE_AA)

    stats_str = f"TIME: {tracker.session.elapsed_str}   FPS: {fps:.1f}"
    cv2.putText(frame, stats_str, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, CLR_TEXT, 1, cv2.LINE_AA)

    # Battery Stamina Dashboard on camera video
    draw_battery_rep_dashboard(frame, tracker.rep_count, tracker.target_reps, x=w - 248, y=8, label="SHOULDER BATT")

    # ── Professional Rep Gauge Card ──────────────────────────────────
    cw, ch = 348, 220
    cx = (w - cw) // 2
    cy = h - ch - 42

    card_border = CLR_GOLD if tracker.session.is_popping else ((0, 200, 120) if tracker.rep_count >= tracker.target_reps else (55, 60, 75))
    draw_glass_panel(frame, (cx, cy), (cx + cw, cy + ch),
                     bg_color=(15, 17, 24), alpha=0.82, radius=16,
                     border_color=card_border, border_thickness=2 if tracker.session.is_popping else 1)

    cv2.line(frame, (cx + 18, cy + 2), (cx + cw - 18, cy + 2), CLR_CYAN, 3, cv2.LINE_AA)

    if not tracker.visible:
        cv2.putText(frame, tracker.feedback,
                    (cx + 25, cy + ch // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, CLR_GRAY, 1, cv2.LINE_AA)
    else:
        # Left column: Commercial Circular Rep Gauge (shows "of 10 REPS")
        gauge_center = (cx + 65, cy + 105)
        draw_circular_rep_gauge(
            frame=frame,
            center=gauge_center,
            radius=48,
            rep_count=tracker.rep_count,
            progress_pct=tracker.progress_pct,
            is_pop=tracker.session.is_popping,
            accent_color=CLR_CYAN,
            target_reps=tracker.target_reps,
        )

        # Right column: Target sign (RED if < 10, GREEN if >= 10)
        rx = cx + 135
        draw_target_sign(frame, rx, cy + 22, reps=tracker.rep_count, target=tracker.target_reps, width=195, height=44)

        # Stage & Tempo pills
        stage_clr = CLR_GREEN if tracker.stage == "up" else CLR_CYAN
        draw_metric_pill(frame, rx, cy + 74, label="STAGE", value=tracker.stage.upper(), color=stage_clr, width=93, height=40)
        draw_metric_pill(frame, rx + 101, cy + 74, label="TEMPO", value=tracker.session.avg_tempo_str, color=(255, 255, 255), width=94, height=40)

        # Symmetry & Angle Readout
        sym_clr = CLR_RED if tracker.symmetry_diff > 25 else CLR_GREEN
        sym_val = f"DIFF {int(tracker.symmetry_diff)}\u00b0" if tracker.symmetry_diff > 25 else "BALANCED"
        draw_metric_pill(frame, rx, cy + 120, label="SYMMETRY", value=sym_val, color=sym_clr, width=93, height=40)
        draw_metric_pill(frame, rx + 101, cy + 120, label="ANGLE", value=f"{int(tracker.avg_angle)}\u00b0", color=CLR_TEXT, width=94, height=40)

        # Hold-frame progress bar
        hold    = max(tracker._down_hold, tracker._up_hold)
        bar_pct = min(hold / tracker.MIN_HOLD_FRAMES, 1.0)
        bx1, by1 = cx + 22, cy + 172
        bx2, by2 = cx + cw - 22, cy + 180
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (38, 42, 52), -1)
        fill = int(bx1 + bar_pct * (bx2 - bx1))
        bar_clr = CLR_GREEN if tracker.stage == "down" else CLR_CYAN
        if fill > bx1:
            cv2.rectangle(frame, (bx1, by1), (fill, by2), bar_clr, -1)

        # Live feedback string
        fb_clr = CLR_GOLD if tracker.session.is_popping else (CLR_RED if "evenly" in tracker.feedback else CLR_TEXT)
        cv2.putText(frame, tracker.feedback,
                    (cx + 22, cy + 206),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, fb_clr, 1, cv2.LINE_AA)

    # ── Bottom hint bar ───────────────────────────────────────────────
    draw_glass_panel(frame, (0, h - 36), (w, h), bg_color=(12, 14, 20), alpha=0.65, radius=0)
    cv2.putText(frame, "R = reset reps   |   M = menu   |   Q / ESC = quit",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_TEXT, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  Exercise runner
# ─────────────────────────────────────────────

def run(camera_index: int = 0) -> dict:
    """
    Full Shoulder Press exercise loop with 10-rep target evaluation.

    Returns
    -------
    dict  {'reps': int, 'exit': 'quit' | 'menu'}
    """
    tracker = ShoulderPressTracker(target_reps=10)

    _LM_STYLE  = _DRW.DrawingSpec(color=(255, 185, 30), thickness=2, circle_radius=4)
    _CON_STYLE = _DRW.DrawingSpec(color=(200, 200, 200), thickness=2)

    with _MP.Pose(
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
        model_complexity=1,
    ) as pose:

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return {"reps": 0, "exit": "quit"}

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        prev_time   = time.time()
        exit_reason = "quit"
        print(f"[INFO] Shoulder Press started (Goal: {tracker.target_reps} reps).  R=reset  M=menu  Q/ESC=quit")

        last_rendered_frame = None

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            curr = time.time()
            fps  = 1.0 / max(curr - prev_time, 1e-9)
            prev_time = curr

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
                                    _LM_STYLE, _CON_STYLE)

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

                    # Live status banner
                    banner = f"Avg angle: {int(tracker.avg_angle)}\u00b0  (UP>{tracker.ANGLE_UP}\u00b0 / DOWN<{tracker.ANGLE_DOWN}\u00b0)"
                    cv2.putText(frame, banner,
                                (20, 108), cv2.FONT_HERSHEY_SIMPLEX,
                                0.68, CLR_CYAN, 2, cv2.LINE_AA)

                    # Draw angle values at elbows
                    cv2.putText(frame, f"{int(l_angle)}\u00b0",
                                (le[0] + 12, le[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_CYAN, 2, cv2.LINE_AA)
                    cv2.putText(frame, f"{int(r_angle)}\u00b0",
                                (re[0] + 12, re[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_CYAN, 2, cv2.LINE_AA)
                else:
                    tracker.mark_invisible()

            else:
                tracker.mark_invisible()
                cv2.putText(frame,
                            "No person detected \u2014 step into frame",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 100, 220), 2, cv2.LINE_AA)

            render_overlay(frame, tracker, fps)
            last_rendered_frame = frame.copy()
            cv2.imshow("AI Gym Trainer \u2014 Shoulder Press", frame)

            # Trigger Goku UI effect when 10 reps are completed!
            if tracker.session.should_play_goku_effect:
                tracker.session.should_play_goku_effect = False
                play_goku_ui_video_effect(window_name="AI Gym Trainer \u2014 Shoulder Press")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                tracker.reset()
                print("[INFO] Reps reset.")
            elif key == ord('m'):
                exit_reason = "menu"
                break

        # Show Exit Evaluation Screen with Red Sign if < 10 reps (and play sound)
        if last_rendered_frame is not None:
            show_exit_evaluation(last_rendered_frame, "Shoulder Press", tracker.rep_count,
                                 target_reps=tracker.target_reps, window_name="AI Gym Trainer \u2014 Shoulder Press",
                                 pace_avg=tracker.session.avg_pace, streak=tracker.session.streak)

        cap.release()
        cv2.destroyAllWindows()

    print("\n===== Shoulder Press Summary =====")
    print(f"  Reps   : {tracker.rep_count} / {tracker.target_reps}")
    if tracker.rep_count < tracker.target_reps:
        print(f"  Target : NOT ACHIEVED (Red Sign Alert \u2716 - Needed {tracker.target_reps})")
    else:
        print(f"  Target : ACHIEVED! (Full {tracker.target_reps} Reps Complete \u2605)")
    print(f"  Pace   : {tracker.session.avg_tempo_str}")
    print(f"  Streak : {tracker.session.streak}")
    print("==================================")
    return {"reps": tracker.rep_count, "exit": exit_reason}


if __name__ == "__main__":
    run(0)
