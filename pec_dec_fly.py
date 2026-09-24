"""
Pec Dec Fly (Chest Fly) exercise tracker module.
================================================

Rep counting rule:
  1. SPLIT  (arms open wide) : hands are apart (hand_ratio > 0.75 or avg angle > 88°)
  2. JOINED / CROSSED        : both hands come together and meet / cross in front of chest
                               (hands cross in x-axis or hand_dist < 0.35 * shoulder_dist)
  3. Rep counts immediately when both hands join and cross together!
  4. User splits arms back open to re-arm for the next repetition.

Target: 10 Reps
  - If reps < 10: Prominent RED WARNING SIGN on HUD & exit evaluation.
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

class PecDecTracker:
    """
    Bilateral Pec Dec / Chest Fly rep tracker with 10-rep target goal.
    """

    RATIO_SPLIT = 0.75   # hand distance / shoulder width when arms open
    RATIO_JOIN  = 0.35   # hand distance / shoulder width when hands join

    SMOOTH_WINDOW   = 5
    MIN_HOLD_FRAMES = 2   # consecutive frames to confirm state transition

    def __init__(self, target_reps: int = DEFAULT_TARGET_REPS):
        self.target_reps      = target_reps
        self.rep_count        = 0
        self.stage            = "split"     # "split" (open) or "joined" (closed/crossed)
        self.left_angle       = 0.0
        self.right_angle      = 0.0
        self.avg_angle        = 0.0
        self.hand_dist        = 0.0
        self.shoulder_dist    = 1.0
        self.hand_ratio       = 1.0
        self.hands_crossed    = False
        self.hands_joined     = False
        self.feedback         = "Split arms open to begin"
        self.visible          = True
        self._first_rep_armed = False

        self._ratio_buf  = deque(maxlen=self.SMOOTH_WINDOW)
        self._angle_buf  = deque(maxlen=self.SMOOTH_WINDOW)
        self._close_hold = 0
        self._open_hold  = 0

        # Session cadence & celebration engine
        self.session = WorkoutSession(target_reps=self.target_reps)

    @property
    def progress_pct(self) -> float:
        """Calculate real-time 0.0 -> 1.0 rep completion percentage."""
        if self.stage == "split":
            pct = (0.85 - self.hand_ratio) / (0.85 - 0.32)
        else:
            pct = 1.0
        return max(0.0, min(pct, 1.0))

    def update(self, l_raw: float, r_raw: float,
               lw: list, rw: list, ls: list, rs: list):
        self.left_angle  = l_raw
        self.right_angle = r_raw

        avg_raw = (l_raw + r_raw) / 2.0
        self._angle_buf.append(avg_raw)
        self.avg_angle = float(np.mean(self._angle_buf))

        raw_hand_dist     = float(np.hypot(lw[0] - rw[0], lw[1] - rw[1]))
        raw_shoulder_dist = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        self.hand_dist     = raw_hand_dist
        self.shoulder_dist = max(raw_shoulder_dist, 1.0)

        raw_ratio = raw_hand_dist / self.shoulder_dist
        self._ratio_buf.append(raw_ratio)
        ratio = float(np.mean(self._ratio_buf))
        self.hand_ratio = ratio

        crossed = (rw[0] >= lw[0] - 25)
        self.hands_crossed = crossed

        joined = crossed or (ratio < self.RATIO_JOIN) or (raw_hand_dist < 55) or (self.avg_angle < 65.0)
        self.hands_joined = joined

        split = (ratio > self.RATIO_SPLIT) or (self.avg_angle > 88.0)

        # First rep arming
        if not self._first_rep_armed:
            if split:
                self._first_rep_armed = True
                self.stage    = "split"
                self.feedback = "Bring hands together & cross!"
            else:
                self.stage    = "joined"
                self.feedback = "Split arms wide to start"
                return

        # Hold counters
        if joined:
            self._close_hold += 1
            self._open_hold   = 0
        elif split:
            self._open_hold  += 1
            self._close_hold  = 0
        else:
            self._close_hold = 0
            self._open_hold  = 0

        # Rep counts when both hands join and cross together!
        if self._close_hold >= self.MIN_HOLD_FRAMES and self.stage == "split":
            self.rep_count  += 1
            self.stage       = "joined"
            self._close_hold = 0
            self.session.on_rep_completed(self.rep_count)

            if crossed:
                self.feedback = "Hands crossed! Rep counted \u2713"
            else:
                self.feedback = "Hands joined! Rep counted \u2713"
            return

        if self._open_hold >= self.MIN_HOLD_FRAMES and self.stage == "joined":
            self.stage      = "split"
            self._open_hold = 0
            self.feedback   = "Arms split – Bring hands together!"
            return

        self._evaluate_form(joined, split)

    def _evaluate_form(self, joined: bool, split: bool):
        if self.stage == "split":
            if self._close_hold > 0:
                self.feedback = "Joining hands..."
            elif self.hand_ratio < 0.55:
                self.feedback = "Squeeze closer to cross hands!"
            else:
                if self.rep_count >= self.target_reps:
                    self.feedback = "Target 10 Reps Reached! \u2605"
                else:
                    self.feedback = f"Bring hands together ({self.target_reps - self.rep_count} to goal)"
        elif self.stage == "joined":
            if self._open_hold > 0:
                self.feedback = "Splitting arms..."
            elif self.hands_crossed:
                self.feedback = "Hands crossed! Now split open \u2713"
            else:
                self.feedback = "Hands joined! Now split open \u2713"

    def mark_invisible(self):
        self.visible     = False
        self._close_hold = 0
        self._open_hold  = 0
        self.feedback    = "Chest & arms not visible"

    def mark_visible(self):
        self.visible = True

    def reset(self):
        self.rep_count        = 0
        self.stage            = "split"
        self.feedback         = "Split arms open to begin"
        self.hands_crossed    = False
        self.hands_joined     = False
        self._ratio_buf.clear()
        self._angle_buf.clear()
        self._close_hold      = 0
        self._open_hold       = 0
        self._first_rep_armed = False
        self.session.reset()


# ─────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────
CLR_GOLD    = (255, 215,   0)
CLR_GREEN   = (0,   200, 120)
CLR_ORANGE  = (0,   165, 255)
CLR_TEXT    = (230, 230, 230)
CLR_GRAY    = (120, 120, 120)
CLR_BLUE    = (220, 100,   0)


# ─────────────────────────────────────────────
#  Rendering helpers
# ─────────────────────────────────────────────

def render_overlay(frame, tracker: PecDecTracker, fps: float = 0.0):
    """Draw the professional Pec Dec Fly HUD on frame in-place."""
    h, w = frame.shape[:2]

    # ── Top bar (frosted glass) ──────────────────────────────────────
    draw_glass_panel(frame, (0, 0), (w, 75), bg_color=(12, 14, 20), alpha=0.78, radius=0)
    cv2.putText(frame, "AI GYM TRAINER  |  Pec Dec Fly",
                (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_GOLD, 2, cv2.LINE_AA)

    stats_str = f"TIME: {tracker.session.elapsed_str}   FPS: {fps:.1f}"
    cv2.putText(frame, stats_str, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, CLR_TEXT, 1, cv2.LINE_AA)

    # Battery Stamina Dashboard on camera video
    draw_battery_rep_dashboard(frame, tracker.rep_count, tracker.target_reps, x=w - 248, y=8, label="PEC DEC BATT")

    # ── Professional Rep Gauge Card ──────────────────────────────────
    cw, ch = 348, 220
    cx = (w - cw) // 2
    cy = h - ch - 42

    card_border = CLR_GOLD if tracker.session.is_popping else ((0, 200, 120) if tracker.rep_count >= tracker.target_reps else (55, 60, 75))
    draw_glass_panel(frame, (cx, cy), (cx + cw, cy + ch),
                     bg_color=(15, 17, 24), alpha=0.82, radius=16,
                     border_color=card_border, border_thickness=2 if tracker.session.is_popping else 1)

    accent_bar_clr = CLR_GOLD if tracker.hands_crossed else (CLR_GREEN if tracker.stage == "joined" else CLR_ORANGE)
    cv2.line(frame, (cx + 18, cy + 2), (cx + cw - 18, cy + 2), accent_bar_clr, 3, cv2.LINE_AA)

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
            accent_color=CLR_GOLD if tracker.hands_crossed else CLR_ORANGE,
            target_reps=tracker.target_reps,
        )

        # Right column: Target sign (RED if < 10, GREEN if >= 10)
        rx = cx + 135
        draw_target_sign(frame, rx, cy + 22, reps=tracker.rep_count, target=tracker.target_reps, width=195, height=44)

        # Stage pill
        if tracker.hands_crossed:
            stg_txt, stg_clr = "CROSSED \u2605", CLR_GOLD
        elif tracker.stage == "joined":
            stg_txt, stg_clr = "JOINED", CLR_GREEN
        else:
            stg_txt, stg_clr = "SPLIT", CLR_ORANGE

        draw_metric_pill(frame, rx, cy + 74, label="STAGE", value=stg_txt, color=stg_clr, width=93, height=40)
        draw_metric_pill(frame, rx + 101, cy + 74, label="TEMPO", value=tracker.session.avg_tempo_str, color=(255, 255, 255), width=94, height=40)

        # Hand split percentage & Gap
        draw_metric_pill(frame, rx, cy + 120, label="GAP", value=f"{int(tracker.hand_dist)}px", color=CLR_TEXT, width=93, height=40)
        draw_metric_pill(frame, rx + 101, cy + 120, label="SPLIT", value=f"{int(tracker.hand_ratio * 100)}%", color=accent_bar_clr, width=94, height=40)

        # Hold-frame progress bar
        hold    = max(tracker._close_hold, tracker._open_hold)
        bar_pct = min(hold / tracker.MIN_HOLD_FRAMES, 1.0)
        bx1, by1 = cx + 22, cy + 172
        bx2, by2 = cx + cw - 22, cy + 180
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (38, 42, 52), -1)
        fill = int(bx1 + bar_pct * (bx2 - bx1))
        bar_clr = CLR_GREEN if tracker.stage == "split" else CLR_ORANGE
        if fill > bx1:
            cv2.rectangle(frame, (bx1, by1), (fill, by2), bar_clr, -1)

        # Live feedback string
        fb_clr = CLR_GOLD if tracker.session.is_popping else (CLR_GREEN if ("Good" in tracker.feedback or "counted" in tracker.feedback) else CLR_TEXT)
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
    Full Pec Dec Fly exercise loop with 10-rep target evaluation.

    Returns
    -------
    dict  {'reps': int, 'exit': 'quit' | 'menu'}
    """
    tracker = PecDecTracker(target_reps=10)

    _LM_STYLE  = _DRW.DrawingSpec(color=(0, 200, 120), thickness=2, circle_radius=4)
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
        print(f"[INFO] Pec Dec Fly started (Goal: {tracker.target_reps} reps).  R=reset  M=menu  Q/ESC=quit")

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

                    l_angle = calculate_angle(le, ls, rs)
                    r_angle = calculate_angle(re, rs, ls)

                    tracker.update(l_angle, r_angle, lw, rw, ls, rs)

                    # Live status banner
                    if tracker.hands_crossed:
                        banner_txt = "Hands: CROSSED \u2713"
                        banner_clr = CLR_GOLD
                    elif tracker.stage == "joined":
                        banner_txt = f"Hands: JOINED ({int(tracker.hand_dist)}px)"
                        banner_clr = CLR_GREEN
                    else:
                        banner_txt = f"Hands: SPLIT ({int(tracker.hand_dist)}px)"
                        banner_clr = CLR_ORANGE

                    cv2.putText(frame, banner_txt,
                                (20, 108), cv2.FONT_HERSHEY_SIMPLEX,
                                0.68, banner_clr, 2, cv2.LINE_AA)

                    # Visual bridge between hands
                    mid_x = (lw[0] + rw[0]) // 2
                    mid_y = (lw[1] + rw[1]) // 2

                    if tracker.hands_crossed or tracker.hands_joined:
                        glow_clr = CLR_GOLD if tracker.hands_crossed else CLR_GREEN
                        cv2.circle(frame, (lw[0], lw[1]), 10, glow_clr, -1, cv2.LINE_AA)
                        cv2.circle(frame, (rw[0], rw[1]), 10, glow_clr, -1, cv2.LINE_AA)
                        cv2.line(frame, (lw[0], lw[1]), (rw[0], rw[1]), glow_clr, 3, cv2.LINE_AA)
                        tag = "CROSSED!" if tracker.hands_crossed else "JOINED!"
                        cv2.putText(frame, tag,
                                    (mid_x - 38, mid_y - 14),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.65, glow_clr, 2, cv2.LINE_AA)
                    else:
                        cv2.line(frame, (lw[0], lw[1]), (rw[0], rw[1]), (100, 100, 100), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"{int(tracker.hand_dist)}px",
                                    (mid_x - 22, mid_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1, cv2.LINE_AA)

                    # Angle labels at elbows
                    cv2.putText(frame, f"{int(l_angle)}\u00b0",
                                (le[0] + 10, le[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_GREEN, 1, cv2.LINE_AA)
                    cv2.putText(frame, f"{int(r_angle)}\u00b0",
                                (re[0] + 10, re[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_GREEN, 1, cv2.LINE_AA)
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
            cv2.imshow("AI Gym Trainer \u2014 Pec Dec Fly", frame)

            # Trigger Goku UI effect when 10 reps are completed!
            if tracker.session.should_play_goku_effect:
                tracker.session.should_play_goku_effect = False
                play_goku_ui_video_effect(window_name="AI Gym Trainer \u2014 Pec Dec Fly")

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
            show_exit_evaluation(last_rendered_frame, "Pec Dec Fly", tracker.rep_count,
                                 target_reps=tracker.target_reps, window_name="AI Gym Trainer — Pec Dec Fly",
                                 pace_avg=tracker.session.avg_pace, streak=tracker.session.streak)

        cap.release()
        cv2.destroyAllWindows()

    print("\n===== Pec Dec Fly Summary =====")
    print(f"  Reps   : {tracker.rep_count} / {tracker.target_reps}")
    if tracker.rep_count < tracker.target_reps:
        print(f"  Target : NOT ACHIEVED (Red Sign Alert \u2716 - Needed {tracker.target_reps})")
    else:
        print(f"  Target : ACHIEVED! (Full {tracker.target_reps} Reps Complete \u2605)")
    print(f"  Pace   : {tracker.session.avg_tempo_str}")
    print(f"  Streak : {tracker.session.streak}")
    print("================================")
    return {"reps": tracker.rep_count, "exit": exit_reason}


if __name__ == "__main__":
    run(0)
