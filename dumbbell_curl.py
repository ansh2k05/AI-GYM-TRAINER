import numpy as np
import cv2
from collections import deque
from hud_ui import (
    draw_glass_panel,
    draw_battery_rep_dashboard,
    play_rep_chime,
    play_rep_below_10_sound,
    play_target_completed_sound,
    play_target_not_met_sound,
    show_exit_evaluation,
    DEFAULT_TARGET_REPS,
)
from goku_effect import (
    play_goku_ui_sound,
    play_goku_ui_video_effect,
)


# ─────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────

def calculate_angle(a, b, c):
    """
    Calculate the angle (degrees) at point B formed by vectors BA and BC.

    Parameters
    ----------
    a, b, c : array-like  [x, y]
        Three 2-D points where B is the vertex of the angle.

    Returns
    -------
    float : angle in degrees [0, 180]
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def get_landmark_coords(landmarks, idx, frame_shape, mirror=False):
    """
    Convert a normalised MediaPipe landmark to pixel coordinates.

    Parameters
    ----------
    landmarks   : mediapipe landmark list
    idx         : int  - landmark index
    frame_shape : tuple (height, width, channels)
    mirror      : bool - if True, flip x so coords match a horizontally
                         flipped display frame (webcam mirror mode)

    Returns
    -------
    list [x_px, y_px]
    """
    h, w = frame_shape[:2]
    lm = landmarks[idx]
    x = int((1.0 - lm.x) * w) if mirror else int(lm.x * w)
    return [x, int(lm.y * h)]


def check_visibility(landmarks, indices, threshold=0.65):
    """
    Return True only when ALL landmark indices exceed the visibility threshold.

    MediaPipe Pose gives each landmark a [0-1] visibility score.
    We gate all angle calculations behind this check so that
    partially occluded or guessed landmarks cannot trigger false reps.

    Parameters
    ----------
    landmarks  : mediapipe landmark list
    indices    : list[int]  – landmark indices to check
    threshold  : float      – minimum acceptable visibility

    Returns
    -------
    bool
    """
    return all(landmarks[i].visibility >= threshold for i in indices)


# ─────────────────────────────────────────────
#  Per-arm state machine
# ─────────────────────────────────────────────

class ArmCurlTracker:
    """
    Robust curl-rep tracker for a single arm.

    Reliability layers
    ------------------
    1. Angle smoothing  – rolling mean over SMOOTH_WINDOW frames
                          eliminates jitter from landmark noise.
    2. Hold-frames gate – the arm must sustain the UP or DOWN angle
                          zone for MIN_HOLD_FRAMES consecutive frames
                          before a state transition is accepted.
                          This blocks false reps from quick hand flicks.
    3. Visibility gate  – applied in main.py before calling update().
    4. Elbow-anchor check – caller checks the elbow Y-position isn't
                            wildly shifting (upper arm swinging).

    States
    ------
    'down'  – arm extended   (smoothed angle > ANGLE_DOWN)
    'up'    – arm curled     (smoothed angle < ANGLE_UP)

    A complete rep = down -> up -> down.
    """

    # ── Thresholds ──────────────────────────────────────────────────
    ANGLE_DOWN  = 150    # degrees – arm considered "straight"
    ANGLE_UP    = 55     # degrees – arm considered "curled"

    # ── Reliability tuning ──────────────────────────────────────────
    SMOOTH_WINDOW   = 5   # frames to average
    MIN_HOLD_FRAMES = 3   # consecutive in-zone frames to confirm state

    def __init__(self, label: str, target_reps: int = DEFAULT_TARGET_REPS):
        self.label          = label       # "LEFT" or "RIGHT"
        self.target_reps    = target_reps
        self.rep_count      = 0
        self.stage          = "down"
        self.current_angle  = 0.0        # smoothed angle shown on HUD
        self.feedback       = ""
        self.visible        = True        # set False when landmarks hidden
        self.target_achieved = False
        self.should_play_goku_effect = False

        # Rolling angle buffer for smoothing
        self._angle_buffer: deque = deque(maxlen=self.SMOOTH_WINDOW)

        # Hold-frame counters
        self._up_hold   = 0   # consecutive frames angle < ANGLE_UP
        self._down_hold = 0   # consecutive frames angle > ANGLE_DOWN

    # ────────────────────────────────────────────────────────────────
    def update(self, raw_angle: float):
        """
        Feed raw elbow angle from the current frame and advance the
        state machine with all reliability filters applied.

        Parameters
        ----------
        raw_angle : float  – elbow angle in degrees (from calculate_angle)
        """
        # ── 1. Smooth ────────────────────────────────────────────────
        self._angle_buffer.append(raw_angle)
        angle = float(np.mean(self._angle_buffer))
        self.current_angle = angle

        # ── 2. Accumulate hold-frame counters ────────────────────────
        if angle < self.ANGLE_UP:
            self._up_hold   += 1
            self._down_hold  = 0
        elif angle > self.ANGLE_DOWN:
            self._down_hold += 1
            self._up_hold    = 0
        else:
            # In the middle – neither extreme, reset both
            self._up_hold   = 0
            self._down_hold = 0

        # ── 3. State transitions (gated by MIN_HOLD_FRAMES) ─────────
        if self._up_hold >= self.MIN_HOLD_FRAMES and self.stage == "down":
            self.stage  = "up"
            self._up_hold = 0          # reset so it doesn't re-trigger

        if self._down_hold >= self.MIN_HOLD_FRAMES and self.stage == "up":
            self.rep_count += 1
            self.stage      = "down"
            self._down_hold = 0
            if self.rep_count >= self.target_reps and not self.target_achieved:
                self.target_achieved = True
                self.should_play_goku_effect = True
            elif self.rep_count < self.target_reps:
                play_rep_below_10_sound(self.rep_count)
            else:
                play_rep_chime(1350, 100)

        # ── 4. Form feedback ────────────────────────────────────────
        self._evaluate_form(angle)

    # ────────────────────────────────────────────────────────────────
    def mark_invisible(self):
        """
        Call when landmarks are below visibility threshold.
        Clears the hold counters so that partial detections do not
        accumulate toward a false state transition.
        """
        self.visible    = False
        self._up_hold   = 0
        self._down_hold = 0
        self.feedback   = "Arm not visible"

    def mark_visible(self):
        self.visible = True

    # ────────────────────────────────────────────────────────────────
    def _evaluate_form(self, angle: float):
        """Generate real-time coaching cue based on smoothed angle."""
        if self.stage == "up" and angle > self.ANGLE_UP + 20:
            self.feedback = "Curl higher!"
        elif self.stage == "down" and angle < self.ANGLE_DOWN - 10:
            self.feedback = "Extend fully!"
        elif self._up_hold > 0:
            self.feedback = "Going up..."
        elif self._down_hold > 0:
            self.feedback = "Coming down..."
        else:
            self.feedback = "Good form ✓"

    # ────────────────────────────────────────────────────────────────
    def reset(self):
        """Reset rep count, state, and all buffers."""
        self.rep_count          = 0
        self.stage              = "down"
        self.feedback           = ""
        self.target_achieved         = False
        self.should_play_goku_effect = False
        self._angle_buffer.clear()
        self._up_hold           = 0
        self._down_hold         = 0


# ─────────────────────────────────────────────
#  Overlay renderer
# ─────────────────────────────────────────────

# Colour palette (BGR format for OpenCV)
CLR_BG        = (15,  15,  15)
CLR_ACCENT    = (0,  200, 120)     # green
CLR_WARNING   = (0,  165, 255)     # orange
CLR_TEXT      = (230, 230, 230)
CLR_HIGHLIGHT = (255, 215,   0)    # gold
CLR_RED       = (60,   60, 220)
CLR_GRAY      = (120, 120, 120)


def _draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    """Draw a filled rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)


def draw_angle_arc(frame, vertex, angle, active=True):
    """
    Draw a small dynamic arc at the elbow to visualise the joint angle.
    Greyed out when the landmark is not visible enough.
    """
    radius = 30
    if not active:
        cv2.ellipse(frame, tuple(vertex), (radius, radius),
                    0, -90, 90, CLR_GRAY, 2)
        return

    ratio = min(angle / 180.0, 1.0)
    arc_color = (
        int((1 - ratio) * 60  + ratio * 0),
        int((1 - ratio) * 60  + ratio * 200),
        int((1 - ratio) * 220 + ratio * 120),
    )
    cv2.ellipse(frame, tuple(vertex), (radius, radius),
                0, -90, int(angle - 90), arc_color, 3)
    cv2.putText(frame, f"{int(angle)}", (vertex[0]+35, vertex[1]+8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, arc_color, 2)


def render_overlay(frame,
                   left_tracker: ArmCurlTracker,
                   right_tracker: ArmCurlTracker,
                   fps: float = 0.0):
    """
    Draw the full HUD on frame (in-place).

    Parameters
    ----------
    frame         : BGR numpy array
    left_tracker  : ArmCurlTracker for the left arm
    right_tracker : ArmCurlTracker for the right arm
    fps           : current frames-per-second
    """
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ─────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "AI GYM TRAINER  |  Dumbbell Curl",
                (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_HIGHLIGHT, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1)

    # Battery Stamina Dashboard on camera video
    max_reps = max(left_tracker.rep_count, right_tracker.rep_count)
    draw_battery_rep_dashboard(frame, max_reps, left_tracker.target_reps, x=w - 248, y=10, label="CURL REPS")

    # ── Rep counter cards ────────────────────────────────────────────
    # In mirror/selfie view: person's LEFT arm appears on RIGHT side of screen
    _draw_arm_card(frame, left_tracker,  x=w - 225, y=h - 195)  # RIGHT side of screen
    _draw_arm_card(frame, right_tracker, x=20,       y=h - 195)  # LEFT  side of screen

    # ── Bottom hint bar ──────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 36), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "R = reset reps   |   Q / ESC = quit",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_TEXT, 1)


def _draw_arm_card(frame, tracker: ArmCurlTracker, x: int, y: int):
    """
    Draw a compact info card for one arm showing:
    reps, target status (red sign if < 10), stage, angle, progress bar, feedback.
    """
    card_w, card_h = 205, 185
    overlay = frame.copy()
    _draw_rounded_rect(overlay, (x, y), (x + card_w, y + card_h),
                       (20, 20, 20), radius=14)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

    if not tracker.visible:
        cv2.putText(frame, f"{tracker.label} ARM",
                    (x + 14, y + 28), cv2.FONT_HERSHEY_DUPLEX,
                    0.62, CLR_GRAY, 1)
        cv2.putText(frame, "Not visible",
                    (x + 14, y + 95), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, CLR_GRAY, 1)
        return

    # Label
    lbl_color = CLR_ACCENT if tracker.label == "RIGHT" else CLR_WARNING
    cv2.putText(frame, f"{tracker.label} ARM",
                (x + 14, y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.62, lbl_color, 2)

    # Rep count
    rep_str = str(tracker.rep_count)
    cv2.putText(frame, rep_str,
                (x + 14, y + 80), cv2.FONT_HERSHEY_DUPLEX, 2.0, CLR_HIGHLIGHT, 3)
    cv2.putText(frame, f"of {tracker.target_reps} REPS",
                (x + (65 if tracker.rep_count < 10 else 90), y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, CLR_TEXT, 1)

    # Target indicator (RED SIGN if < 10, GREEN if >= 10)
    is_met = tracker.rep_count >= tracker.target_reps
    badge_bg = (18, 38, 22) if is_met else (32, 16, 20)
    badge_border = (0, 215, 120) if is_met else (45, 45, 235)  # Bright Red
    badge_txt = f"\u2713 TARGET 10 MET!" if is_met else f"[!] TARGET: {tracker.rep_count}/10 (UNDER)"
    txt_clr = (0, 230, 130) if is_met else (65, 85, 255)

    draw_glass_panel(frame, (x + 10, y + 92), (x + card_w - 10, y + 118),
                     bg_color=badge_bg, alpha=0.88, radius=6,
                     border_color=badge_border, border_thickness=2 if not is_met else 1)
    cv2.putText(frame, badge_txt, (x + 15, y + 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, txt_clr, 1, cv2.LINE_AA)

    # Stage + angle
    stage_color = CLR_ACCENT if tracker.stage == "up" else CLR_RED
    cv2.putText(frame, tracker.stage.upper(),
                (x + 14, y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.50, stage_color, 2)
    cv2.putText(frame, f"{int(tracker.current_angle)}\u00b0",
                (x + 115, y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_TEXT, 1)

    # Hold-frame progress bar
    hold = max(tracker._up_hold, tracker._down_hold)
    bar_max = tracker.MIN_HOLD_FRAMES
    bar_pct = min(hold / bar_max, 1.0)
    bar_x1, bar_y1 = x + 14, y + 148
    bar_x2, bar_y2 = x + card_w - 14, y + 155
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
    bar_fill = int(bar_x1 + bar_pct * (bar_x2 - bar_x1))
    bar_clr  = CLR_ACCENT if tracker.stage == "down" else CLR_WARNING
    if bar_fill > bar_x1:
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_fill, bar_y2), bar_clr, -1)

    # Feedback text
    fb_color = CLR_ACCENT if "Good" in tracker.feedback or "..." in tracker.feedback \
               else CLR_WARNING
    cv2.putText(frame, tracker.feedback,
                (x + 10, y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.38, fb_color, 1)


# ─────────────────────────────────────────────
#  Exercise runner
# ─────────────────────────────────────────────

def run(camera_index: int = 0) -> dict:
    """
    Full Dumbbell Curl exercise loop.

    Returns
    -------
    dict  {'left_reps': int, 'right_reps': int, 'exit': 'quit' | 'menu'}
    """
    import copy
    import time
    import mediapipe as mp

    try:
        _MP  = mp.solutions.pose
        _DRW = mp.solutions.drawing_utils
    except AttributeError:
        import mediapipe.python.solutions.pose as _MP
        import mediapipe.python.solutions.drawing_utils as _DRW

    L_SHOULDER = _MP.PoseLandmark.LEFT_SHOULDER.value
    L_ELBOW    = _MP.PoseLandmark.LEFT_ELBOW.value
    L_WRIST    = _MP.PoseLandmark.LEFT_WRIST.value
    R_SHOULDER = _MP.PoseLandmark.RIGHT_SHOULDER.value
    R_ELBOW    = _MP.PoseLandmark.RIGHT_ELBOW.value
    R_WRIST    = _MP.PoseLandmark.RIGHT_WRIST.value

    VISIBILITY_THRESH  = 0.65
    MAX_ELBOW_DRIFT_PX = 80

    _LM_STYLE  = _DRW.DrawingSpec(color=(0, 200, 120), thickness=2, circle_radius=4)
    _CON_STYLE = _DRW.DrawingSpec(color=(200, 200, 200), thickness=2)

    left_tracker  = ArmCurlTracker("LEFT")
    right_tracker = ArmCurlTracker("RIGHT")
    prev_l_elbow  = None
    prev_r_elbow  = None

    with _MP.Pose(
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
        model_complexity=1,
    ) as pose:

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return {"left_reps": 0, "right_reps": 0, "exit": "quit"}

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        prev_time   = time.time()
        exit_reason = "quit"
        print("[INFO] Dumbbell Curl started.  R=reset  M=menu  Q/ESC=quit")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            curr = time.time()
            fps  = 1.0 / max(curr - prev_time, 1e-9)
            prev_time = curr

            # Process RAW frame for correct LEFT/RIGHT anatomy
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            # Flip for mirror display
            frame = cv2.flip(frame, 1)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                # Mirror skeleton x-coords for flipped display
                mirrored = copy.deepcopy(results.pose_landmarks)
                for lm in mirrored.landmark:
                    lm.x = 1.0 - lm.x
                _DRW.draw_landmarks(frame, mirrored, _MP.POSE_CONNECTIONS,
                                    _LM_STYLE, _CON_STYLE)

                # ── LEFT arm ─────────────────────────────────────────
                if check_visibility(lms, [L_SHOULDER, L_ELBOW, L_WRIST], VISIBILITY_THRESH):
                    left_tracker.mark_visible()
                    ls = get_landmark_coords(lms, L_SHOULDER, frame.shape, mirror=True)
                    le = get_landmark_coords(lms, L_ELBOW,    frame.shape, mirror=True)
                    lw = get_landmark_coords(lms, L_WRIST,    frame.shape, mirror=True)
                    drift_ok = (prev_l_elbow is None or
                                abs(le[1] - prev_l_elbow[1]) <= MAX_ELBOW_DRIFT_PX)
                    if drift_ok:
                        l_angle = calculate_angle(ls, le, lw)
                        left_tracker.update(l_angle)
                        draw_angle_arc(frame, le, l_angle, active=True)
                    else:
                        draw_angle_arc(frame, le, left_tracker.current_angle, active=True)
                    prev_l_elbow = le
                else:
                    left_tracker.mark_invisible()
                    prev_l_elbow = None

                # ── RIGHT arm ────────────────────────────────────────
                if check_visibility(lms, [R_SHOULDER, R_ELBOW, R_WRIST], VISIBILITY_THRESH):
                    right_tracker.mark_visible()
                    rs = get_landmark_coords(lms, R_SHOULDER, frame.shape, mirror=True)
                    re = get_landmark_coords(lms, R_ELBOW,    frame.shape, mirror=True)
                    rw = get_landmark_coords(lms, R_WRIST,    frame.shape, mirror=True)
                    drift_ok = (prev_r_elbow is None or
                                abs(re[1] - prev_r_elbow[1]) <= MAX_ELBOW_DRIFT_PX)
                    if drift_ok:
                        r_angle = calculate_angle(rs, re, rw)
                        right_tracker.update(r_angle)
                        draw_angle_arc(frame, re, r_angle, active=True)
                    else:
                        draw_angle_arc(frame, re, right_tracker.current_angle, active=True)
                    prev_r_elbow = re
                else:
                    right_tracker.mark_invisible()
                    prev_r_elbow = None

            else:
                left_tracker.mark_invisible()
                right_tracker.mark_invisible()
                prev_l_elbow = prev_r_elbow = None
                cv2.putText(frame, "No person detected — move into frame",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 100, 220), 2)

            render_overlay(frame, left_tracker, right_tracker, fps)
            last_rendered_frame = frame.copy()
            cv2.imshow("AI Gym Trainer — Dumbbell Curl", frame)

            # Trigger Goku UI effect when 10 reps are completed!
            if left_tracker.should_play_goku_effect or right_tracker.should_play_goku_effect:
                left_tracker.should_play_goku_effect = False
                right_tracker.should_play_goku_effect = False
                play_goku_ui_video_effect(window_name="AI Gym Trainer — Dumbbell Curl")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                left_tracker.reset()
                right_tracker.reset()
                prev_l_elbow = prev_r_elbow = None
                print("[INFO] Reps reset.")
            elif key == ord('m'):
                exit_reason = "menu"
                break

        # Show Exit Evaluation Screen with Red Sign if < 10 reps (and play sound)
        if last_rendered_frame is not None:
            max_reps = max(left_tracker.rep_count, right_tracker.rep_count)
            max_pace = max(left_tracker.session.avg_pace, right_tracker.session.avg_pace)
            max_streak = max(left_tracker.session.streak, right_tracker.session.streak)
            show_exit_evaluation(last_rendered_frame, "Dumbbell Curl", max_reps,
                                 target_reps=10, window_name="AI Gym Trainer — Dumbbell Curl",
                                 pace_avg=max_pace, streak=max_streak)

        cap.release()
        cv2.destroyAllWindows()

    max_reps = max(left_tracker.rep_count, right_tracker.rep_count)
    print("\n===== Dumbbell Curl Summary =====")
    print(f"  LEFT  arm reps : {left_tracker.rep_count} / {left_tracker.target_reps}")
    print(f"  RIGHT arm reps : {right_tracker.rep_count} / {right_tracker.target_reps}")
    if max_reps < 10:
        print("  Target         : NOT ACHIEVED (Red Sign Alert \u2716 - Needed 10)")
    else:
        print("  Target         : ACHIEVED! (Full 10 Reps Complete \u2605)")
    print("==================================")
    return {
        "left_reps":  left_tracker.rep_count,
        "right_reps": right_tracker.rep_count,
        "reps":       max_reps,
        "exit":       exit_reason,
    }
