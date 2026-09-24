"""
AI Gym Trainer — Professional HUD, Rep Counting & Target Engine
================================================================
Provides state-of-the-art visual telemetry, circular progress gauges,
target goal (10 reps) tracking with red warning signs if not met,
celebration animations, tempo tracking, and audio chimes.
"""

import time
import math
import threading
import cv2
import numpy as np

DEFAULT_TARGET_REPS = 10

from goku_effect import (
    play_goku_ui_sound,
    play_goku_ui_video_effect,
)

# Optional audio feedback (built into Windows Python, runs in background thread)
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False


def play_rep_chime(frequency: int = 1050, duration_ms: int = 90):
    """Play a gym audio chime asynchronously without blocking video processing."""
    if not _HAS_WINSOUND:
        return None

    def _beep():
        try:
            winsound.Beep(frequency, duration_ms)
        except Exception:
            pass

    t = threading.Thread(target=_beep, daemon=True)
    t.start()
    return t


def play_rep_below_10_sound(rep_count: int = 1):
    """
    Audio feedback played whenever a rep is counted and total reps are BELOW 10.
    Produces a crisp dual-tone ascending chime indicating progress towards the 10-rep target.
    """
    if not _HAS_WINSOUND:
        return None

    def _chime():
        try:
            # Ascending dual-tone frequency tied to current rep progress towards 10
            f1 = min(800 + (rep_count % 10) * 25, 1150)
            f2 = min(1020 + (rep_count % 10) * 25, 1350)
            winsound.Beep(f1, 55)
            winsound.Beep(f2, 65)
        except Exception:
            pass

    t = threading.Thread(target=_chime, daemon=True)
    t.start()
    return t


def play_target_completed_sound(blocking: bool = False):
    """
    Triumphant ascending fanfare played when 10 reps are achieved.
    Notes: G4 -> B4 -> D5 -> G5 [784Hz, 988Hz, 1175Hz, 1568Hz].
    """
    if not _HAS_WINSOUND:
        return None

    def _fanfare():
        try:
            tones = [(784, 110), (988, 110), (1175, 110), (1568, 280)]
            for freq, dur in tones:
                winsound.Beep(freq, dur)
                time.sleep(0.04)
        except Exception:
            pass

    t = threading.Thread(target=_fanfare, daemon=True)
    t.start()
    if blocking:
        t.join()
    return t


def play_target_not_met_sound(blocking: bool = False):
    """
    Distinct warning buzzer played when reps are below the target goal of 10.
    Descending alert notes: [520Hz, 370Hz, 290Hz].
    """
    if not _HAS_WINSOUND:
        return None

    def _buzzer():
        try:
            tones = [(520, 160), (370, 240), (290, 300)]
            for freq, dur in tones:
                winsound.Beep(freq, dur)
                time.sleep(0.05)
        except Exception:
            pass

    t = threading.Thread(target=_buzzer, daemon=True)
    t.start()
    if blocking:
        t.join()
    return t


# ─────────────────────────────────────────────
#  Session Statistics & Cadence Tracker
# ─────────────────────────────────────────────

class WorkoutSession:
    """
    Maintains workout metrics:
      - Target goal: 10 reps
      - Elapsed workout duration
      - Per-rep cadence / tempo (seconds per rep)
      - Rep streak
      - Target achieved flag
      - Rep completion pop animation trigger
    """

    def __init__(self, target_reps: int = DEFAULT_TARGET_REPS):
        self.target_reps      = target_reps
        self.start_time       = time.time()
        self.last_rep_time    = time.time()
        self.last_rep_pace    = 0.0
        self.total_reps       = 0
        self.streak           = 0
        self.pop_until        = 0.0
        self.rep_durations    = []
        self.target_achieved  = False
        self.should_play_goku_effect = False

    def on_rep_completed(self, new_rep_count: int):
        """Call whenever rep counter increments."""
        now = time.time()
        self.total_reps = new_rep_count
        self.streak += 1

        if self.last_rep_time > 0:
            pace = now - self.last_rep_time
            if 0.5 < pace < 15.0:
                self.last_rep_pace = pace
                self.rep_durations.append(pace)

        self.last_rep_time = now
        self.pop_until     = now + 1.25

        # 10 reps done -> trigger Goku UI video effect!
        if self.total_reps >= self.target_reps and not self.target_achieved:
            self.target_achieved = True
            self.should_play_goku_effect = True
        elif self.total_reps < self.target_reps:
            # Sound for reps below 10
            play_rep_below_10_sound(self.total_reps)
        else:
            # Beyond 10 reps: high gold celebration chime
            play_rep_chime(1350, 100)

    @property
    def is_popping(self) -> bool:
        return time.time() < self.pop_until

    @property
    def is_target_met(self) -> bool:
        return self.total_reps >= self.target_reps

    @property
    def elapsed_str(self) -> str:
        elapsed = int(time.time() - self.start_time)
        mins = elapsed // 60
        secs = elapsed % 60
        return f"{mins:02d}:{secs:02d}"

    @property
    def avg_tempo_str(self) -> str:
        if self.last_rep_pace > 0:
            return f"{self.last_rep_pace:.1f}s"
        return "--"

    @property
    def avg_pace(self) -> float:
        if self.rep_durations:
            return float(sum(self.rep_durations) / len(self.rep_durations))
        return float(self.last_rep_pace)

    def reset(self):
        self.start_time              = time.time()
        self.last_rep_time           = time.time()
        self.last_rep_pace           = 0.0
        self.total_reps              = 0
        self.streak                  = 0
        self.pop_until               = 0.0
        self.target_achieved         = False
        self.should_play_goku_effect = False
        self.rep_durations.clear()


# ─────────────────────────────────────────────
#  Professional UI Drawing Helpers
# ─────────────────────────────────────────────

def draw_glass_panel(img: np.ndarray,
                     pt1: tuple,
                     pt2: tuple,
                     bg_color: tuple = (16, 18, 24),
                     alpha: float = 0.76,
                     radius: int = 14,
                     border_color: tuple = None,
                     border_thickness: int = 1):
    """Draw a frosted rounded glass panel with optional glowing border."""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return

    overlay = img.copy()
    r = radius

    # Fill rounded rectangle
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), bg_color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), bg_color, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(overlay, (cx, cy), r, bg_color, -1)

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # Optional border
    if border_color is not None:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), border_color, border_thickness, cv2.LINE_AA)
        for cx, cy, sa in [(x1+r, y1+r, 180), (x2-r, y1+r, 270),
                           (x2-r, y2-r, 0),   (x1+r, y2-r, 90)]:
            cv2.ellipse(img, (cx, cy), (r, r), 0, sa, sa + 90, border_color, border_thickness, cv2.LINE_AA)


def draw_circular_rep_gauge(frame: np.ndarray,
                            center: tuple,
                            radius: int,
                            rep_count: int,
                            progress_pct: float,
                            is_pop: bool = False,
                            accent_color: tuple = (255, 185, 30),
                            target_reps: int = DEFAULT_TARGET_REPS):
    """
    Draw a commercial fitness style circular rep gauge showing current count of target.
    """
    cx, cy = center
    progress = max(0.0, min(progress_pct, 1.0))

    # Inner circular dark plate
    plate_overlay = frame.copy()
    cv2.circle(plate_overlay, (cx, cy), radius + 10, (14, 16, 22), -1)
    cv2.addWeighted(plate_overlay, 0.78, frame, 0.22, 0, frame)

    # Outer track ring
    cv2.circle(frame, (cx, cy), radius, (38, 42, 52), 7, cv2.LINE_AA)

    # Dynamic Progress Arc
    arc_deg = int(progress * 360)
    if arc_deg > 0:
        if progress >= 0.90:
            arc_clr = (0, 225, 130)  # Bright green at peak
        else:
            arc_clr = accent_color

        cv2.ellipse(frame, (cx, cy), (radius, radius),
                    0, -90, -90 + arc_deg, arc_clr, 7, cv2.LINE_AA)

        rad = math.radians(-90 + arc_deg)
        dot_x = int(cx + radius * math.cos(rad))
        dot_y = int(cy + radius * math.sin(rad))
        cv2.circle(frame, (dot_x, dot_y), 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Celebration pop effect
    if is_pop:
        halo_clr = (0, 225, 130) if rep_count >= target_reps else (255, 215, 0)
        cv2.circle(frame, (cx, cy), radius + 8, halo_clr, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius + 14, (0, 220, 255), 1, cv2.LINE_AA)

        badge_w, badge_h = 110, 26
        bx = cx - badge_w // 2
        by = cy - radius - 32
        badge_bg = (0, 180, 100) if rep_count >= target_reps else (0, 140, 220)
        draw_glass_panel(frame, (bx, by), (bx + badge_w, by + badge_h),
                         bg_color=badge_bg, alpha=0.92, radius=8,
                         border_color=(255, 255, 255), border_thickness=1)

        pop_msg = "GOAL MET! \u2605" if rep_count >= target_reps else "+1 REP! \u2605"
        cv2.putText(frame, pop_msg, (bx + 8, by + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Center Rep Count text
    num_str = str(rep_count)
    font_scale = 1.9 if len(num_str) <= 2 else 1.4
    thickness = 3
    t_size = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
    num_x = cx - t_size[0] // 2
    num_y = cy + t_size[1] // 2 - 5

    num_clr = (0, 240, 140) if rep_count >= target_reps else ((255, 225, 0) if is_pop else (255, 255, 255))
    cv2.putText(frame, num_str, (num_x, num_y),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, num_clr, thickness, cv2.LINE_AA)

    # Target subtitle label (e.g. "of 10 REPS")
    lbl_text = f"of {target_reps} REPS"
    lbl_size = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    lbl_clr = (0, 220, 130) if rep_count >= target_reps else (180, 185, 200)
    cv2.putText(frame, lbl_text, (cx - lbl_size[0] // 2, cy + radius - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, lbl_clr, 1, cv2.LINE_AA)


def draw_target_sign(frame: np.ndarray,
                     x: int,
                     y: int,
                     reps: int,
                     target: int = DEFAULT_TARGET_REPS,
                     width: int = 188,
                     height: int = 42):
    """
    Draw a prominent Target sign.
    - If reps < target: Displays a bright RED SIGN / alert badge.
    - If reps >= target: Displays an Emerald Green / Gold achieved badge.
    """
    is_met = reps >= target

    if is_met:
        # Green / Gold Achieved Sign
        bg_col     = (18, 38, 22)
        border_col = (0, 215, 120)
        badge_txt  = f"\u2713 TARGET {target} MET!"
        val_txt    = f"{reps} / {target} REPS"
        txt_clr    = (0, 230, 130)
    else:
        # RED WARNING SIGN
        bg_col     = (20, 18, 45)       # dark red/crimson
        border_col = (45, 45, 235)      # Bright Red in BGR
        badge_txt  = f"[!] TARGET: {target} (UNDER)"
        val_txt    = f"{reps} / {target} REPS"
        txt_clr    = (70, 90, 255)      # Red/coral text

    draw_glass_panel(frame, (x, y), (x + width, y + height),
                     bg_color=bg_col, alpha=0.88, radius=8,
                     border_color=border_col, border_thickness=2 if not is_met else 1)

    cv2.putText(frame, badge_txt, (x + 8, y + 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, txt_clr, 1, cv2.LINE_AA)
    cv2.putText(frame, val_txt, (x + 8, y + 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255) if is_met else txt_clr, 1, cv2.LINE_AA)


def draw_metric_pill(frame: np.ndarray,
                     x: int,
                     y: int,
                     label: str,
                     value: str,
                     color: tuple = (255, 185, 30),
                     width: int = 110,
                     height: int = 40):
    """Draw a rounded stats pill (e.g. TEMPO: 2.1s, STREAK: 4)."""
    draw_glass_panel(frame, (x, y), (x + width, y + height),
                     bg_color=(20, 22, 30), alpha=0.75, radius=8,
                     border_color=(50, 54, 66), border_thickness=1)

    cv2.putText(frame, label.upper(), (x + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 145, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x + 10, y + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, color, 1, cv2.LINE_AA)


def draw_battery_rep_dashboard(frame: np.ndarray,
                               reps: int,
                               target_reps: int = DEFAULT_TARGET_REPS,
                               x: int = None,
                               y: int = None,
                               label: str = "REP BATTERY",
                               width: int = 236,
                               height: int = 58):
    """
    Renders a futuristic battery stamina dashboard on camera video.
    Color dynamics:
      - Low reps (< 40% of target, e.g. 0-3 reps): Alert RED
      - Some reps (40% - 79% of target, e.g. 4-7 reps): Energy ORANGE
      - Reps completed (>= 80% / 100%, e.g. 8-10 reps): Glowing Emerald GREEN
    Shows battery shell with internal segmented charging cells, exact reps/target,
    and completion percentage.
    """
    h, w = frame.shape[:2]
    if x is None:
        x = w - width - 14
    if y is None:
        y = 7

    x = max(0, min(x, w - width))
    y = max(0, min(y, h - height))

    pct = min(1.0, max(0.0, reps / target_reps if target_reps else 0.0))
    is_met = reps >= target_reps

    # Color thresholding: Red (<40%) -> Orange (40-79%) -> Green (>=80%)
    if reps < target_reps * 0.4:
        clr = (40, 45, 240)    # BGR Alert Red
        state_txt = "LOW"
        status_sub = "STARTING SET"
    elif reps < target_reps * 0.8:
        clr = (0, 160, 255)    # BGR Energy Orange
        state_txt = "CHARGING"
        status_sub = "HALFWAY"
    else:
        clr = (0, 235, 115)    # BGR Emerald Green
        state_txt = "FULL 100%" if is_met else "HIGH"
        status_sub = "GOAL MET!" if is_met else "FINAL PUSH"

    # Frosted dark glass container
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (10, 14, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Accent borders
    if is_met:
        # Triumphant glowing double border
        cv2.rectangle(frame, (x - 1, y - 1), (x + width + 1, y + height + 1), (0, 180, 80), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + width, y + height), clr, 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x, y), (x + width, y + height), clr, 1, cv2.LINE_AA)

    # Top header inside card: Tag + Label and [STATE]
    cv2.putText(frame, f"[BATT] {label}", (x + 9, y + 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, (215, 220, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{state_txt}]", (x + width - 76, y + 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, clr, 1, cv2.LINE_AA)

    # Battery graphic
    bx = x + 9
    by = y + 24
    bw = 80
    bh = 25

    # Positive terminal (nub) on right
    cv2.rectangle(frame, (bx + bw + 1, by + 6), (bx + bw + 5, by + bh - 6), (190, 200, 220), -1)
    # Outer battery shell
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (190, 200, 220), 1, cv2.LINE_AA)

    # Battery internal fill
    inner_w = bw - 4
    inner_h = bh - 4
    fill_w = int(inner_w * pct)
    if fill_w > 0:
        cv2.rectangle(frame, (bx + 2, by + 2), (bx + 2 + fill_w, by + 2 + inner_h), clr, -1)

    # Cell dividers (3 vertical tick lines)
    for frac in (0.25, 0.50, 0.75):
        div_x = bx + 2 + int(inner_w * frac)
        cv2.line(frame, (div_x, by + 2), (div_x, by + 2 + inner_h), (20, 24, 32), 1)

    # Rep count numbers and percentage
    cv2.putText(frame, f"{reps}/{target_reps}", (x + 102, y + 43),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{int(pct * 100)}%", (x + 176, y + 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, clr, 1, cv2.LINE_AA)
    cv2.putText(frame, status_sub, (x + 176, y + 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (170, 175, 190), 1, cv2.LINE_AA)


def show_exit_evaluation(frame: np.ndarray,
                         exercise_name: str,
                         rep_count: int,
                         target_reps: int = DEFAULT_TARGET_REPS,
                         window_name: str = None,
                         duration_sec: float = 2.5,
                         pace_avg: float = 0.0,
                         streak: int = 0):
    """
    Renders an evaluation card directly onto frame before returning to menu:
      - If reps < target: Prominent RED WARNING SIGN with alert details!
                          Plays descending warning buzzer sound.
      - If reps >= target: Emerald green/gold target completed celebration!
                           Plays triumphant Goku UI soundtrack.
      - AI Coach Insight: Evaluates biomechanics, cadence, and form cue via
                          Google Gemini (or offline heuristic engine).
    """
    h, w = frame.shape[:2]
    cw, ch = 560, 310
    cx = (w - cw) // 2
    cy = (h - ch) // 2

    is_met = rep_count >= target_reps

    # Play distinct sound: Goku UI audio for 10 reps, warning buzzer if below 10
    if is_met:
        snd_thread = play_goku_ui_sound()
    else:
        snd_thread = play_target_not_met_sound()

    # Generate AI Coach critique
    coach_feedback = ""
    coach_model = "AI Coach"
    try:
        from llm_coach import evaluate_workout_session, get_coach_status
        coach_feedback = evaluate_workout_session(
            exercise_name=exercise_name,
            rep_count=rep_count,
            target_reps=target_reps,
            pace_avg=pace_avg,
            streak=streak,
        )
        status = get_coach_status()
        coach_model = status.get("model", "AI Coach")
    except Exception:
        coach_feedback = f"[FORM] Maintain steady cadence and strict range of motion on every rep."

    # Print full critique to console
    print("\n" + "=" * 60)
    print(f"  AI FITNESS COACH REPORT ({coach_model})")
    print(f"  SESSION: {exercise_name.upper()} | REPS: {rep_count}/{target_reps}")
    print("=" * 60)
    print(coach_feedback)
    print("=" * 60 + "\n")

    # Parse cues for on-screen display
    cue_lines = [l.strip() for l in coach_feedback.splitlines() if l.strip()]
    form_line = ""
    cadence_line = ""
    for l in cue_lines:
        if "FORM" in l.upper() or "BIOMECHANIC" in l.upper():
            form_line = l
        elif "CADENCE" in l.upper() or "TEMPO" in l.upper():
            cadence_line = l
    if not form_line and cue_lines:
        form_line = cue_lines[0]
    if not cadence_line and len(cue_lines) > 1:
        cadence_line = cue_lines[1]
    elif not cadence_line:
        cadence_line = f"Session pace: {pace_avg:.1f}s/rep | Streak: {streak}"

    disp_line1 = form_line[:64] + ("..." if len(form_line) > 64 else "")
    disp_line2 = cadence_line[:64] + ("..." if len(cadence_line) > 64 else "")

    if is_met:
        card_border = (255, 215, 0)    # Ultra Instinct Gold/Silver
        bg_color    = (12, 16, 28)     # Deep cosmic blue
        sign_col    = (255, 230, 120)
        sign_head   = "\u2605 ULTRA INSTINCT ACHIEVED! \u2605"
        sign_sub    = f"Full 10 reps mastered: {rep_count}/{target_reps} reps."
        icon_txt    = "\u2605"
    else:
        card_border = (45, 45, 235)    # RED SIGN
        bg_color    = (26, 15, 18)
        sign_col    = (60, 75, 255)
        sign_head   = "\u26a0 TARGET NOT ACHIEVED!"
        sign_sub    = f"Goal was 10 reps \u2014 completed only {rep_count} of {target_reps} reps."
        icon_txt    = "\u2716"

    # Frosted background card
    draw_glass_panel(frame, (cx, cy), (cx + cw, cy + ch),
                     bg_color=bg_color, alpha=0.92, radius=16,
                     border_color=card_border, border_thickness=2)

    # Accent top banner
    cv2.line(frame, (cx + 25, cy + 3), (cx + cw - 25, cy + 3), card_border, 3, cv2.LINE_AA)

    # Exercise title
    title_str = f"WORKOUT REPORT  \u2014  {exercise_name.upper()}"
    cv2.putText(frame, title_str, (cx + 30, cy + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (230, 235, 245), 1, cv2.LINE_AA)

    # Prominent sign badge (RED if not met, ULTRA INSTINCT if met)
    badge_w, badge_h = cw - 60, 82
    bx, by = cx + 30, cy + 44
    draw_glass_panel(frame, (bx, by), (bx + badge_w, by + badge_h),
                     bg_color=(35, 15, 20) if not is_met else (15, 25, 45),
                     alpha=0.88, radius=10,
                     border_color=card_border, border_thickness=2)

    # Icon circle
    cv2.circle(frame, (bx + 38, by + 41), 22, card_border, -1, cv2.LINE_AA)
    cv2.putText(frame, icon_txt, (bx + 28, by + 51),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    # Sign text
    cv2.putText(frame, sign_head, (bx + 72, by + 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, sign_col, 2, cv2.LINE_AA)
    cv2.putText(frame, sign_sub, (bx + 72, by + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 220, 235), 1, cv2.LINE_AA)

    # ── AI Coach Biomechanical Insight Box ───────────────────────────
    ai_w, ai_h = cw - 60, 80
    ax, ay = cx + 30, cy + 134
    draw_glass_panel(frame, (ax, ay), (ax + ai_w, ay + ai_h),
                     bg_color=(18, 22, 36), alpha=0.88, radius=10,
                     border_color=(235, 90, 160), border_thickness=1)

    ai_header = f"[AI COACH] BIOMECHANICS & CADENCE  \u2014  {coach_model}"
    cv2.putText(frame, ai_header, (ax + 14, ay + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 145, 205), 1, cv2.LINE_AA)
    cv2.putText(frame, disp_line1, (ax + 14, ay + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 235, 245), 1, cv2.LINE_AA)
    cv2.putText(frame, disp_line2, (ax + 14, ay + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (185, 195, 215), 1, cv2.LINE_AA)

    # ── Rep progress status bar at bottom of card ────────────────────
    bar_x1, bar_y1 = cx + 30, cy + 226
    bar_x2, bar_y2 = cx + cw - 30, cy + 238
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (38, 42, 52), -1)

    pct = min(rep_count / float(target_reps), 1.0)
    fill = int(bar_x1 + pct * (bar_x2 - bar_x1))
    if fill > bar_x1:
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill, bar_y2), card_border, -1)

    prog_txt = f"{rep_count} / {target_reps} Reps ({int(pct * 100)}%)"
    cv2.putText(frame, prog_txt, (cx + 30, cy + 260),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)

    # Bottom exit note
    cv2.putText(frame, "Returning to menu...", (cx + cw - 170, cy + 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 145, 160), 1, cv2.LINE_AA)

    if window_name:
        start_t = time.time()
        while time.time() - start_t < duration_sec:
            cv2.imshow(window_name, frame)
            k = cv2.waitKey(40) & 0xFF
            if k in (ord('q'), 27, ord(' ')):
                break
        if snd_thread is not None and hasattr(snd_thread, 'is_alive') and snd_thread.is_alive():
            snd_thread.join(timeout=1.2)

    return snd_thread
