"""
AI Gym Trainer — Daily Workout Dashboard & Multi-Exercise Menu
==============================================================
Select an exercise to track in real-time.
Tracks cumulative daily reps against a 10-rep goal per exercise,
and flags remaining exercises with a prominent red warning sign.

Controls:
---------
  1 / 2 / 3     – Launch selected exercise
  4 / C         – AI Fitness Coach & Biomechanics Q&A
  R             – Reset today's cumulative stats
  Q / ESC       – Quit the application
"""

import numpy as np
import cv2

from daily_tracker import (
    load_daily_stats,
    record_exercise_reps,
    reset_daily_stats,
    get_daily_summary,
    DEFAULT_TARGET,
)
from hud_ui import draw_glass_panel

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
CAMERA_INDEX = 0   # change if your webcam is not device 0

_EXERCISES = [
    {
        "key":   "1",
        "id":    "curl",
        "name":  "Dumbbell Curl",
        "desc":  "Target: 10 Reps  \u2022  Biceps isolation & elbow flexion",
        "color": (0, 200, 120),     # BGR green
    },
    {
        "key":   "2",
        "id":    "pec_dec",
        "name":  "Pec Dec Fly",
        "desc":  "Target: 10 Reps  \u2022  Pectoral contraction & arm crossover",
        "color": (0, 165, 255),     # BGR orange
    },
    {
        "key":   "3",
        "id":    "shoulder_press",
        "name":  "Shoulder Press",
        "desc":  "Target: 10 Reps  \u2022  Deltoids & overhead lockout",
        "color": (255, 185,  30),   # BGR cyan
    },
]


def _draw_dashboard_frame() -> np.ndarray:
    """Build and render the daily exercise dashboard (880 x 720 BGR)."""
    W, H = 880, 720
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Gradient background
    for y in range(H):
        v = int(10 + (y / H) * 16)
        canvas[y] = (v + 2, v, v + 4)

    stats = load_daily_stats()
    summary = get_daily_summary(stats)
    date_str = stats.get("date", "")
    exercises_data = stats.get("exercises", {})

    # ── Header Bar (Frosted Glass) ───────────────────────────────────
    draw_glass_panel(canvas, (0, 0), (W, 70), bg_color=(12, 14, 20), alpha=0.88, radius=0)
    cv2.putText(canvas, "AI GYM TRAINER  \u2014  DAILY DASHBOARD",
                (30, 44), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 215, 0), 2, cv2.LINE_AA)

    # Date badge
    date_badge = f"DATE: {date_str}"
    t_size = cv2.getTextSize(date_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    cv2.putText(canvas, date_badge, (W - t_size[0] - 30, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 185, 200), 1, cv2.LINE_AA)

    # ── Overall Daily Completion Battery Logo (Replaces Line) ────────
    total_done = summary["total_done"]
    total_target = summary["total_target"]
    pct = summary["percentage"]

    # Color thresholding: Red (<40%) -> Orange (40-79%) -> Green (>=80%)
    if total_done < total_target * 0.4:
        b_clr = (40, 45, 240)    # BGR Red
        b_state = "LOW BATTERY"
    elif total_done < total_target * 0.8:
        b_clr = (0, 160, 255)    # BGR Orange
        b_state = "CHARGING"
    else:
        b_clr = (0, 235, 115)    # BGR Green
        b_state = "FULL 100% ★" if summary["all_completed"] else "HIGH BATTERY"

    prog_txt = f"DAILY BATTERY: {total_done} / {total_target} REPS ({int(pct * 100)}%)  [{b_state}]"
    cv2.putText(canvas, prog_txt, (32, 94),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, b_clr, 1, cv2.LINE_AA)

    # Horizontal Battery Logo Chassis with Positive Nub
    bx1, by1 = 30, 102
    bx2, by2 = W - 42, 118
    # Outer Battery Chassis
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), b_clr, 1, cv2.LINE_AA)
    # Positive Terminal Nub on Right
    cv2.rectangle(canvas, (bx2 + 1, by1 + 3), (bx2 + 6, by2 - 3), b_clr, -1)
    # Internal liquid fill
    inner_bw = (bx2 - bx1) - 4
    fill_x = bx1 + 2 + int(pct * inner_bw)
    if fill_x > bx1 + 2:
        cv2.rectangle(canvas, (bx1 + 2, by1 + 2), (fill_x, by2 - 2), b_clr, -1)
    # Cell Divider Ticks (25%, 50%, 75%)
    for frac in (0.25, 0.50, 0.75):
        tick_x = bx1 + 2 + int(inner_bw * frac)
        cv2.line(canvas, (tick_x, by1 + 2), (tick_x, by2 - 2), (18, 20, 26), 1)

    # ── Daily Alert / Celebration Banner ─────────────────────────────
    ban_x1, ban_y1 = 30, 126
    ban_x2, ban_y2 = W - 30, 180

    if summary["all_completed"]:
        # Ultra Instinct Goku Mastery celebration banner
        draw_glass_panel(canvas, (ban_x1, ban_y1), (ban_x2, ban_y2),
                         bg_color=(12, 16, 32), alpha=0.92, radius=12,
                         border_color=(255, 215, 0), border_thickness=2)
        cv2.circle(canvas, (ban_x1 + 32, ban_y1 + 27), 18, (255, 215, 0), -1, cv2.LINE_AA)
        cv2.putText(canvas, "\u2605", (ban_x1 + 24, ban_y1 + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)

        cv2.putText(canvas, f"DAILY TARGET ACHIEVED! ALL {total_target} REPS COMPLETED! \u2605",
                    (ban_x1 + 62, ban_y1 + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 230, 110), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Great job today! Prioritize protein intake, hydration, and muscle recovery.",
                    (ban_x1 + 62, ban_y1 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 240, 255), 1, cv2.LINE_AA)

    else:
        # RED WARNING SIGN BANNER: list remaining exercises
        remaining = summary["remaining_exercises"]
        draw_glass_panel(canvas, (ban_x1, ban_y1), (ban_x2, ban_y2),
                         bg_color=(32, 16, 22), alpha=0.92, radius=12,
                         border_color=(45, 45, 235), border_thickness=2)

        # Red warning icon
        cv2.circle(canvas, (ban_x1 + 32, ban_y1 + 27), 18, (45, 45, 235), -1, cv2.LINE_AA)
        cv2.putText(canvas, "!", (ban_x1 + 28, ban_y1 + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        head_txt = f"TARGETS REMAINING: {len(remaining)} EXERCISE(S) LEFT TODAY!"
        cv2.putText(canvas, head_txt,
                    (ban_x1 + 62, ban_y1 + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (65, 85, 255), 1, cv2.LINE_AA)

        # Build list of remaining exercises
        rem_parts = [f"{item['name']} ({item['needed']} reps left)" for item in remaining]
        sub_txt = "Left today: " + "  \u2022  ".join(rem_parts)
        if len(sub_txt) > 85:
            sub_txt = sub_txt[:82] + "..."
        cv2.putText(canvas, sub_txt,
                    (ban_x1 + 62, ban_y1 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 220, 225), 1, cv2.LINE_AA)

    # ── Exercise Cards ───────────────────────────────────────────────
    card_h = 90
    for i, ex in enumerate(_EXERCISES):
        y0 = 186 + i * 98
        color = ex["color"]
        ex_id = ex["id"]

        ex_stats = exercises_data.get(ex_id, {})
        reps = ex_stats.get("reps", 0)
        is_completed = (reps >= DEFAULT_TARGET)

        card_x1, card_x2 = 30, W - 30

        # Card border color: Green if completed, Red if under target, Neutral if not started
        card_border = (0, 215, 120) if is_completed else ((45, 45, 235) if reps > 0 else (50, 54, 66))
        card_bg = (18, 30, 22) if is_completed else ((28, 18, 22) if reps > 0 else (20, 22, 30))

        draw_glass_panel(canvas, (card_x1, y0), (card_x2, y0 + card_h),
                         bg_color=card_bg, alpha=0.82, radius=12,
                         border_color=card_border, border_thickness=2 if not is_completed else 1)

        # Accent left stripe
        cv2.rectangle(canvas, (card_x1, y0 + 8), (card_x1 + 6, y0 + card_h - 8), color, -1)

        # Key badge (filled circle)
        badge_cx, badge_cy = card_x1 + 44, y0 + card_h // 2
        cv2.circle(canvas, (badge_cx, badge_cy), 22, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, ex["key"], (badge_cx - 7, badge_cy + 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.90, (15, 15, 15), 2, cv2.LINE_AA)

        # Name + description
        cv2.putText(canvas, ex["name"], (card_x1 + 82, y0 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (240, 240, 245), 2, cv2.LINE_AA)
        cv2.putText(canvas, ex["desc"], (card_x1 + 82, y0 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 155, 170), 1, cv2.LINE_AA)

        # Mini Battery Logo on Card (Replaces Flat Line)
        ex_pct = min(reps / float(DEFAULT_TARGET), 1.0)
        mb_x1, mb_y1 = card_x1 + 82, y0 + 62
        mb_w, mb_h = 100, 16

        # Color thresholds: Red (<40%) -> Orange (40-79%) -> Green (>=80%)
        if reps < DEFAULT_TARGET * 0.4:
            card_b_clr = (40, 45, 240)    # Red
            card_b_txt = "LOW"
        elif reps < DEFAULT_TARGET * 0.8:
            card_b_clr = (0, 160, 255)    # Orange
            card_b_txt = "MID"
        else:
            card_b_clr = (0, 235, 115)    # Green
            card_b_txt = "FULL" if is_completed else "HIGH"

        # Outer Battery Shell + Nub
        cv2.rectangle(canvas, (mb_x1, mb_y1), (mb_x1 + mb_w, mb_y1 + mb_h), card_b_clr, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (mb_x1 + mb_w + 1, mb_y1 + 3), (mb_x1 + mb_w + 4, mb_y1 + mb_h - 3), card_b_clr, -1)
        # Inner fill
        card_fill = int((mb_w - 4) * ex_pct)
        if card_fill > 0:
            cv2.rectangle(canvas, (mb_x1 + 2, mb_y1 + 2), (mb_x1 + 2 + card_fill, mb_y1 + mb_h - 2), card_b_clr, -1)
        # Cell dividers
        for frac in (0.33, 0.66):
            cx_tick = mb_x1 + 2 + int((mb_w - 4) * frac)
            cv2.line(canvas, (cx_tick, mb_y1 + 2), (cx_tick, mb_y1 + mb_h - 2), (18, 20, 26), 1)

        cv2.putText(canvas, f"[{card_b_txt}] {reps}/{DEFAULT_TARGET} Reps ({int(ex_pct * 100)}%)",
                    (mb_x1 + mb_w + 14, mb_y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, card_b_clr, 1, cv2.LINE_AA)

        # ── Target Status Sign (Right side of card) ───────────────────
        sb_w, sb_h = 240, 52
        sb_x = card_x2 - sb_w - 18
        sb_y = y0 + (card_h - sb_h) // 2

        if is_completed:
            # Green Sign: COMPLETED
            draw_glass_panel(canvas, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h),
                             bg_color=(15, 42, 24), alpha=0.92, radius=8,
                             border_color=(0, 215, 120), border_thickness=2)
            cv2.putText(canvas, "\u2713 TARGET ACHIEVED! \u2605",
                        (sb_x + 12, sb_y + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 235, 135), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Completed {reps} of {DEFAULT_TARGET} reps today",
                        (sb_x + 12, sb_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 220, 210), 1, cv2.LINE_AA)

        elif reps > 0:
            # RED WARNING SIGN: PARTIALLY DONE, TARGET REPS LEFT
            draw_glass_panel(canvas, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h),
                             bg_color=(36, 16, 20), alpha=0.92, radius=8,
                             border_color=(45, 45, 235), border_thickness=2)
            cv2.putText(canvas, f"\u26a0 {DEFAULT_TARGET - reps} REPS LEFT TODAY!",
                        (sb_x + 12, sb_y + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (65, 85, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Target: {DEFAULT_TARGET} reps (Incomplete)",
                        (sb_x + 12, sb_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (225, 200, 205), 1, cv2.LINE_AA)

        else:
            # RED WARNING SIGN: NOT STARTED TODAY
            draw_glass_panel(canvas, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h),
                             bg_color=(28, 16, 20), alpha=0.88, radius=8,
                             border_color=(45, 45, 235), border_thickness=2)
            cv2.putText(canvas, f"\u2716 10 REPS LEFT TODAY",
                        (sb_x + 12, sb_y + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (65, 85, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Not started today \u2014 Tap to launch",
                        (sb_x + 12, sb_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 180, 185), 1, cv2.LINE_AA)

    # ── Card 4: AI Fitness Coach Card ───────────────────────────────
    coach_y0 = 486
    coach_color = (235, 90, 160)
    card_x1, card_x2 = 30, W - 30

    draw_glass_panel(canvas, (card_x1, coach_y0), (card_x2, coach_y0 + card_h),
                     bg_color=(24, 18, 32), alpha=0.88, radius=12,
                     border_color=coach_color, border_thickness=1)

    cv2.rectangle(canvas, (card_x1, coach_y0 + 8), (card_x1 + 6, coach_y0 + card_h - 8), coach_color, -1)

    # Key badge (filled circle)
    badge_cx, badge_cy = card_x1 + 44, coach_y0 + card_h // 2
    cv2.circle(canvas, (badge_cx, badge_cy), 22, coach_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "4", (badge_cx - 7, badge_cy + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.90, (15, 15, 15), 2, cv2.LINE_AA)

    # Name + description
    cv2.putText(canvas, "AI Fitness & Nutrition Coach", (card_x1 + 82, coach_y0 + 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, (245, 240, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Biomechanics form critique & personalized diet plans in table format",
                (card_x1 + 82, coach_y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (195, 185, 215), 1, cv2.LINE_AA)

    # Status sign on right
    sb_w, sb_h = 240, 52
    sb_x = card_x2 - sb_w - 18
    sb_y = coach_y0 + (card_h - sb_h) // 2

    try:
        from llm_coach import get_coach_status
        c_status = get_coach_status()
        is_online = c_status.get("configured", False)
    except Exception:
        is_online = False

    if is_online:
        draw_glass_panel(canvas, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h),
                         bg_color=(15, 36, 32), alpha=0.92, radius=8,
                         border_color=(0, 220, 130), border_thickness=2)
        cv2.putText(canvas, "\u2605 GEMINI AI ONLINE",
                    (sb_x + 12, sb_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 235, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press 4 or C to Chat with Coach",
                    (sb_x + 12, sb_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 235, 225), 1, cv2.LINE_AA)
    else:
        draw_glass_panel(canvas, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h),
                         bg_color=(32, 22, 38), alpha=0.92, radius=8,
                         border_color=(235, 90, 160), border_thickness=1)
        cv2.putText(canvas, "\u26a1 AI COACH READY (Offline)",
                    (sb_x + 12, sb_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 145, 205), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press 4 or C (Key optional in .env)",
                    (sb_x + 12, sb_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (225, 210, 230), 1, cv2.LINE_AA)

    # ── Footer Bar ───────────────────────────────────────────────────
    draw_glass_panel(canvas, (0, H - 42), (W, H), bg_color=(10, 12, 18), alpha=0.75, radius=0)
    hint = "Press 1-3 for Exercise   |   4 or C for AI Coach   |   R = reset stats   |   Q or ESC to quit"
    t_sz = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
    cv2.putText(canvas, hint, ((W - t_sz[0]) // 2, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 165, 180), 1, cv2.LINE_AA)

    return canvas


def show_menu() -> str | None:
    """
    Display the interactive exercise dashboard until user selects an exercise,
    opens the AI Coach console, or quits.

    Returns
    -------
    str  – exercise id ('curl' | 'pec_dec' | 'shoulder_press' | 'coach'), or None to quit.
    """
    key_map = {ord(ex["key"]): ex["id"] for ex in _EXERCISES}
    key_map[ord('4')] = "coach"
    key_map[ord('c')] = "coach"
    key_map[ord('C')] = "coach"

    while True:
        frame = _draw_dashboard_frame()
        cv2.imshow("AI Gym Trainer \u2014 Dashboard", frame)

        k = cv2.waitKey(30) & 0xFF

        if k in key_map:
            cv2.destroyAllWindows()
            return key_map[k]

        if k == ord('r'):
            reset_daily_stats()
            print("[INFO] Today's workout progress has been reset to 0.")

        if k in (ord('q'), 27):
            cv2.destroyAllWindows()
            return None


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  AI Gym Trainer  |  Daily Target Dashboard & Coach")
    print("=" * 52)

    while True:
        choice = show_menu()

        if choice is None:
            print("[INFO] Goodbye!")
            break

        if choice == "coach":
            from llm_coach import run_coach_console
            curr_stats = load_daily_stats()
            curr_summary = get_daily_summary(curr_stats)
            run_coach_console(curr_summary)
            continue

        reps_done = 0
        if choice == "curl":
            from dumbbell_curl import run as run_curl
            result = run_curl(CAMERA_INDEX)
            reps_done = result.get("reps", 0)

        elif choice == "pec_dec":
            from pec_dec_fly import run as run_pec_dec
            result = run_pec_dec(CAMERA_INDEX)
            reps_done = result.get("reps", 0)

        elif choice == "shoulder_press":
            from shoulder_press import run as run_shoulder_press
            result = run_shoulder_press(CAMERA_INDEX)
            reps_done = result.get("reps", 0)

        else:
            result = {"exit": "quit"}

        # Record today's reps in the daily dashboard tracker
        if reps_done > 0:
            record_exercise_reps(choice, reps_done)
            print(f"[INFO] Logged {reps_done} reps for {choice}. Daily dashboard updated.")

        # If all exercises reached 10 reps today, trigger Goku Ultra Instinct video effect!
        new_stats = load_daily_stats()
        new_summary = get_daily_summary(new_stats)
        if new_summary["all_completed"] and reps_done >= 10:
            from goku_effect import play_goku_ui_video_effect
            play_goku_ui_video_effect(window_name="AI Gym Trainer \u2014 Dashboard")

        # 'quit' from inside an exercise → close the app
        # 'menu' from inside an exercise → loop back to dashboard
        if result.get("exit", "quit") == "quit":
            print("[INFO] Goodbye!")
            break


if __name__ == "__main__":
    main()
