"""
AI Gym Trainer — Goku Ultra Instinct Video Effect Engine
========================================================
Plays the user-provided 'goku ui.mp4' video clip with its original
soundtrack whenever 10 reps are completed in any exercise.
"""

import os
import time
import cv2

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(_BASE_DIR, "goku ui.mp4")
AUDIO_PATH = os.path.join(_BASE_DIR, "assets", "goku_ui_audio.wav")


def play_goku_ui_sound(blocking: bool = False):
    """Play the authentic Goku UI audio track extracted from the video."""
    if not _HAS_WINSOUND or not os.path.exists(AUDIO_PATH):
        return None

    flags = winsound.SND_FILENAME
    if not blocking:
        flags |= winsound.SND_ASYNC
    winsound.PlaySound(AUDIO_PATH, flags)
    return None


def play_goku_ui_video_effect(window_name: str = None, target_size: tuple = None) -> bool:
    """
    Play 'goku ui.mp4' with synchronized sound directly inside the OpenCV window.
    - User can watch the full effect (~8.2 seconds).
    - Can be skipped at any moment by pressing SPACE, ESC, or 'q'.
    - Automatically cleans up and returns focus to the workout.
    """
    if not os.path.exists(VIDEO_PATH):
        print(f"[WARN] Video not found: {VIDEO_PATH}")
        return False

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {VIDEO_PATH}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / fps

    # Start audio track synchronously with video playback
    if _HAS_WINSOUND and os.path.exists(AUDIO_PATH):
        winsound.PlaySound(AUDIO_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)

    start_time = time.time()
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Resize frame if requested
            if target_size is not None:
                frame = cv2.resize(frame, target_size)

            # Draw sleek hint bar at bottom
            h, w = frame.shape[:2]
            cv2.putText(frame, "GOKU ULTRA INSTINCT  |  Press SPACE or ESC to skip",
                        (25, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            if window_name:
                cv2.imshow(window_name, frame)

            # Precise timing sync to video FPS
            expected_time = frame_idx * frame_interval
            elapsed_time = time.time() - start_time
            sleep_ms = int(max(1, (expected_time - elapsed_time) * 1000))

            k = cv2.waitKey(sleep_ms) & 0xFF
            if k in (ord('q'), 27, ord(' ')):  # 'q', ESC, or SPACE
                break
    finally:
        cap.release()
        # Cleanly stop audio if ended early
        if _HAS_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    return True
