"""
CS2 Enemy Alert
───────────────
Captures your screen in real-time, runs a fine-tuned YOLO model to detect
enemies, and plays an alert sound whenever one is spotted.

Usage:
    python main.py

Controls (overlay window must be focused):
    Q  →  quit
"""

import time
import cv2
import mss
import numpy as np

import config
from detector import EnemyDetector
from audio import AlertPlayer


def grab_frame(sct: mss.mss, region=None) -> np.ndarray:
    """Capture screen and return a BGR numpy array."""
    monitor = region if region else sct.monitors[1]  # monitors[1] = primary
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)                      # BGRA
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def main():
    detector = EnemyDetector()
    alert    = AlertPlayer()

    print("[Main] Starting capture loop. Press Q (in overlay window) to quit.")

    with mss.mss() as sct:
        while True:
            t0 = time.perf_counter()

            # 1. Grab frame
            frame = grab_frame(sct, config.CAPTURE_REGION)

            # 2. Detect
            enemy_found, annotated = detector.detect(frame)

            # 3. Alert
            if enemy_found:
                alert.trigger()

            # 4. Optional overlay
            if config.SHOW_OVERLAY:
                # Show FPS
                fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}  {'⚠ ENEMY' if enemy_found else ''}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if not enemy_found else (0, 0, 255),
                    2,
                )
                cv2.imshow(config.OVERLAY_WINDOW_NAME, annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Main] Quit signal received.")
                    break

    cv2.destroyAllWindows()
    print("[Main] Done.")


if __name__ == "__main__":
    main()
