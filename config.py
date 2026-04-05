# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = "weights/best.pt"
CONFIDENCE_THRESHOLD = 0.50      # minimum confidence to count as a detection
TARGET_CLASS = "enemy"           # class name in your YOLO model (adjust if needed)

# ─── Screen Capture ───────────────────────────────────────────────────────────
# None = full primary monitor. Or pass {"top":0,"left":0,"width":1920,"height":1080}
CAPTURE_REGION = None
INFERENCE_RESIZE = 640           # resize frame to this before inference (speed/accuracy tradeoff)

# ─── Alert ────────────────────────────────────────────────────────────────────
ALERT_SOUND_PATH = "assets/alert.wav"
ALERT_COOLDOWN_SEC = 1.0         # minimum seconds between alerts (avoid spam)

# ─── Display ──────────────────────────────────────────────────────────────────
SHOW_OVERLAY = True              # show a cv2 window with bounding boxes
OVERLAY_WINDOW_NAME = "CS2 Enemy Detector"
