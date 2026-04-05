import cv2
import numpy as np
from ultralytics import YOLO
import config


class EnemyDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        # Resolve target class index from the model's names dict
        self._target_idx = self._resolve_class_index()
        print(f"[Detector] Model loaded. Target class: '{config.TARGET_CLASS}' → index {self._target_idx}")

    def _resolve_class_index(self):
        names = self.model.names  # {0: 'enemy', 1: 'teammate', ...}
        for idx, name in names.items():
            if name.lower() == config.TARGET_CLASS.lower():
                return idx
        # If not found, warn and fall back to 0
        print(f"[Detector] WARNING: class '{config.TARGET_CLASS}' not found in model. Falling back to index 0.")
        return 0

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Run inference on a BGR frame.
        Returns:
            enemy_found (bool): True if at least one enemy detected above threshold.
            annotated  (np.ndarray): Frame with bounding boxes drawn.
        """
        results = self.model(
            frame,
            imgsz=config.INFERENCE_RESIZE,
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        enemy_found = False
        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            if cls_id != self._target_idx:
                continue

            enemy_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"enemy {conf:.2f}"
            cv2.putText(
                annotated, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2,
            )

        return enemy_found, annotated
