import threading
import time
import pygame
import config


class AlertPlayer:
    """Non-blocking audio alert with cooldown."""

    def __init__(self):
        pygame.mixer.init()
        self._sound = pygame.mixer.Sound(config.ALERT_SOUND_PATH)
        self._last_played = 0.0
        self._lock = threading.Lock()

    def trigger(self):
        """Play alert if cooldown has elapsed. Non-blocking."""
        now = time.time()
        with self._lock:
            if now - self._last_played >= config.ALERT_COOLDOWN_SEC:
                self._last_played = now
                threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        self._sound.play()
        pygame.time.wait(int(self._sound.get_length() * 1000))
