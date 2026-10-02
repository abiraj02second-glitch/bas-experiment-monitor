"""Reliable offline voice/sound agent for BAS notifications.

The agent owns speech synthesis in one worker so API and vision threads never block on
TTS. It supports bounded priority queuing, duplicate suppression, runtime controls,
and a status snapshot suitable for the dashboard.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeechJob:
    text: str
    priority: int
    created: float


class VoiceSoundAgent:
    def __init__(self, enabled: bool = True, volume: float = 1.0,
                 rate: int = 165, voice: Optional[str] = None,
                 max_queue: int = 64):
        self._queue: queue.PriorityQueue[tuple[int, int, SpeechJob]] = queue.PriorityQueue(maxsize=max_queue)
        self._seq = 0
        self._lock = threading.RLock()
        self.enabled = enabled
        self.volume = max(0.0, min(1.0, float(volume)))
        self.rate = max(80, min(300, int(rate)))
        self.voice = voice or ""
        self.max_queue = max_queue
        self.backend = "unavailable"
        self.last_text = ""
        self.last_error = ""
        self.spoken = 0
        self.dropped = 0
        self.started_at = time.time()
        self._last_enqueued: tuple[str, float] = ("", 0.0)
        self._thread = threading.Thread(target=self._loop, name="bas-voice-agent", daemon=True)
        self._thread.start()

    def _init_engine(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("volume", self.volume)
            engine.setProperty("rate", self.rate)
            if self.voice:
                for item in engine.getProperty("voices") or []:
                    if self.voice.lower() in (getattr(item, "id", "") + " " + getattr(item, "name", "")).lower():
                        engine.setProperty("voice", item.id)
                        break
            self.backend = "pyttsx3"
            return engine
        except Exception as exc:
            self.backend = "unavailable"
            self.last_error = str(exc)
            return None

    def _loop(self):
        engine = self._init_engine()
        while True:
            _, _, job = self._queue.get()
            try:
                with self._lock:
                    enabled = self.enabled
                    volume = self.volume
                    rate = self.rate
                if not enabled or not job.text.strip():
                    continue
                if engine is None:
                    continue
                engine.setProperty("volume", volume)
                engine.setProperty("rate", rate)
                engine.say(job.text)
                engine.runAndWait()
                with self._lock:
                    self.last_text = job.text
                    self.spoken += 1
                    self.last_error = ""
            except Exception as exc:
                with self._lock:
                    self.last_error = str(exc)
            finally:
                self._queue.task_done()

    def say(self, text: str, priority: int = 5, dedupe_seconds: float = 1.5) -> bool:
        text = " ".join(str(text or "").split())
        if not text:
            return False
        now = time.time()
        with self._lock:
            if not self.enabled:
                return False
            if text == self._last_enqueued[0] and now - self._last_enqueued[1] < dedupe_seconds:
                return False
            self._last_enqueued = (text, now)
            self._seq += 1
            seq = self._seq
        try:
            self._queue.put_nowait((-int(priority), seq, SpeechJob(text, priority, now)))
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return False

    def configure(self, enabled=None, volume=None, rate=None, voice=None):
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if volume is not None:
                self.volume = max(0.0, min(1.0, float(volume)))
            if rate is not None:
                self.rate = max(80, min(300, int(rate)))
            if voice is not None:
                self.voice = str(voice)

    def clear(self):
        while True:
            try:
                self._queue.get_nowait(); self._queue.task_done()
            except queue.Empty:
                return

    def snapshot(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "backend": self.backend,
                "volume": self.volume,
                "rate": self.rate,
                "voice": self.voice,
                "queued": self._queue.qsize(),
                "spoken": self.spoken,
                "dropped": self.dropped,
                "last_text": self.last_text,
                "last_error": self.last_error,
            }
