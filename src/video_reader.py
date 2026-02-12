from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecodeConfig:
    backend: str = "opencv"
    buffer_size: int = 8
    drop_policy: str = "drop_oldest"


@dataclass
class FramePacket:
    frame: np.ndarray
    source_ts_ms: int | None
    capture_ts_ms: int
    frame_index: int


class BufferedVideoReader:
    """Decode frames in a dedicated thread and expose buffered pop API."""

    def __init__(
        self,
        source: str,
        config: DecodeConfig | None = None,
        reconnect_sleep_s: float = 2.0,
    ) -> None:
        self.source_arg = source
        self.config = config or DecodeConfig()
        self.reconnect_sleep_s = reconnect_sleep_s

        self._buffer: deque[FramePacket] = deque()
        self._condition = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None

        self._decoded_frames = 0
        self._dropped_frames = 0
        self._reconnects = 0
        self._backend_used = self.config.backend.lower()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._decode_loop, name="video-decoder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def pop_frame(self, timeout_s: float = 1.0, prefer_latest: bool = True) -> FramePacket | None:
        deadline = time.time() + max(0.0, timeout_s)
        with self._condition:
            while self._running and not self._buffer:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

            if not self._buffer:
                return None

            if prefer_latest:
                packet = self._buffer[-1]
                self._buffer.clear()
            else:
                packet = self._buffer.popleft()
            return packet

    def stats(self) -> dict[str, Any]:
        with self._condition:
            buffered = len(self._buffer)
        return {
            "backend": self._backend_used,
            "decoded_frames": self._decoded_frames,
            "dropped_frames": self._dropped_frames,
            "buffered_frames": buffered,
            "reconnects": self._reconnects,
        }

    def _decode_loop(self) -> None:
        while self._running:
            try:
                backend = self.config.backend.lower()
                if backend == "pyav":
                    try:
                        self._decode_with_pyav()
                        self._backend_used = "pyav"
                    except Exception as exc:
                        logger.warning("pyav decode failed, fallback to opencv: %s", exc)
                        self._backend_used = "opencv"
                        self._decode_with_opencv()
                else:
                    self._backend_used = "opencv"
                    self._decode_with_opencv()
            except Exception as exc:
                logger.warning("decode loop error: %s", exc)

            if not self._running:
                break
            self._reconnects += 1
            time.sleep(self.reconnect_sleep_s)

    def _decode_with_opencv(self) -> None:
        capture = cv2.VideoCapture(self._resolve_source())
        if not capture.isOpened():
            logger.warning("failed to open source with OpenCV: %s", self.source_arg)
            return

        frame_idx = 0
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    if self._looks_like_file():
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.01)
                        continue
                    break

                frame_idx += 1
                ts_pos = capture.get(cv2.CAP_PROP_POS_MSEC)
                source_ts_ms = int(ts_pos) if ts_pos > 0 else None
                packet = FramePacket(
                    frame=frame,
                    source_ts_ms=source_ts_ms,
                    capture_ts_ms=int(time.time() * 1000),
                    frame_index=frame_idx,
                )
                self._push(packet)
        finally:
            capture.release()

    def _decode_with_pyav(self) -> None:
        import av

        if self.source_arg.isdigit():
            raise RuntimeError("pyav backend does not support camera index source")

        options = {
            "fflags": "nobuffer",
            "flags": "low_delay",
        }
        if str(self.source_arg).startswith("rtsp://"):
            options["rtsp_transport"] = "tcp"

        container = av.open(self.source_arg, options=options)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        frame_idx = 0
        try:
            while self._running:
                has_frame = False
                for frame in container.decode(video=0):
                    has_frame = True
                    if not self._running:
                        break
                    arr = frame.to_ndarray(format="bgr24")
                    source_ts_ms = int(float(frame.time) * 1000) if frame.time is not None else None
                    frame_idx += 1
                    packet = FramePacket(
                        frame=arr,
                        source_ts_ms=source_ts_ms,
                        capture_ts_ms=int(time.time() * 1000),
                        frame_index=frame_idx,
                    )
                    self._push(packet)

                if not has_frame:
                    if self._looks_like_file():
                        container.seek(0)
                        continue
                    break
        finally:
            container.close()

    def _push(self, packet: FramePacket) -> None:
        buffer_size = max(1, int(self.config.buffer_size))
        policy = self.config.drop_policy.lower()

        with self._condition:
            if len(self._buffer) >= buffer_size:
                self._dropped_frames += 1
                if policy == "drop_oldest":
                    self._buffer.popleft()
                else:
                    self._decoded_frames += 1
                    self._condition.notify_all()
                    return

            self._buffer.append(packet)
            self._decoded_frames += 1
            self._condition.notify_all()

    def _resolve_source(self) -> str | int:
        if self.source_arg.isdigit():
            return int(self.source_arg)
        return self.source_arg

    def _looks_like_file(self) -> bool:
        source = self.source_arg
        return ("://" not in source) and (not source.isdigit())
