"""
Background streaming runner for the dashboard device agent.

This module captures frames from a local video file, runs the existing
YOLO + occupancy + visualization pipeline, and stores the latest annotated
JPEG frame for MJPEG streaming.
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

import config

# Make project root importable when running from device-agent directory.
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.quantum.quantum_counting import compute_classical_count, quantum_counting
from src.vision.boxes_to_occupancy import boxes_to_occupancy, directional_occupancy
from src.vision.video_processor import VideoProcessor
from src.vision.visualization import create_visualization

log = logging.getLogger(__name__)


class PipelineStreamRunner:
    """Owns the lifecycle of the background vision stream pipeline."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._running = False
        self._frames_processed = 0
        self._last_error: Optional[str] = None

        self._latest_jpeg: Optional[bytes] = None
        self._state_lock = threading.Lock()

    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    @property
    def last_error(self) -> Optional[str]:
        with self._state_lock:
            return self._last_error

    def start(self) -> tuple[bool, str]:
        if self.is_running():
            return False, "Already running"

        if not _is_power_of_two(config.ROWS * config.COLS):
            return False, "Grid size ROWS*COLS must be a power of 2"

        self._stop_event.clear()
        with self._state_lock:
            self._running = True
            self._frames_processed = 0
            self._last_error = None
            self._latest_jpeg = None

        self._thread = threading.Thread(target=self._run_loop, name="stream-runner", daemon=True)
        self._thread.start()

        # Give the thread a short moment to fail fast on config/model/source errors.
        time.sleep(0.2)
        if not self.is_running():
            return False, self.last_error or "Failed to start stream"

        return True, "Stream started"

    def stop(self) -> tuple[bool, str]:
        if not self.is_running():
            return False, "Not running"

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

        with self._state_lock:
            self._running = False
            self._latest_jpeg = None

        return True, "Stopped"

    def mjpeg_chunks(self) -> Iterator[bytes]:
        delay = 1.0 / max(config.TARGET_FPS, 1)
        while True:
            frame = self._get_latest_or_placeholder()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
            time.sleep(delay)

    def _set_error(self, message: str) -> None:
        with self._state_lock:
            self._last_error = message
            self._running = False
            self._latest_jpeg = None

    def _set_latest_frame(self, frame_jpeg: bytes) -> None:
        with self._state_lock:
            self._latest_jpeg = frame_jpeg
            self._frames_processed += 1

    def _get_latest_or_placeholder(self) -> bytes:
        with self._state_lock:
            running = self._running
            err = self._last_error
            latest = self._latest_jpeg

        if err:
            text = f"ERROR: {err}"
        elif running and latest is not None:
            return latest
        elif running:
            text = "Starting stream..."
        else:
            text = "Stream offline. Press Start in dashboard."

        return _build_placeholder_jpeg(text)

    def _run_loop(self) -> None:
        quantum_executor: Optional[ThreadPoolExecutor] = None
        quantum_future: Optional[Future] = None
        capture = None

        try:
            source_path = _resolve_video_source(config.VIDEO_SOURCE)

            log.info("Opening video file: %s", source_path)
            if not source_path.exists():
                raise RuntimeError(f"Video file not found: {source_path}")

            capture = cv2.VideoCapture(str(source_path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video file: {source_path}")

            processor = VideoProcessor(
                model_path=config.MODEL_PATH,
                confidence_threshold=config.CONFIDENCE_THRESHOLD,
                device=config.YOLO_DEVICE,
            )

            n_regions = config.ROWS * config.COLS
            target_frame_duration = 1.0 / max(config.TARGET_FPS, 1)

            last_quantum_density = None
            last_quantum_count = None
            frames_since_quantum = config.QUANTUM_EVERY_N

            if config.USE_QUANTUM:
                quantum_executor = ThreadPoolExecutor(max_workers=1)

            frame_number = 0

            while not self._stop_event.is_set():
                loop_started = time.perf_counter()

                ok, frame = capture.read()
                if not ok:
                    if config.LOOP_VIDEO:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                frame_h, frame_w = frame.shape[:2]
                result = processor.process_frame(frame, frame_number=frame_number)
                frame_number += 1

                occupancy = boxes_to_occupancy(
                    result.boxes_xyxy,
                    config.ROWS,
                    config.COLS,
                    frame_w,
                    frame_h,
                )
                classical_count = compute_classical_count(occupancy)
                classical_density = classical_count / n_regions

                direction_data = None
                if config.DIRECTION_SPLIT:
                    direction_data = directional_occupancy(
                        result.boxes_xyxy,
                        config.ROWS,
                        config.COLS,
                        frame_w,
                        frame_h,
                        split=config.DIRECTION_SPLIT,
                    )

                if config.USE_QUANTUM and quantum_executor is not None:
                    if quantum_future is None and frames_since_quantum >= max(config.QUANTUM_EVERY_N, 1):
                        quantum_future = quantum_executor.submit(
                            quantum_counting,
                            list(occupancy),
                            config.PRECISION_QUBITS,
                            config.SHOTS,
                        )
                        frames_since_quantum = 0
                    else:
                        frames_since_quantum += 1

                    if quantum_future is not None and quantum_future.done():
                        try:
                            last_quantum_count, last_quantum_density, _ = quantum_future.result()
                        except Exception as exc:
                            log.warning("Quantum step failed: %s", exc)
                            last_quantum_count, last_quantum_density = None, None
                        finally:
                            quantum_future = None

                labels = [d.class_name for d in result.detections]
                confidences = [d.confidence for d in result.detections]

                vis_frame = create_visualization(
                    frame=result.frame,
                    boxes=result.boxes_xyxy,
                    occupancy=occupancy,
                    rows=config.ROWS,
                    cols=config.COLS,
                    classical_density=classical_density,
                    quantum_density=last_quantum_density,
                    quantum_count=last_quantum_count,
                    labels=labels,
                    confidences=confidences,
                    direction_data=direction_data,
                    show_info=config.SHOW_INFO,
                )

                ok_jpg, buf = cv2.imencode(
                    ".jpg",
                    vis_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(config.JPEG_QUALITY)],
                )
                if ok_jpg:
                    self._set_latest_frame(buf.tobytes())

                elapsed = time.perf_counter() - loop_started
                sleep_for = target_frame_duration - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except Exception as exc:
            log.exception("Stream loop crashed")
            self._set_error(str(exc))
        finally:
            if quantum_future is not None:
                quantum_future.cancel()
            if quantum_executor is not None:
                quantum_executor.shutdown(wait=False)
            if capture is not None:
                capture.release()
            with self._state_lock:
                self._running = False


def _resolve_video_source(raw_source: str) -> Path:
    candidate = Path(str(raw_source).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate.resolve(strict=False)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def _build_placeholder_jpeg(message: str) -> bytes:
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        config.DEVICE_NAME,
        (24, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 220, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message[:72],
        (24, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        time.strftime("%Y-%m-%d %H:%M:%S"),
        (24, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return b""
    return buf.tobytes()
