"""
Background streaming runner for the dashboard device agent.

This module captures frames from a local video file, runs the existing
YOLO + occupancy + visualization pipeline, and stores the latest annotated
JPEG frame for MJPEG streaming.
"""

import logging
import queue
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
from src.utils.grafana import push_metrics_to_grafana
from src.vision.boxes_to_occupancy import boxes_to_occupancy, directional_occupancy
from src.vision.video_processor import VideoProcessor
from src.vision.visualization import create_visualization

log = logging.getLogger(__name__)

GRAFANA_QUEUE_MAXSIZE = 256


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

        # Runtime params (set on start, fall back to config defaults)
        self._video_source: str = config.VIDEO_SOURCE
        self._rows: int = config.ROWS
        self._cols: int = config.COLS
        self._direction_split: Optional[str] = config.DIRECTION_SPLIT

    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    @property
    def last_error(self) -> Optional[str]:
        with self._state_lock:
            return self._last_error

    def start(
        self,
        video_source: Optional[str] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        direction_split: Optional[str] = "UNSET",
    ) -> tuple[bool, str]:
        if self.is_running():
            return False, "Already running"

        effective_rows = rows if rows is not None else config.ROWS
        effective_cols = cols if cols is not None else config.COLS

        if not _is_power_of_two(effective_rows * effective_cols):
            return False, f"Grid size {effective_rows}×{effective_cols}={effective_rows * effective_cols} must be a power of 2 (e.g. 4×4, 4×8, 8×8)"

        self._video_source = video_source if video_source is not None else config.VIDEO_SOURCE
        self._rows = effective_rows
        self._cols = effective_cols
        self._direction_split = config.DIRECTION_SPLIT if direction_split == "UNSET" else direction_split

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
        grafana_queue: Optional[queue.Queue] = None
        grafana_thread: Optional[threading.Thread] = None
        dropped_grafana_pushes = 0

        try:
            source_path = _resolve_video_source(self._video_source)

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

            n_regions = self._rows * self._cols
            target_frame_duration = 1.0 / max(config.TARGET_FPS, 1)

            last_quantum_density = None
            last_quantum_count = None
            last_quantum_metrics = None
            frames_since_quantum = config.QUANTUM_EVERY_N

            if config.USE_QUANTUM:
                quantum_executor = ThreadPoolExecutor(max_workers=1)

            if config.GRAFANA_PUSH:
                grafana_queue = queue.Queue(maxsize=GRAFANA_QUEUE_MAXSIZE)
                grafana_thread = threading.Thread(
                    target=self._grafana_worker,
                    args=(grafana_queue,),
                    name="grafana-push-worker",
                    daemon=True,
                )
                grafana_thread.start()

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
                timestamp_ms = (
                    capture.get(cv2.CAP_PROP_POS_MSEC)
                    if capture is not None
                    else float(frame_number)
                )

                occupancy = boxes_to_occupancy(
                    result.boxes_xyxy,
                    self._rows,
                    self._cols,
                    frame_w,
                    frame_h,
                )
                classical_count = compute_classical_count(occupancy)
                classical_density = classical_count / n_regions

                direction_data = None
                if self._direction_split:
                    direction_data = directional_occupancy(
                        result.boxes_xyxy,
                        self._rows,
                        self._cols,
                        frame_w,
                        frame_h,
                        split=self._direction_split,
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
                            (
                                last_quantum_count,
                                last_quantum_density,
                                last_quantum_metrics,
                            ) = quantum_future.result()
                        except Exception as exc:
                            log.warning("Quantum step failed: %s", exc)
                            last_quantum_count, last_quantum_density = None, None
                            last_quantum_metrics = None
                        finally:
                            quantum_future = None

                labels = [d.class_name for d in result.detections]
                confidences = [d.confidence for d in result.detections]

                vis_frame = create_visualization(
                    frame=result.frame,
                    boxes=result.boxes_xyxy,
                    occupancy=occupancy,
                    rows=self._rows,
                    cols=self._cols,
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

                if config.GRAFANA_PUSH:
                    push_every_n = max(int(config.GRAFANA_PUSH_EVERY_N), 1)
                    if frame_number % push_every_n == 0:
                        density_difference = (
                            (last_quantum_density - classical_density)
                            if last_quantum_density is not None
                            else None
                        )
                        count_agreement = (
                            (last_quantum_count == classical_count)
                            if last_quantum_count is not None
                            else None
                        )
                        error = (
                            abs(last_quantum_count - classical_count)
                            if last_quantum_count is not None
                            else None
                        )
                        relative_error_pct = (
                            (error / classical_count * 100)
                            if error is not None and classical_count != 0
                            else None
                        )

                        payload = {
                            "classical_count": classical_count,
                            "quantum_count": last_quantum_count,
                            "classical_density": classical_density,
                            "quantum_density": last_quantum_density,
                            "error": error,
                            "relative_error_pct": relative_error_pct,
                            "density_A": direction_data["density_A"] if direction_data else None,
                            "density_B": direction_data["density_B"] if direction_data else None,
                            "num_detections": len(result.detections),
                            "count_agreement": count_agreement,
                            "theoretical_speedup": (n_regions ** 0.5) if config.USE_QUANTUM else None,
                            "classical_count_time_ns": (
                                last_quantum_metrics.classical_count_time_ns if last_quantum_metrics else None
                            ),
                            "circuit_build_time_ms": (
                                last_quantum_metrics.circuit_build_time_ms if last_quantum_metrics else None
                            ),
                            "transpile_time_ms": (
                                last_quantum_metrics.transpile_time_ms if last_quantum_metrics else None
                            ),
                            "simulation_run_time_ms": (
                                last_quantum_metrics.simulation_run_time_ms if last_quantum_metrics else None
                            ),
                            "estimated_qpu_time_ns": (
                                last_quantum_metrics.estimated_qpu_time_ns if last_quantum_metrics else None
                            ),
                            "simulation_overhead_ms": (
                                last_quantum_metrics.simulation_overhead_ms if last_quantum_metrics else None
                            ),
                            "circuit_depth": (
                                last_quantum_metrics.circuit_depth if last_quantum_metrics else None
                            ),
                            "estimated_speedup_vs_classical": (
                                last_quantum_metrics.estimated_speedup_vs_classical if last_quantum_metrics else None
                            ),
                        }

                        if grafana_queue is not None:
                            try:
                                grafana_queue.put_nowait(payload)
                            except queue.Full:
                                dropped_grafana_pushes += 1
                                if dropped_grafana_pushes % 50 == 1:
                                    log.warning(
                                        "Grafana queue full; dropping metrics (%d dropped)",
                                        dropped_grafana_pushes,
                                    )

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

            if grafana_queue is not None:
                try:
                    grafana_queue.put_nowait(None)
                except queue.Full:
                    pass
            if grafana_thread is not None:
                grafana_thread.join(timeout=2)

            if capture is not None:
                capture.release()
            with self._state_lock:
                self._running = False

    def _grafana_worker(self, grafana_queue: queue.Queue) -> None:
        while True:
            try:
                payload = grafana_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if payload is None:
                break

            try:
                push_metrics_to_grafana(payload)
            except Exception as exc:
                log.warning("Grafana worker push failed: %s", exc)
            finally:
                grafana_queue.task_done()


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
