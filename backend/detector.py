import threading
import time
from base64 import b64encode
from pathlib import Path
import subprocess
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO


class BottleDetectorService:
    def __init__(
        self,
        camera_source: int | str = 0,
        confidence: float = 0.75,
        iou: float = 0.4,
        image_size: int = 650,
        augment: bool = False,
    ) -> None:
        self.model_path = Path(__file__).with_name("best4.pt")
        self.camera_source = self._normalize_camera_source(camera_source)
        self.confidence = confidence
        self.image_size = image_size
        self.iou = iou
        self.augment = augment
        self.model = YOLO(str(self.model_path))

        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._analysis_active = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None

        self._latest_frame = self._build_status_frame("Starting detector...")
        self._live_count = 0
        self._total_count = 0
        self._camera_active = False
        self._status_message = "Model loaded. Waiting for camera..."
        self._camera_retry_delay_seconds = 5.0
        self._video_playback_slowdown_factor = 1.75
        self._current_source_frame_delay_seconds = 0.0

        self._analysis_sample_interval_seconds = 0.12
        self._analysis_max_processed_frames = 180
        self._analysis_image_size = 512
        self._analysis_tracking_confidence = max(0.5, confidence - 0.12)
        self._analysis_tracking_iou_threshold = 0.2
        self._analysis_tracking_center_distance_ratio = 0.65
        self._analysis_tracking_max_missed_frames = 18
        self._analysis_min_track_hits = 2
        self._analysis_immediate_confirm_confidence = min(max(confidence + 0.18, 0.82), 0.95)

        self._min_box_width_pixels = 10
        self._min_box_height_pixels = 16
        self._min_box_area_ratio = 0.0005
        self._max_box_area_ratio = 0.65
        self._max_box_aspect_ratio = 5.5

        self._live_active_tracks: dict[int, dict[str, object]] = {}
        self._live_next_track_id = 1
        self._live_frame_index = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._reset_live_tracker()

    def get_count(self) -> int:
        with self._lock:
            return self._live_count

    def get_total(self) -> int:
        with self._lock:
            return self._total_count

    def get_status(self) -> dict[str, object]:
        with self._lock:
            return {
                "camera_active": self._camera_active,
                "message": self._status_message,
                "count": self._live_count,
                "total": self._total_count,
                "camera_source": str(self.camera_source),
                "model_path": str(self.model_path),
            }

    def analyze_video_file(
        self,
        video_path: str | Path,
        *,
        frame_stride: int | None = None,
        annotated_output_path: str | Path | None = None,
        playback_slowdown_factor: float = 1.0,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        realtime_playback: bool = False,
    ) -> dict[str, object]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("Unable to open the uploaded video")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        selected_frame_stride = 1 if realtime_playback else self._resolve_analysis_stride(
            frame_stride,
            total_frames=total_frames,
            fps=fps,
        )

        processed_frames = 0
        sampled_frames = 0
        frame_index = 0
        total_bottle_count = 0
        highest_frame_bottle_count = 0
        final_frame_bottle_count = 0
        peak_frame_index: int | None = None
        peak_frame_base64: str | None = None
        peak_boxes: list[dict[str, object]] = []
        active_tracks: dict[int, dict[str, object]] = {}
        next_track_id = 1
        annotated_temp_path: Path | None = None
        video_writer: cv2.VideoWriter | None = None

        self._analysis_active.set()
        try:
            with self._lock:
                self._status_message = "Upload analysis in progress..."

            if annotated_output_path:
                annotated_output = Path(annotated_output_path)
                annotated_output.parent.mkdir(parents=True, exist_ok=True)
                annotated_temp_path = annotated_output.with_name(f"{annotated_output.stem}.raw.mp4")
                output_fps = self._resolve_output_fps(fps, playback_slowdown_factor)
                video_writer = cv2.VideoWriter(
                    str(annotated_temp_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (frame_width, frame_height),
                )
                if not video_writer.isOpened():
                    raise RuntimeError("Unable to create the detected video output")

            self._reset_video_tracker()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                bottle_boxes = self._detect_tracked_bottle_boxes(
                    frame,
                    image_size=self._analysis_image_size,
                )
                if self._has_track_ids(bottle_boxes):
                    tracked_boxes, new_bottles_count, active_tracks = self._update_confirmed_tracked_boxes(
                        bottle_boxes,
                        active_tracks=active_tracks,
                        frame_index=frame_index,
                    )
                    current_bottle_count = len(tracked_boxes)
                else:
                    tracked_boxes, new_bottles_count, active_tracks, next_track_id = self._track_video_bottles(
                        bottle_boxes,
                        active_tracks=active_tracks,
                        next_track_id=next_track_id,
                        frame_index=frame_index,
                    )
                    current_bottle_count = len(tracked_boxes)

                total_bottle_count += new_bottles_count
                final_frame_bottle_count = current_bottle_count
                sampled_frames += 1

                if current_bottle_count > highest_frame_bottle_count:
                    highest_frame_bottle_count = current_bottle_count
                    peak_frame_index = frame_index
                    peak_boxes = [dict(box) for box in tracked_boxes]

                annotated_frame = self._annotate_frame(
                    frame.copy(),
                    tracked_boxes,
                    current_count=current_bottle_count,
                    total_count=total_bottle_count,
                )
                processed_frames += 1
                encoded_frame = self._encode_frame(annotated_frame)

                if progress_callback is not None:
                    progress_callback(
                        {
                            "frame_index": frame_index,
                            "processed_frames": processed_frames,
                            "sampled_frames": sampled_frames,
                            "total_frames": total_frames,
                            "frame_stride": selected_frame_stride,
                            "current_bottle_count": current_bottle_count,
                            "total_bottle_count": total_bottle_count,
                            "new_bottles_count": new_bottles_count,
                            "highest_frame_bottle_count": highest_frame_bottle_count,
                            "peak_frame_index": peak_frame_index,
                            "peak_time_seconds": (
                                round(peak_frame_index / fps, 2)
                                if fps > 0 and peak_frame_index is not None
                                else None
                            ),
                            "duration_seconds": (
                                round(total_frames / self._resolve_output_fps(fps, playback_slowdown_factor), 2)
                                if total_frames > 0
                                else None
                            ),
                            "frame_jpeg": encoded_frame,
                        }
                    )

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                if peak_frame_index == frame_index:
                    peak_frame_base64 = b64encode(encoded_frame).decode("ascii")

                frame_index += 1
        finally:
            capture.release()
            if video_writer is not None:
                video_writer.release()
            self._reset_video_tracker()
            self._analysis_active.clear()

        if annotated_output_path and annotated_temp_path is not None:
            self._finalize_annotated_video(
                annotated_temp_path=annotated_temp_path,
                annotated_output_path=Path(annotated_output_path),
            )

        if processed_frames == 0:
            raise RuntimeError("The uploaded video could not be decoded into readable frames")

        return {
            "total_bottle_count": total_bottle_count,
            "highest_frame_bottle_count": highest_frame_bottle_count,
            "final_frame_bottle_count": final_frame_bottle_count,
            "processed_frames": processed_frames,
            "sampled_frames": sampled_frames,
            "total_frames": total_frames,
            "frame_stride": selected_frame_stride,
            "peak_frame_index": peak_frame_index,
            "peak_time_seconds": (
                round(peak_frame_index / fps, 2)
                if fps > 0 and peak_frame_index is not None
                else None
            ),
            "duration_seconds": (
                round(total_frames / self._resolve_output_fps(fps, playback_slowdown_factor), 2)
                if total_frames > 0
                else None
            ),
            "peak_boxes": peak_boxes,
            "peak_frame_jpeg_base64": peak_frame_base64,
        }

    def frames(self):
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._latest_frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_started_at = time.perf_counter()
            if self._analysis_active.is_set():
                time.sleep(0.1)
                continue

            if self._capture is None or not self._capture.isOpened():
                self._capture = cv2.VideoCapture(self.camera_source)
                if not self._capture.isOpened():
                    if self._capture is not None:
                        self._capture.release()
                        self._capture = None
                    self._set_state(
                        camera_active=False,
                        message=f"No camera detected on source {self.camera_source}. Upload mode is still available.",
                        frame=self._build_status_frame("Camera unavailable"),
                    )
                    self._reset_live_tracker()
                    time.sleep(self._camera_retry_delay_seconds)
                    continue

                self._current_source_frame_delay_seconds = self._resolve_source_frame_delay()
                self._set_state(
                    camera_active=True,
                    message=f"Active on source {self.camera_source}",
                )

            success, frame = self._capture.read()
            if not success:
                if isinstance(self.camera_source, str):
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                self._set_state(
                    camera_active=False,
                    message="Frame read failed. Retrying...",
                    frame=self._build_status_frame("Waiting for frame..."),
                )
                self._reset_live_tracker()
                self._capture.release()
                self._capture = None
                time.sleep(0.5)
                continue

            try:
                annotated_frame, live_count, _ = self._detect_bottles(frame)
                encoded = self._encode_frame(annotated_frame)

                with self._lock:
                    self._latest_frame = encoded
                    self._live_count = live_count
                    self._total_count = live_count
                    self._camera_active = True
                    self._status_message = f"Detecting bottles on source {self.camera_source}"

                if self._current_source_frame_delay_seconds > 0:
                    self._sleep_for_remaining_frame_delay(
                        loop_started_at,
                        self._current_source_frame_delay_seconds,
                    )
            except Exception as exc:
                self._set_state(
                    camera_active=False,
                    message=f"Detection error: {exc}",
                    frame=self._build_status_frame("Detection error"),
                )
                self._reset_live_tracker()
                time.sleep(0.5)

    def _detect_bottles(
        self,
        frame,
        *,
        image_size: int | None = None,
    ) -> tuple[object, int, list[dict[str, object]]]:
        bottle_boxes = self._detect_bottle_boxes(frame, image_size=image_size)
        tracked_boxes, _, self._live_active_tracks, self._live_next_track_id = self._track_video_bottles(
            bottle_boxes,
            active_tracks=self._live_active_tracks,
            next_track_id=self._live_next_track_id,
            frame_index=self._live_frame_index,
        )
        self._live_frame_index += 1
        annotated_frame = self._annotate_frame(frame, tracked_boxes)
        return annotated_frame, len(tracked_boxes), tracked_boxes

    def _detect_bottle_boxes(
        self,
        frame,
        *,
        image_size: int | None = None,
    ) -> list[dict[str, object]]:
        results = self._predict(frame, imgsz=image_size)
        names = results.names
        bottle_boxes: list[dict[str, object]] = []

        for box in results.boxes:
            cls = int(box.cls[0])
            class_name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            if "bottle" not in str(class_name).strip().lower():
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bottle_boxes.append(
                {
                    "label": str(class_name),
                    "confidence": round(float(box.conf[0]), 3),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        filtered_boxes = self._filter_bottle_boxes(
            bottle_boxes,
            frame_shape=frame.shape,
            min_confidence=self.confidence,
        )
        return self._merge_overlapping_boxes(filtered_boxes)

    def _detect_tracked_bottle_boxes(
        self,
        frame,
        *,
        image_size: int | None = None,
    ) -> list[dict[str, object]]:
        results = self._track_predict(frame, imgsz=image_size)
        names = results.names
        track_ids: list[int] | None = None

        if results.boxes.id is not None:
            track_ids = results.boxes.id.int().tolist()

        bottle_boxes: list[dict[str, object]] = []
        for index, box in enumerate(results.boxes):
            cls = int(box.cls[0])
            class_name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
            if "bottle" not in str(class_name).strip().lower():
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bottle_box = {
                "label": str(class_name),
                "confidence": round(float(box.conf[0]), 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            if track_ids is not None and index < len(track_ids):
                bottle_box["track_id"] = int(track_ids[index])

            bottle_boxes.append(bottle_box)

        filtered_boxes = self._filter_bottle_boxes(
            bottle_boxes,
            frame_shape=frame.shape,
            min_confidence=self._analysis_tracking_confidence,
        )

        if self._has_track_ids(filtered_boxes):
            return filtered_boxes

        return self._merge_overlapping_boxes(filtered_boxes)

    @staticmethod
    def _has_track_ids(bottle_boxes: list[dict[str, object]]) -> bool:
        return any(box.get("track_id") is not None for box in bottle_boxes)

    def _filter_bottle_boxes(
        self,
        bottle_boxes: list[dict[str, object]],
        *,
        frame_shape: tuple[int, ...],
        min_confidence: float,
    ) -> list[dict[str, object]]:
        if not bottle_boxes:
            return []

        frame_height = max(int(frame_shape[0]), 1)
        frame_width = max(int(frame_shape[1]), 1)
        frame_area = float(frame_width * frame_height)
        filtered_boxes: list[dict[str, object]] = []

        for bottle_box in bottle_boxes:
            confidence = float(bottle_box.get("confidence", 0.0))
            if confidence < min_confidence:
                continue

            width, height = self._box_dimensions(bottle_box)
            if width < self._min_box_width_pixels or height < self._min_box_height_pixels:
                continue

            area_ratio = (width * height) / frame_area
            if area_ratio < self._min_box_area_ratio or area_ratio > self._max_box_area_ratio:
                continue

            aspect_ratio = max(width / max(height, 1.0), height / max(width, 1.0))
            if aspect_ratio > self._max_box_aspect_ratio:
                continue

            filtered_boxes.append(dict(bottle_box))

        return filtered_boxes

    def _update_confirmed_tracked_boxes(
        self,
        bottle_boxes: list[dict[str, object]],
        *,
        active_tracks: dict[int, dict[str, object]],
        frame_index: int,
    ) -> tuple[list[dict[str, object]], int, dict[int, dict[str, object]]]:
        self._prune_inactive_tracks(active_tracks, frame_index=frame_index)

        tracked_boxes: list[dict[str, object]] = []
        new_bottles_count = 0

        for bottle_box in bottle_boxes:
            track_id = bottle_box.get("track_id")
            if track_id is None:
                continue

            normalized_track_id = int(track_id)
            previous_track_state = active_tracks.get(normalized_track_id, {})
            hits = int(previous_track_state.get("hits", 0)) + 1
            was_confirmed = bool(previous_track_state.get("confirmed", False))
            is_confirmed = was_confirmed or self._should_confirm_track(bottle_box, hits)
            if is_confirmed and not was_confirmed:
                new_bottles_count += 1

            active_tracks[normalized_track_id] = {
                "box": dict(bottle_box),
                "last_seen_frame": frame_index,
                "hits": hits,
                "confirmed": is_confirmed,
            }

            if is_confirmed:
                tracked_box = dict(bottle_box)
                tracked_box["track_id"] = normalized_track_id
                tracked_boxes.append(tracked_box)

        return tracked_boxes, new_bottles_count, active_tracks

    def _track_video_bottles(
        self,
        bottle_boxes: list[dict[str, object]],
        *,
        active_tracks: dict[int, dict[str, object]],
        next_track_id: int,
        frame_index: int,
    ) -> tuple[list[dict[str, object]], int, dict[int, dict[str, object]], int]:
        self._prune_inactive_tracks(active_tracks, frame_index=frame_index)

        matched_track_ids: set[int] = set()
        tracked_boxes: list[dict[str, object]] = []
        new_bottles_count = 0

        for bottle_box in bottle_boxes:
            track_id = self._match_existing_track(
                bottle_box,
                active_tracks=active_tracks,
                matched_track_ids=matched_track_ids,
                frame_index=frame_index,
            )

            if track_id is None:
                track_id = next_track_id
                next_track_id += 1

            matched_track_ids.add(track_id)
            previous_track_state = active_tracks.get(track_id, {})
            hits = int(previous_track_state.get("hits", 0)) + 1
            was_confirmed = bool(previous_track_state.get("confirmed", False))
            is_confirmed = was_confirmed or self._should_confirm_track(bottle_box, hits)
            if is_confirmed and not was_confirmed:
                new_bottles_count += 1

            active_tracks[track_id] = {
                "box": dict(bottle_box),
                "last_seen_frame": frame_index,
                "hits": hits,
                "confirmed": is_confirmed,
            }

            if is_confirmed:
                tracked_box = dict(bottle_box)
                tracked_box["track_id"] = track_id
                tracked_boxes.append(tracked_box)

        return tracked_boxes, new_bottles_count, active_tracks, next_track_id

    def _match_existing_track(
        self,
        bottle_box: dict[str, object],
        *,
        active_tracks: dict[int, dict[str, object]],
        matched_track_ids: set[int],
        frame_index: int,
    ) -> int | None:
        best_track_id: int | None = None
        best_score = -1.0

        for track_id, track_state in active_tracks.items():
            if track_id in matched_track_ids:
                continue

            last_seen_frame = int(track_state.get("last_seen_frame", -1))
            if frame_index - last_seen_frame > self._analysis_tracking_max_missed_frames:
                continue

            previous_box = track_state.get("box")
            if not isinstance(previous_box, dict):
                continue

            score = self._box_tracking_score(bottle_box, previous_box)
            if score > best_score:
                best_score = score
                best_track_id = track_id

        if best_score < 0:
            return None

        return best_track_id

    def _box_tracking_score(
        self,
        bottle_box: dict[str, object],
        previous_box: dict[str, object],
    ) -> float:
        iou_score = self._box_iou(bottle_box, previous_box)
        if iou_score >= self._analysis_tracking_iou_threshold:
            return 2.0 + iou_score

        center_distance_ratio = self._box_center_distance_ratio(bottle_box, previous_box)
        if center_distance_ratio <= self._analysis_tracking_center_distance_ratio:
            return 1.0 - center_distance_ratio

        return -1.0

    def _box_center_distance_ratio(
        self,
        bottle_box: dict[str, object],
        previous_box: dict[str, object],
    ) -> float:
        bottle_center_x, bottle_center_y = self._box_center(bottle_box)
        previous_center_x, previous_center_y = self._box_center(previous_box)
        center_distance = float(
            np.hypot(bottle_center_x - previous_center_x, bottle_center_y - previous_center_y)
        )

        bottle_width, bottle_height = self._box_dimensions(bottle_box)
        previous_width, previous_height = self._box_dimensions(previous_box)
        scale = max(
            bottle_width,
            bottle_height,
            previous_width,
            previous_height,
            1.0,
        )
        return center_distance / scale

    @staticmethod
    def _box_center(bottle_box: dict[str, object]) -> tuple[float, float]:
        return (
            (int(bottle_box["x1"]) + int(bottle_box["x2"])) / 2.0,
            (int(bottle_box["y1"]) + int(bottle_box["y2"])) / 2.0,
        )

    @staticmethod
    def _box_dimensions(bottle_box: dict[str, object]) -> tuple[float, float]:
        return (
            float(max(1, int(bottle_box["x2"]) - int(bottle_box["x1"]))),
            float(max(1, int(bottle_box["y2"]) - int(bottle_box["y1"]))),
        )

    def _should_confirm_track(self, bottle_box: dict[str, object], hits: int) -> bool:
        confidence = float(bottle_box.get("confidence", 0.0))
        if confidence >= self._analysis_immediate_confirm_confidence:
            return True
        return hits >= self._analysis_min_track_hits

    def _prune_inactive_tracks(
        self,
        active_tracks: dict[int, dict[str, object]],
        *,
        frame_index: int,
    ) -> None:
        stale_track_ids = [
            track_id
            for track_id, track_state in active_tracks.items()
            if frame_index - int(track_state.get("last_seen_frame", -1))
            > self._analysis_tracking_max_missed_frames
        ]
        for stale_track_id in stale_track_ids:
            active_tracks.pop(stale_track_id, None)

    def _merge_overlapping_boxes(
        self,
        bottle_boxes: list[dict[str, object]],
        *,
        iou_threshold: float = 0.45,
    ) -> list[dict[str, object]]:
        if not bottle_boxes:
            return []

        sorted_boxes = sorted(
            bottle_boxes,
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        kept_boxes: list[dict[str, object]] = []

        for candidate in sorted_boxes:
            if all(self._box_iou(candidate, kept) < iou_threshold for kept in kept_boxes):
                kept_boxes.append(candidate)

        return kept_boxes

    @staticmethod
    def _box_iou(box_a: dict[str, object], box_b: dict[str, object]) -> float:
        ax1 = int(box_a["x1"])
        ay1 = int(box_a["y1"])
        ax2 = int(box_a["x2"])
        ay2 = int(box_a["y2"])
        bx1 = int(box_b["x1"])
        by1 = int(box_b["y1"])
        bx2 = int(box_b["x2"])
        by2 = int(box_b["y2"])

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height
        if intersection == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0

        return intersection / union

    def _draw_bottle_boxes(self, frame, bottle_boxes: list[dict[str, object]]) -> None:
        for bottle_box in bottle_boxes:
            x1 = int(bottle_box["x1"])
            y1 = int(bottle_box["y1"])
            x2 = int(bottle_box["x2"])
            y2 = int(bottle_box["y2"])
            label = f'{bottle_box["label"]} {bottle_box["confidence"]:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    def _annotate_frame(
        self,
        frame,
        bottle_boxes: list[dict[str, object]],
        *,
        current_count: int | None = None,
        total_count: int | None = None,
    ):
        self._draw_bottle_boxes(frame, bottle_boxes)
        display_count = len(bottle_boxes) if current_count is None else current_count
        overlay_lines = [f"Total Count: {total_count}"] if total_count is not None else [f"Count: {display_count}"]

        top = 12
        left = 12
        line_height = 34
        overlay_height = 18 + (len(overlay_lines) * line_height)
        max_text_width = max((len(text) for text in overlay_lines), default=0)
        overlay_width = max(240, min(420, 28 + (max_text_width * 14)))

        cv2.rectangle(
            frame,
            (left, top),
            (left + overlay_width, top + overlay_height),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (left, top),
            (left + overlay_width, top + overlay_height),
            (0, 255, 255),
            2,
        )

        for line_index, text in enumerate(overlay_lines):
            cv2.putText(
                frame,
                text,
                (left + 12, top + 28 + (line_index * line_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        return frame

    def _predict(self, frame, *, imgsz: int | None = None):
        with self._model_lock:
            return self.model.predict(
                frame,
                imgsz=self.image_size if imgsz is None else imgsz,
                conf=self.confidence,
                iou=self.iou,
                augment=self.augment,
                verbose=False,
            )[0]

    def _track_predict(self, frame, *, imgsz: int | None = None):
        with self._model_lock:
            return self.model.track(
                frame,
                persist=True,
                imgsz=self.image_size if imgsz is None else imgsz,
                conf=self._analysis_tracking_confidence,
                iou=self.iou,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

    def _reset_live_tracker(self) -> None:
        with self._lock:
            self._live_active_tracks = {}
            self._live_next_track_id = 1
            self._live_frame_index = 0
            self._live_count = 0
            self._total_count = 0

    def _reset_video_tracker(self) -> None:
        with self._model_lock:
            predictor = self.model.predictor
            if predictor is None or not hasattr(predictor, "trackers"):
                return

            for tracker in predictor.trackers:
                tracker.reset()

            predictor.vid_path = [None] * len(predictor.trackers)

    def _build_status_frame(self, message: str) -> bytes:
        frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Bottle Detector",
            (140, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            message,
            (80, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 80, 80),
            2,
        )
        return self._encode_frame(frame)

    def build_status_frame(self, message: str) -> bytes:
        return self._build_status_frame(message)

    def _encode_frame(self, frame) -> bytes:
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to encode frame")
        return buffer.tobytes()

    def _set_state(
        self,
        *,
        camera_active: bool,
        message: str,
        frame: bytes | None = None,
    ) -> None:
        with self._lock:
            self._camera_active = camera_active
            self._status_message = message
            self._live_count = 0 if not camera_active else self._live_count
            if frame is not None:
                self._latest_frame = frame

    def _resolve_analysis_stride(
        self,
        frame_stride: int | None,
        *,
        total_frames: int,
        fps: float,
    ) -> int:
        if frame_stride is not None and frame_stride > 0:
            return frame_stride

        stride_from_interval = 1
        if fps > 0:
            stride_from_interval = max(int(round(fps * self._analysis_sample_interval_seconds)), 1)

        stride_from_frame_budget = 1
        if total_frames > 0:
            stride_from_frame_budget = max(
                int(np.ceil(total_frames / self._analysis_max_processed_frames)),
                1,
            )

        return max(stride_from_interval, stride_from_frame_budget)

    @staticmethod
    def _resolve_output_fps(fps: float, playback_slowdown_factor: float) -> float:
        if fps <= 0:
            return 8.0
        return max(fps / max(playback_slowdown_factor, 1.0), 4.0)

    @classmethod
    def _resolve_analysis_frame_delay(
        cls,
        fps: float,
        playback_slowdown_factor: float,
    ) -> float:
        return 1 / cls._resolve_output_fps(fps, playback_slowdown_factor)

    @staticmethod
    def _sleep_for_remaining_frame_delay(
        frame_started_at: float,
        target_frame_delay_seconds: float,
    ) -> None:
        remaining_delay = target_frame_delay_seconds - (time.perf_counter() - frame_started_at)
        if remaining_delay > 0:
            time.sleep(remaining_delay)

    @staticmethod
    def _finalize_annotated_video(
        *,
        annotated_temp_path: Path,
        annotated_output_path: Path,
    ) -> None:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(annotated_temp_path),
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-movflags",
                "+faststart",
                str(annotated_output_path),
            ],
            capture_output=True,
            text=True,
        )
        if annotated_temp_path.exists():
            annotated_temp_path.unlink()
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Unable to build the detected video output")

    def _resolve_source_frame_delay(self) -> float:
        if not self._is_video_file_source():
            return 0.0

        fps = 0.0
        if self._capture is not None:
            fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0)

        if fps <= 0:
            return 0.12

        return max((1 / fps) * self._video_playback_slowdown_factor, 0.08)

    def _is_video_file_source(self) -> bool:
        return isinstance(self.camera_source, str) and Path(self.camera_source).exists()

    @staticmethod
    def _normalize_camera_source(camera_source: int | str) -> int | str:
        if isinstance(camera_source, str) and camera_source.isdigit():
            return int(camera_source)
        return camera_source
