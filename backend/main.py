import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from .detector import BottleDetectorService
except ImportError:
    from detector import BottleDetectorService


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
GENERATED_VIDEOS_DIR = BASE_DIR / "generated_videos"
ASSETS_DIR = BASE_DIR / "assets"          # ← assets folder
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/generated_videos",
    StaticFiles(directory=str(GENERATED_VIDEOS_DIR)),
    name="generated_videos",
)

live_detector: BottleDetectorService | None = None
analysis_detector: BottleDetectorService | None = None
analysis_jobs: dict[str, dict[str, Any]] = {}
analysis_jobs_lock = threading.Lock()
analysis_run_lock = threading.Lock()


def _camera_source() -> int | str:
    configured_source = os.getenv("CAMERA_SOURCE", "0").strip() or "0"
    return int(configured_source) if configured_source.isdigit() else configured_source


def _get_live_detector() -> BottleDetectorService:
    global live_detector
    if live_detector is None:
        live_detector = BottleDetectorService(camera_source=_camera_source())
        live_detector.start()
    return live_detector


def _get_analysis_detector() -> BottleDetectorService:
    global analysis_detector
    if analysis_detector is None:
        analysis_detector = BottleDetectorService(camera_source=_camera_source())
    return analysis_detector


def _sanitize_filename(filename: str) -> tuple[str, str]:
    source_name = Path(filename or "video.mp4").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).stem).strip("._")
    suffix = Path(source_name).suffix if Path(source_name).suffix else ".mp4"
    return stem or "video", suffix


def _serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if not key.startswith("_") and key not in {"source_path", "output_path"}
    }


def _get_job(job_id: str) -> dict[str, Any]:
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        return dict(job)


def _update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        job.update(updates)
        return dict(job)


def _build_stream_chunk(frame: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


def _analysis_stream(job_id: str):
    last_frame: bytes | None = None
    while True:
        job = _get_job(job_id)
        frame = job.get("_latest_frame")
        state = job.get("state")
        if isinstance(frame, bytes):
            last_frame = frame
            yield _build_stream_chunk(frame)
        elif last_frame is not None:
            yield _build_stream_chunk(last_frame)
        if state in {"completed", "failed"}:
            break
        time.sleep(0.08)


def _progress_message(progress_percent: float | None, total_bottle_count: int) -> str:
    if progress_percent is None:
        return f"Uploaded video detection is running. Total bottles counted: {total_bottle_count}."
    return (
        f"Uploaded video detection is running. "
        f"{progress_percent:.1f}% complete, total bottles counted: {total_bottle_count}."
    )


def _run_analysis_job(job_id: str, source_path: Path, output_path: Path) -> None:
    detector = _get_analysis_detector()
    try:
        _update_job(job_id, state="preparing", message="Detector is preparing the uploaded video...")

        def progress_callback(update: dict[str, object]) -> None:
            total_frames = int(update.get("total_frames") or 0)
            processed_frames = int(update.get("processed_frames") or 0)
            total_bottle_count = int(update.get("total_bottle_count") or 0)
            progress_percent = (
                round(min((processed_frames / total_frames) * 100, 100.0), 1)
                if total_frames > 0
                else None
            )
            _update_job(
                job_id,
                state="processing",
                message=_progress_message(progress_percent, total_bottle_count),
                processed_frames=processed_frames,
                sampled_frames=int(update.get("sampled_frames") or 0),
                total_frames=total_frames,
                frame_stride=int(update.get("frame_stride") or 1),
                current_bottle_count=int(update.get("current_bottle_count") or 0),
                total_bottle_count=total_bottle_count,
                highest_frame_bottle_count=int(update.get("highest_frame_bottle_count") or 0),
                peak_frame_index=update.get("peak_frame_index"),
                peak_time_seconds=update.get("peak_time_seconds"),
                duration_seconds=update.get("duration_seconds"),
                progress_percent=progress_percent,
                _latest_frame=update.get("frame_jpeg"),
            )

        result = detector.analyze_video_file(
            source_path,
            annotated_output_path=output_path,
            progress_callback=progress_callback,
        )
        _update_job(
            job_id,
            state="completed",
            message="Uploaded video analysis completed successfully.",
            error=None,
            progress_percent=100.0,
            current_bottle_count=result["final_frame_bottle_count"],
            annotated_video_url=f"/generated_videos/{output_path.name}",
            **result,
        )
    except Exception as exc:
        failure_frame = _get_live_detector().build_status_frame("Analysis failed")
        _update_job(
            job_id,
            state="failed",
            message="Uploaded video analysis failed.",
            error=str(exc),
            _latest_frame=failure_frame,
        )
    finally:
        analysis_run_lock.release()


@app.on_event("startup")
def startup() -> None:
    _get_live_detector()


@app.get("/")
def home():
    return {"message": "Bottle Counter Running"}


@app.get("/status")
def status():
    return _get_live_detector().get_status()


# ── assets folder-ல் உள்ள videos list ──────────────────────────
@app.get("/videos")
def list_videos():
    videos = [
        {"name": f.name}
        for f in sorted(ASSETS_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return {"videos": videos}


@app.post("/reset_analysis")
def reset_analysis():
    try:
        analysis_run_lock.release()
    except RuntimeError:
        pass
    return {"message": "Analysis lock reset"}


# ── assets folder video-ஐ analyze பண்ண ──────────────────────────
@app.post("/analyze_assets_video", status_code=202)
async def analyze_assets_video(payload: dict):
    video_name = (payload.get("video") or "").strip()
    if not video_name:
        raise HTTPException(status_code=400, detail="video name is required.")

    source_path = ASSETS_DIR / video_name
    if not source_path.exists() or source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Video not found in assets folder.")

    if not analysis_run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another video analysis is already running.")

    job_id = uuid.uuid4().hex
    stem, suffix = _sanitize_filename(video_name)
    output_path = GENERATED_VIDEOS_DIR / f"{stem}_{job_id}.mp4"

    try:
        initial_frame = _get_live_detector().build_status_frame("Preparing upload...")
        initial_job = {
            "job_id": job_id,
            "filename": video_name,
            "state": "queued",
            "message": "Upload received. Preparing live detection...",
            "error": None,
            "processed_frames": 0,
            "sampled_frames": 0,
            "total_frames": 0,
            "frame_stride": 1,
            "current_bottle_count": 0,
            "total_bottle_count": 0,
            "highest_frame_bottle_count": 0,
            "peak_frame_index": None,
            "peak_time_seconds": None,
            "duration_seconds": None,
            "progress_percent": 0.0,
            "annotated_video_url": None,
            "stream_url": f"/analysis_jobs/{job_id}/stream",
            "source_path": str(source_path),
            "output_path": str(output_path),
            "_latest_frame": initial_frame,
        }
        with analysis_jobs_lock:
            analysis_jobs[job_id] = initial_job

        worker = threading.Thread(
            target=_run_analysis_job,
            args=(job_id, source_path, output_path),
            daemon=True,
        )
        worker.start()
        return _serialize_job(initial_job)
    except Exception:
        analysis_run_lock.release()
        raise


@app.post("/analyze_video", status_code=202)
@app.post("/upload_video", status_code=202)
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")

    if not analysis_run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another video analysis is already running.")

    job_id = uuid.uuid4().hex
    stem, suffix = _sanitize_filename(file.filename)
    source_path = UPLOADS_DIR / f"{stem}_{job_id}{suffix}"
    output_path = GENERATED_VIDEOS_DIR / f"{stem}_{job_id}.mp4"

    try:
        with source_path.open("wb") as upload_file:
            shutil.copyfileobj(file.file, upload_file)

        initial_frame = _get_live_detector().build_status_frame("Preparing upload...")
        initial_job = {
            "job_id": job_id,
            "filename": file.filename,
            "state": "queued",
            "message": "Upload received. Preparing live detection...",
            "error": None,
            "processed_frames": 0,
            "sampled_frames": 0,
            "total_frames": 0,
            "frame_stride": 1,
            "current_bottle_count": 0,
            "total_bottle_count": 0,
            "highest_frame_bottle_count": 0,
            "peak_frame_index": None,
            "peak_time_seconds": None,
            "duration_seconds": None,
            "progress_percent": 0.0,
            "annotated_video_url": None,
            "stream_url": f"/analysis_jobs/{job_id}/stream",
            "source_path": str(source_path),
            "output_path": str(output_path),
            "_latest_frame": initial_frame,
        }
        with analysis_jobs_lock:
            analysis_jobs[job_id] = initial_job

        worker = threading.Thread(
            target=_run_analysis_job,
            args=(job_id, source_path, output_path),
            daemon=True,
        )
        worker.start()
        return _serialize_job(initial_job)
    except Exception:
        analysis_run_lock.release()
        raise


@app.get("/analysis_jobs/{job_id}")
def analysis_job_status(job_id: str):
    return _serialize_job(_get_job(job_id))


@app.get("/analysis_jobs/{job_id}/stream")
def analysis_job_stream(job_id: str):
    _get_job(job_id)
    return StreamingResponse(
        _analysis_stream(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video")
def video():
    return StreamingResponse(
        _get_live_detector().frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/count")
def count():
    return {"count": _get_live_detector().get_count()}


@app.on_event("shutdown")
def shutdown() -> None:
    if live_detector is not None:
        live_detector.stop()
    if analysis_detector is not None:
        analysis_detector.stop()