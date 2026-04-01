from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from src.analyzer import ErgoVideoAnalyzer

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output"
JOBS_DIR = BASE_DIR / "jobs"
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_full.task"

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024 * 2  # 2GB

job_file_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def job_json_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)

    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.stem + "_",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

        last_error = None
        for _ in range(10):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as e:
                last_error = e
                time.sleep(0.05)

        if last_error:
            raise last_error

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def save_job(job_id: str, data: dict) -> None:
    path = job_json_path(job_id)
    data = dict(data)
    data["updated_at"] = datetime.now().isoformat()

    with job_file_lock:
        atomic_write_json(path, data)


def load_job(job_id: str, retries: int = 10, retry_sleep: float = 0.05) -> dict | None:
    path = job_json_path(job_id)
    if not path.exists():
        return None

    for attempt in range(retries):
        try:
            with job_file_lock:
                if not path.exists():
                    return None

                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    raise json.JSONDecodeError("empty json", "", 0)

                return json.loads(text)

        except (json.JSONDecodeError, PermissionError):
            if attempt == retries - 1:
                raise
            time.sleep(retry_sleep)

    return None


def update_job(job_id: str, **kwargs) -> None:
    path = job_json_path(job_id)

    with job_file_lock:
        if not path.exists():
            return

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            job = {}
        else:
            job = json.loads(text)

        job.update(kwargs)
        job["updated_at"] = datetime.now().isoformat()
        atomic_write_json(path, job)


def create_job(filename: str, saved_input_path: Path) -> dict:
    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat()

    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "待機中",
        "created_at": now,
        "updated_at": now,
        "input_filename": filename,
        "input_path": str(saved_input_path),
        "result": None,
        "error": None,
    }
    save_job(job_id, job)
    return job


def build_public_result(job_id: str, job: dict) -> dict:
    result = job.get("result") or {}
    public = {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "input_filename": job.get("input_filename"),
        "result": None,
    }

    if result:
        public["result"] = {
            "summary_json_url": url_for("download_result_file", job_id=job_id, kind="summary_json"),
            "report_txt_url": url_for("download_result_file", job_id=job_id, kind="report_txt"),
            "overlay_video_url": url_for("serve_result_file", job_id=job_id, kind="overlay_video"),
            "vtt_url": url_for("serve_result_file", job_id=job_id, kind="vtt"),
            "download_video_url": url_for("download_result_file", job_id=job_id, kind="overlay_video"),
            "download_vtt_url": url_for("download_result_file", job_id=job_id, kind="vtt"),
            "summary_text": result.get("summary_text", ""),
        }

    return public


def run_analysis_job(job_id: str) -> None:
    try:
        job = load_job(job_id)
        if job is None:
            return

        update_job(
            job_id,
            status="running",
            progress=3,
            message="解析を開始しました",
        )

        input_path = Path(job["input_path"])
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        def progress_callback(progress: int, message: str) -> None:
            update_job(
                job_id,
                progress=max(0, min(100, int(progress))),
                message=message,
            )

        analyzer = ErgoVideoAnalyzer(
            model_path=MODEL_PATH,
            output_dir=output_dir,
            person_index=0,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            write_video=True,
            progress_callback=progress_callback,
        )

        result = analyzer.run(input_path)

        update_job(
            job_id,
            status="completed",
            progress=100,
            message="解析が完了しました",
            result=result,
            error=None,
        )

    except Exception as e:
        update_job(
            job_id,
            status="failed",
            progress=100,
            message="解析に失敗しました",
            error=str(e),
        )


@app.route("/")
def index():
    model_exists = MODEL_PATH.exists()
    return render_template("index.html", model_exists=model_exists)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if not MODEL_PATH.exists():
        return jsonify({"ok": False, "error": f"モデルが見つかりません: {MODEL_PATH}"}), 500

    if "file" not in request.files:
        return jsonify({"ok": False, "error": "ファイルがありません"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "ファイル名が空です"}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "許可されていない拡張子です"}), 400

    safe_name = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{timestamp}_{safe_name}"
    saved_input_path = INPUT_DIR / saved_name
    file.save(saved_input_path)

    job = create_job(file.filename, saved_input_path)

    th = threading.Thread(target=run_analysis_job, args=(job["job_id"],), daemon=True)
    th.start()

    return jsonify({
        "ok": True,
        "job_id": job["job_id"],
        "status_url": url_for("api_job_status", job_id=job["job_id"]),
    })


@app.route("/api/jobs/<job_id>/status", methods=["GET"])
def api_job_status(job_id: str):
    try:
        job = load_job(job_id)
    except json.JSONDecodeError:
        return jsonify({
            "ok": False,
            "error": "ジョブ状態ファイルの読み込みに失敗しました。再試行してください。"
        }), 500

    if job is None:
        return jsonify({"ok": False, "error": "job not found"}), 404

    return jsonify({"ok": True, **build_public_result(job_id, job)})


@app.route("/results/<job_id>/<kind>", methods=["GET"])
def serve_result_file(job_id: str, kind: str):
    job = load_job(job_id)
    if job is None or not job.get("result"):
        return jsonify({"ok": False, "error": "result not found"}), 404

    result = job["result"]
    mapping = {
        "overlay_video": result.get("video_path"),
        "vtt": result.get("vtt_path"),
    }
    file_path = mapping.get(kind)
    if not file_path:
        return jsonify({"ok": False, "error": "file kind not found"}), 404

    file_path = Path(file_path)
    if not file_path.exists():
        return jsonify({"ok": False, "error": "file missing"}), 404

    if kind == "vtt":
        return send_file(file_path, mimetype="text/vtt", as_attachment=False)

    return send_from_directory(file_path.parent, file_path.name, as_attachment=False)


@app.route("/download/<job_id>/<kind>", methods=["GET"])
def download_result_file(job_id: str, kind: str):
    job = load_job(job_id)
    if job is None or not job.get("result"):
        return jsonify({"ok": False, "error": "result not found"}), 404

    result = job["result"]
    mapping = {
        "summary_json": result.get("summary_path"),
        "report_txt": result.get("report_txt_path"),
        "overlay_video": result.get("video_path"),
        "vtt": result.get("vtt_path"),
    }
    file_path = mapping.get(kind)
    if not file_path:
        return jsonify({"ok": False, "error": "file kind not found"}), 404

    file_path = Path(file_path)
    if not file_path.exists():
        return jsonify({"ok": False, "error": "file missing"}), 404

    return send_file(file_path, as_attachment=True, download_name=file_path.name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5302, debug=False, threaded=True)