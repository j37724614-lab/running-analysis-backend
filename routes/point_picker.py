from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response


router = APIRouter()

VIDEO_PATH = Path("/home/jeter/runner-analysis-pipeline/Long jump mark.mp4")
OUTPUT_PATH = Path("/home/jeter/runner-analysis-pipeline/output_cut/long_jump_mark_homography/cone_points_frame60.json")
HOMOGRAPHY_OUTPUT_DIR = Path("/home/jeter/runner-analysis-pipeline/output_cut/long_jump_mark_homography/result")
RECTIFY_SCRIPT = Path("/home/jeter/runner-analysis-pipeline/homography_point_picker_bundle/pipeline_tools/rectify_video_from_cone_points.py")
PYTHON_BIN = Path("/home/jeter/.conda/envs/yolo_new/bin/python")
FRAME_INDEX = 60


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Video Frame Point Picker</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; background: #111; color: #eee; }
    .row { margin: 10px 0; }
    button { padding: 8px 12px; margin-right: 8px; font-size: 16px; }
    #wrap { position: relative; display: inline-block; max-width: 100%; }
    #img { max-width: min(96vw, 1400px); height: auto; display: block; cursor: crosshair; }
    .pt {
      position: absolute; width: 14px; height: 14px; border-radius: 50%;
      background: #f00; border: 2px solid #fff; transform: translate(-50%, -50%);
      pointer-events: none;
    }
    .label {
      position: absolute; color: #fff; background: rgba(255,0,0,.85);
      padding: 1px 5px; border-radius: 4px; font-size: 14px;
      transform: translate(8px, -24px); pointer-events: none;
    }
    pre { background: #222; padding: 12px; border-radius: 6px; overflow-x: auto; }
    .ok { color: #6ee36e; }
    .warn { color: #ffd166; }
  </style>
</head>
<body>
  <h2>Video Frame Point Picker</h2>
  <div class="row">
    <div><b>Video:</b> <span id="video"></span></div>
    <div><b>Frame:</b> <span id="frame"></span>, <b>time:</b> <span id="time"></span>s</div>
    <div><b>Image size:</b> <span id="size"></span></div>
  </div>
  <div class="row warn">請點角錐「底部接觸地面的中心點」，不要點角錐頂端。</div>
  <div class="row">
    <button onclick="undoPoint()">Undo</button>
    <button onclick="savePoints()">Save JSON/CSV on server</button>
    <button onclick="clearPoints()">Clear</button>
    <span id="status"></span>
  </div>
  <div id="wrap">
    <img id="img" src="./point-picker/frame.jpg" />
  </div>
  <h3>Points</h3>
  <pre id="json"></pre>

<script>
const meta = __META__;
let points = [];
const img = document.getElementById('img');
const wrap = document.getElementById('wrap');
document.getElementById('video').textContent = meta.video;
document.getElementById('frame').textContent = meta.frame;
document.getElementById('time').textContent = meta.time_s == null ? 'unknown' : meta.time_s.toFixed(3);
document.getElementById('size').textContent = `${meta.width} x ${meta.height}`;

function render() {
  document.querySelectorAll('.pt,.label').forEach(e => e.remove());
  const rect = img.getBoundingClientRect();
  const sx = rect.width / meta.width;
  const sy = rect.height / meta.height;
  points.forEach((p, i) => {
    const dot = document.createElement('div');
    dot.className = 'pt';
    dot.style.left = `${p.x * sx}px`;
    dot.style.top = `${p.y * sy}px`;
    wrap.appendChild(dot);
    const lab = document.createElement('div');
    lab.className = 'label';
    lab.style.left = `${p.x * sx}px`;
    lab.style.top = `${p.y * sy}px`;
    lab.textContent = `${i + 1}`;
    wrap.appendChild(lab);
  });
  document.getElementById('json').textContent = JSON.stringify({
    meta,
    points: points.map((p, i) => ({index: i + 1, x: +p.x.toFixed(2), y: +p.y.toFixed(2)}))
  }, null, 2);
}

img.addEventListener('click', (ev) => {
  const rect = img.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * meta.width / rect.width;
  const y = (ev.clientY - rect.top) * meta.height / rect.height;
  points.push({x, y});
  render();
});

window.addEventListener('resize', render);

function undoPoint() {
  points.pop();
  render();
}

function clearPoints() {
  points = [];
  render();
}

async function savePoints() {
  const payload = {
    meta,
    points: points.map((p, i) => ({index: i + 1, x: +p.x.toFixed(2), y: +p.y.toFixed(2)}))
  };
  const res = await fetch('./point-picker/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  const status = document.getElementById('status');
  if (res.ok) {
    status.className = 'ok';
    status.textContent = `Saved: ${data.json_path}`;
  } else {
    status.className = 'warn';
    status.textContent = `Save failed: ${data.error || res.status}`;
  }
}

render();
</script>
</body>
</html>
"""


def _read_frame() -> tuple[bytes, dict]:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total_frames and FRAME_INDEX >= total_frames:
        cap.release()
        raise ValueError(f"Frame {FRAME_INDEX} outside range 0..{total_frames - 1}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {FRAME_INDEX}")
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame")
    meta = {
        "video": str(VIDEO_PATH),
        "frame": FRAME_INDEX,
        "fps": fps,
        "width": width,
        "height": height,
        "time_s": FRAME_INDEX / fps if fps > 0 else None,
        "total_frames": total_frames,
    }
    return encoded.tobytes(), meta


def _write_points(payload: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = OUTPUT_PATH.with_suffix(".csv")
    rows = []
    for i, point in enumerate(payload.get("points", []), start=1):
        rows.append(
            {
                "id": point.get("id", point.get("index", i)),
                "index": point.get("index", point.get("id", i)),
                "x": point.get("x"),
                "y": point.get("y"),
                "world_x_m": point.get("world_x_m"),
                "world_y_m": point.get("world_y_m"),
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "index", "x", "y", "world_x_m", "world_y_m"])
        writer.writeheader()
        writer.writerows(rows)


@router.get("/point-picker", response_class=HTMLResponse)
def point_picker() -> HTMLResponse:
    try:
        _, meta = _read_frame()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return HTMLResponse(HTML.replace("__META__", json.dumps(meta, ensure_ascii=False)))


@router.get("/point-picker/frame.jpg")
def point_picker_frame() -> Response:
    try:
        frame_jpg, _ = _read_frame()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(frame_jpg, media_type="image/jpeg")


@router.post("/point-picker/save")
async def point_picker_save(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        _write_points(payload)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(
        {
            "ok": True,
            "json_path": str(OUTPUT_PATH),
            "csv_path": str(OUTPUT_PATH.with_suffix(".csv")),
        }
    )


def _save_cone_points(payload: dict) -> None:
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 4:
        raise ValueError("need at least four points")
    for i, point in enumerate(points, start=1):
        for key in ("x", "y", "world_x_m", "world_y_m"):
            if key not in point or point[key] is None:
                raise ValueError(f"point {i} missing {key}")
    _write_points(payload)


def _cone_homography_html() -> str:
    html_path = Path("/home/jeter/runner-analysis-pipeline/homography_point_picker_bundle/frontend_web/click_points_cone.html")
    html = html_path.read_text(encoding="utf-8")
    return (
        html.replace('src="/click_points_frame60.jpg"', 'src="./cone-homography/frame.jpg"')
        .replace('"1.22m_804998631.923345.mp4"', json.dumps(VIDEO_PATH.name))
        .replace("postJson('/save_cone_points')", "postJson('./cone-homography/save_cone_points')")
        .replace("postJson('/run_cone_rectification')", "postJson('./cone-homography/run_cone_rectification')")
    )


@router.get("/cone-homography", response_class=HTMLResponse)
def cone_homography() -> HTMLResponse:
    return HTMLResponse(_cone_homography_html())


@router.get("/cone-homography/frame.jpg")
def cone_homography_frame() -> Response:
    return point_picker_frame()


@router.post("/cone-homography/save_cone_points")
async def save_cone_points(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        _save_cone_points(payload)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "json_path": str(OUTPUT_PATH)})


@router.post("/cone-homography/run_cone_rectification")
async def run_cone_rectification(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        _save_cone_points(payload)
        HOMOGRAPHY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PYTHON_BIN),
            str(RECTIFY_SCRIPT),
            "--video",
            str(VIDEO_PATH),
            "--points-json",
            str(OUTPUT_PATH),
            "--output-dir",
            str(HOMOGRAPHY_OUTPUT_DIR),
            "--frame",
            str(payload.get("frame_index", FRAME_INDEX)),
            "--max-frames",
            "0",
            "--control-indices",
            "1,2,3,4,5,6",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if proc.returncode != 0:
        return JSONResponse(
            {
                "ok": False,
                "error": "rectification failed",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            status_code=500,
        )
    return JSONResponse(
        {
            "ok": True,
            "json_path": str(OUTPUT_PATH),
            "output_dir": str(HOMOGRAPHY_OUTPUT_DIR),
            "homography_rectified_preview": str(HOMOGRAPHY_OUTPUT_DIR / "homography_rectified_preview.mp4"),
            "cone_distance_check": str(HOMOGRAPHY_OUTPUT_DIR / "cone_distance_check.csv"),
            "homography_debug_overlay": str(HOMOGRAPHY_OUTPUT_DIR / "homography_debug_overlay.png"),
            "stdout": proc.stdout,
        }
    )
