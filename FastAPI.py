from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import os
import base64
import json
import uuid
import shutil
from datetime import datetime, timezone
from typing import Optional

from fastapi.responses import StreamingResponse
from pipeline.segmentation import *
from pipeline.siamese import *


app = FastAPI()

STORAGE_DIR = "storage/segmented"
LOG_FILE = "storage/audit_log.jsonl"

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs("storage", exist_ok=True)


# =========================================================
# JSONL AUDIT LOG HELPERS
# =========================================================

def append_log(entry: dict):
    """
    Append a structured record to the JSONL audit log.
    record_id (UUID) and timestamp (UTC ISO) are injected automatically.
    """
    entry["record_id"] = str(uuid.uuid4())
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_logs() -> list:
    """
    Load all log entries. Returns sorted descending by timestamp (newest first).
    """
    if not os.path.exists(LOG_FILE):
        return []

    entries = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries


def rewrite_logs(entries: list):
    """Overwrite the JSONL log file with the given list of entries."""
    with open(LOG_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# =========================================================
# HELPER — Encode image to base64
# =========================================================

def encode_image_to_base64(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


# =========================================================
# GET /logs  — filterable, paginated log retrieval
# =========================================================

@app.get("/logs")
async def get_logs(
    action: Optional[str] = Query(None, description="Filter by action: 'new_user' or 'match'"),
    user_id: Optional[str] = Query(None, description="Filter by user_id (partial match)"),
    matched: Optional[bool] = Query(None, description="For match entries: filter by matched=true/false"),
    from_date: Optional[str] = Query(None, description="ISO date string — return entries at or after this time"),
    to_date: Optional[str] = Query(None, description="ISO date string — return entries at or before this time"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of entries to return"),
    offset: int = Query(0, ge=0, description="Number of entries to skip (for pagination)"),
):
    """
    Return audit log entries with optional filtering and pagination.

    Filters are applied server-side before slicing. All filters are optional
    and can be combined. Results are always newest-first.

    Query params:
      action     — 'new_user' | 'match'
      user_id    — partial case-insensitive match against user_id
      matched    — true/false (only applies to match entries)
      from_date  — ISO 8601 string, inclusive lower bound
      to_date    — ISO 8601 string, inclusive upper bound
      limit      — page size (default 100, max 1000)
      offset     — page start index (default 0)
    """
    entries = load_logs()

    # --- action filter ---
    if action:
        entries = [e for e in entries if e.get("action") == action]

    # --- user_id filter (checks both registration and match fields) ---
    if user_id:
        uid_lower = user_id.lower()
        def matches_user(e):
            reg_uid = (e.get("user_id") or "").lower()
            match_uid = (e.get("best_match_user_id") or "").lower()
            return uid_lower in reg_uid or uid_lower in match_uid
        entries = [e for e in entries if matches_user(e)]

    # --- matched filter (for match-type entries only) ---
    if matched is not None:
        entries = [
            e for e in entries
            if e.get("action") != "match" or e.get("matched") == matched
        ]

    # --- date range filters ---
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date)
            entries = [
                e for e in entries
                if datetime.fromisoformat(e.get("timestamp", "1970-01-01T00:00:00+00:00")) >= from_dt
            ]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid from_date format. Use ISO 8601.")

    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date)
            entries = [
                e for e in entries
                if datetime.fromisoformat(e.get("timestamp", "1970-01-01T00:00:00+00:00")) <= to_dt
            ]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid to_date format. Use ISO 8601.")

    total = len(entries)
    page = entries[offset: offset + limit]

    return JSONResponse(content={
        "total": total,
        "offset": offset,
        "limit": limit,
        "logs": page,
    })


# =========================================================
# GET /logs/export  — download full log file
# =========================================================

@app.get("/logs/export")
async def export_logs():
    """
    Download the full audit_log.jsonl file as an attachment.

    The client receives the raw JSONL file which can be saved locally.
    Each line is a valid JSON object (NDJSON format).
    """
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="No log file found yet.")

    filename = f"audit_log_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    return FileResponse(
        LOG_FILE,
        media_type="application/x-ndjson",
        filename=filename,
    )


# =========================================================
# DELETE /logs/{record_id}  — delete a single log entry
# =========================================================

@app.delete("/logs/{record_id}")
async def delete_log_entry(record_id: str):
    """
    Delete a single audit log entry by its record_id (UUID).

    The log file is rewritten in-place with the matching entry removed.
    Returns 404 if no entry with that record_id exists.
    """
    entries = load_logs()
    original_count = len(entries)
    filtered = [e for e in entries if e.get("record_id") != record_id]

    if len(filtered) == original_count:
        raise HTTPException(status_code=404, detail=f"No log entry found with record_id='{record_id}'.")

    # Restore chronological order before rewriting (load_logs returns newest-first)
    filtered.sort(key=lambda x: x.get("timestamp", ""))
    rewrite_logs(filtered)

    return JSONResponse(content={"deleted": record_id, "remaining": len(filtered)})


# =========================================================
# DELETE /user/{user_id}  — delete a user and all their samples
# =========================================================

@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    """
    Delete a user's stored eye samples from disk and purge all their log entries.

    Removes: storage/segmented/{user_id}/ and all subdirectories.
    Purges:  all 'new_user' log entries where user_id matches,
             and all 'match' entries where best_match_user_id matches.
    """
    user_dir = os.path.join(STORAGE_DIR, user_id)

    if not os.path.exists(user_dir):
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found in storage.")

    shutil.rmtree(user_dir)

    # Purge log entries for this user
    entries = load_logs()
    filtered = [
        e for e in entries
        if not (
            (e.get("action") == "new_user" and e.get("user_id") == user_id) or
            (e.get("action") == "match" and e.get("best_match_user_id") == user_id)
        )
    ]
    filtered.sort(key=lambda x: x.get("timestamp", ""))
    rewrite_logs(filtered)

    purged_count = len(entries) - len(filtered)
    return JSONResponse(content={
        "deleted_user": user_id,
        "storage_removed": user_dir,
        "log_entries_purged": purged_count,
    })


# =========================================================
# GET /image/{user_id}/{eye_side}/{sample}  — serve stored eye image
# =========================================================

@app.get("/image/{user_id}/{eye_side}/{sample}")
async def get_image(user_id: str, eye_side: str, sample: str):
    """
    Return a stored segmented eye image as a base64-encoded PNG.

    Path: storage/segmented/{user_id}/{eye_side}/{sample}
    Eye side is capitalised automatically (left → Left).

    Response: { "image": "<base64 string>" }
    """
    eye_side = eye_side.capitalize()
    image_path = os.path.join(STORAGE_DIR, user_id, eye_side, sample)

    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {image_path}"
        )

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to read image from disk.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    encoded = encode_image_to_base64(img_rgb)

    return JSONResponse(content={"image": encoded, "path": image_path})


# =========================================================
# POST /segment  — register a new sample
# =========================================================

@app.post("/segment")
async def segment_endpoint(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    eye_side: str = Form(...),
    first_name: str = Form(""),
    last_name: str = Form(""),
):
    """
    Segment and store a new eye sample for a user.

    first_name and last_name are optional and stored in the audit log
    for display purposes. user_id remains the primary filesystem key.
    """
    eye_side = eye_side.capitalize()
    if eye_side not in ["Left", "Right"]:
        raise HTTPException(status_code=400, detail="eye_side must be Left or Right")

    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    processed = predict_sclera_and_vessels(img, plot=False)

    user_eye_dir = os.path.join(STORAGE_DIR, user_id, eye_side)
    os.makedirs(user_eye_dir, exist_ok=True)

    existing = sorted(f for f in os.listdir(user_eye_dir) if f.endswith(".png"))
    sample_id = len(existing) + 1
    fname = f"sample_{sample_id:03d}.png"
    save_path = os.path.join(user_eye_dir, fname)

    cv2.imwrite(save_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

    append_log({
        "action": "new_user",
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "eye_side": eye_side,
        "sample": fname,
        "total_samples": sample_id,
        "image_path": save_path,
    })

    processed_base64 = encode_image_to_base64(processed)

    return {
        "message": "Sample stored successfully",
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "eye_side": eye_side,
        "sample": fname,
        "total_samples": sample_id,
        "processed_image": processed_base64,
    }


# =========================================================
# POST /compare  — direct compare (backward compat)
# =========================================================

@app.post("/compare")
async def compare_endpoint(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    eye_side: str = Form(...)
):
    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    processed = predict_sclera_and_vessels(img, plot=False)

    return compare_processed_eye(
        processed_img=processed,
        user_id=user_id,
        eye_side=eye_side.capitalize(),
        base_dir=STORAGE_DIR,
        threshold=0.75
    )


# =========================================================
# POST /identify  — match against all users
# =========================================================

@app.post("/identify")
async def identify_eye(
    image: UploadFile = File(...)
):
    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    processed = predict_sclera_and_vessels(img, plot=False)

    result = identify_processed_eye_across_database_nested(
        processed_query_img=processed,
        base_dir=STORAGE_DIR,
        threshold=0.75
    )

    best = result.get("best_match")
    is_matched = best is not None and best.get("label") == "SAME"

    # Reconstruct the image_path for match log entries so the Records
    # screen can serve the matched image via GET /image/...
    best_image_path = None
    if best:
        best_image_path = os.path.join(
            STORAGE_DIR,
            best["user_id"],
            best["eye_side"],
            best["sample"],
        )

    append_log({
        "action": "match",
        "matched": is_matched,
        "best_match_user_id": best["user_id"] if best else None,
        "best_match_eye_side": best["eye_side"] if best else None,
        "best_match_sample": best["sample"] if best else None,
        "best_match_similarity": round(best["similarity"], 4) if best else None,
        "image_path": best_image_path,
        "total_matches": len(result.get("matches", [])),
    })

    # Encode processed query image
    result["processed_query_image"] = encode_image_to_base64(processed)

    # Encode best match image
    if result["best_match"]:
        path = result["best_match"]["path"]
        if os.path.exists(path):
            matched_bgr = cv2.imread(path)
            matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)
            result["best_match"]["matched_image"] = encode_image_to_base64(matched_rgb)
        else:
            result["best_match"]["matched_image"] = None

    # Encode all matches
    for match in result.get("matches", []):
        if os.path.exists(match["path"]):
            m_bgr = cv2.imread(match["path"])
            m_rgb = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB)
            match["matched_image"] = encode_image_to_base64(m_rgb)
        else:
            match["matched_image"] = None

    return result


# =========================================================
# GET /logs/stream  — raw JSONL file stream
# =========================================================

@app.get("/logs/stream")
async def stream_logs():
    """
    Stream the raw audit_log.jsonl file back to the client as NDJSON.

    IMPORTANT: This is NOT a JSON array. Each line is an independent JSON
    object. The client must parse line-by-line, NOT call JSON.parse() on
    the full response body.

    Python client example:
        for line in response.iter_lines():
            if line:
                entry = json.loads(line)

    Use GET /logs if you need a pre-parsed JSON array with filter support.
    """
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="No log file found yet")

    def file_generator():
        with open(LOG_FILE, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        file_generator(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": "inline; filename=audit_log.jsonl",
            "X-Log-File": LOG_FILE,
        }
    )
