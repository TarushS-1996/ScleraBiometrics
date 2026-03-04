from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import base64
import json
from datetime import datetime, timezone

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
    Timestamp is injected automatically in UTC ISO format.
    """
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
                continue  # skip any malformed lines silently

    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries


# =========================================================
# HELPER - Encode image to base64
# =========================================================

def encode_image_to_base64(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


# =========================================================
# GET /logs  — called on frontend startup
# =========================================================

@app.get("/logs")
async def get_logs():
    """
    Return all audit log entries sorted newest first.
    Frontend calls this on startup to populate the Records screen.

    Each entry has an 'action' field:
      'new_user' → a new eye sample was registered
      'match'    → an identification attempt was made
    """
    return JSONResponse(content={"logs": load_logs()})


# =========================================================
# POST /segment  — register a new sample
# =========================================================

@app.post("/segment")
async def segment_endpoint(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    eye_side: str = Form(...)
):
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

    # --- Log registration ---
    append_log({
        "action": "new_user",
        "user_id": user_id,
        "eye_side": eye_side,
        "sample": fname,
        "total_samples": sample_id,
        "save_path": save_path
    })

    processed_base64 = encode_image_to_base64(processed)

    return {
        "message": "Sample stored successfully",
        "user_id": user_id,
        "eye_side": eye_side,
        "sample": fname,
        "total_samples": sample_id,
        "processed_image": processed_base64
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

    # --- Log identification attempt ---
    best = result.get("best_match")
    is_matched = best is not None and best.get("label") == "SAME"

    append_log({
        "action": "match",
        "matched": is_matched,
        "best_match_user_id": best["user_id"] if best else None,
        "best_match_eye_side": best["eye_side"] if best else None,
        "best_match_sample": best["sample"] if best else None,
        "best_match_similarity": round(best["similarity"], 4) if best else None,
        "total_matches": len(result.get("matches", []))
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
# GET /logs/stream  — raw JSONL file stream (fastest retrieval)
# =========================================================

@app.get("/logs/stream")
async def stream_logs():
    """
    Stream the raw audit_log.jsonl file back to the client.

    - No parsing or re-serialization — bytes go straight off disk
    - Client reads line by line; each line is a valid JSON object
    - Use this for bulk retrieval / populating Records on startup
    - Use GET /logs if you need a pre-parsed JSON array instead
    """
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="No log file found yet")

    def file_generator():
        with open(LOG_FILE, "rb") as f:  # binary mode — no encoding overhead
            while chunk := f.read(65536):  # 64KB chunks
                yield chunk

    return StreamingResponse(
        file_generator(),
        media_type="application/x-ndjson",  # NDJSON = newline-delimited JSON
        headers={
            "Content-Disposition": "inline; filename=audit_log.jsonl",
            "X-Log-File": LOG_FILE
        }
    )