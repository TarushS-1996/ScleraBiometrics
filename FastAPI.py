from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import uuid
import base64

from pipeline.segmentation import *  # your file
from pipeline.siamese import *      # your file


app = FastAPI()

STORAGE_DIR = "storage/segmented"
os.makedirs(STORAGE_DIR, exist_ok=True)

# =========================================================
# HELPER FUNCTION - Encode image to base64
# =========================================================
def encode_image_to_base64(img_rgb: np.ndarray) -> str:
    """
    Convert RGB numpy array to base64 string for JSON transmission
    """
    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Encode as PNG
    _, buffer = cv2.imencode('.png', img_bgr)
    # Convert to base64 string
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# =========================================================
# REGISTER / ADD SAMPLE
# =========================================================
@app.post("/segment")
async def segment_endpoint(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    eye_side: str = Form(...)
):
    """
    Upload eye image → segment → store as a NEW sample
    Returns the processed image for display in UI
    """

    eye_side = eye_side.capitalize()
    if eye_side not in ["Left", "Right"]:
        raise HTTPException(status_code=400, detail="eye_side must be Left or Right")

    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # -------------------------------------------------
    # Run segmentation ONCE
    # -------------------------------------------------
    processed = predict_sclera_and_vessels(img, plot=False)

    # -------------------------------------------------
    # Save as a new sample
    # -------------------------------------------------
    user_eye_dir = os.path.join(STORAGE_DIR, user_id, eye_side)
    os.makedirs(user_eye_dir, exist_ok=True)

    existing = sorted(
        f for f in os.listdir(user_eye_dir) if f.endswith(".png")
    )
    sample_id = len(existing) + 1
    fname = f"sample_{sample_id:03d}.png"

    save_path = os.path.join(user_eye_dir, fname)
    cv2.imwrite(save_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

    # -------------------------------------------------
    # Encode processed image as base64 for frontend
    # -------------------------------------------------
    processed_base64 = encode_image_to_base64(processed)

    return {
        "message": "Sample stored successfully",
        "user_id": user_id,
        "eye_side": eye_side,
        "sample": fname,
        "total_samples": sample_id,
        "processed_image": processed_base64  # For top-right display
    }


# =========================================================
# OPTIONAL: DIRECT COMPARE (kept for backward compatibility)
# =========================================================
@app.post("/compare")
async def compare_endpoint(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    eye_side: str = Form(...)
):
    """
    Compare against BEST sample of a specific user + eye
    """

    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    processed = predict_sclera_and_vessels(img, plot=False)

    result = compare_processed_eye(
        processed_img=processed,
        user_id=user_id,
        eye_side=eye_side.capitalize(),
        base_dir=STORAGE_DIR,
        threshold=0.75
    )

    return result


# =========================================================
# IDENTIFY ACROSS ALL USERS (MAIN MATCH ENDPOINT)
# =========================================================
@app.post("/identify")
async def identify_eye(
    image: UploadFile = File(...)
):
    """
    Identify query eye against ALL users + ALL samples
    Returns:
    - Match results with similarity scores
    - Processed query image (for top-right display)
    - Matched sample image (for bottom-left preview)
    """

    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Process the query image
    processed = predict_sclera_and_vessels(img, plot=False)

    # Use nested traversal to find matches
    result = identify_processed_eye_across_database_nested(
        processed_query_img=processed,
        base_dir=STORAGE_DIR,
        threshold=0.75
    )

    # -------------------------------------------------
    # Encode processed query image for top-right display
    # -------------------------------------------------
    processed_query_base64 = encode_image_to_base64(processed)
    result["processed_query_image"] = processed_query_base64

    # -------------------------------------------------
    # Load and encode the best matched sample for bottom-left preview
    # -------------------------------------------------
    if result["best_match"]:
        best_match_path = result["best_match"]["path"]
        if os.path.exists(best_match_path):
            matched_img_bgr = cv2.imread(best_match_path)
            matched_img_rgb = cv2.cvtColor(matched_img_bgr, cv2.COLOR_BGR2RGB)
            matched_base64 = encode_image_to_base64(matched_img_rgb)
            result["best_match"]["matched_image"] = matched_base64
        else:
            result["best_match"]["matched_image"] = None
    
    # -------------------------------------------------
    # Optionally encode all matched samples (for future use)
    # -------------------------------------------------
    for match in result.get("matches", []):
        if os.path.exists(match["path"]):
            matched_img_bgr = cv2.imread(match["path"])
            matched_img_rgb = cv2.cvtColor(matched_img_bgr, cv2.COLOR_BGR2RGB)
            match["matched_image"] = encode_image_to_base64(matched_img_rgb)
        else:
            match["matched_image"] = None

    return result