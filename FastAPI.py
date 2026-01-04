from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import uuid

from pipeline.segmentation import *  # your file
from pipeline.siamese import *      # your file

app = FastAPI()

STORAGE_DIR = "storage/segmented"
os.makedirs(STORAGE_DIR, exist_ok=True)


@app.post("/segment")
async def segment_endpoint(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    eye_side: str = Form(...)
):
    """
    Upload eye image → sclera segmentation + vein skeletonization → store result
    """

    # -------------------------
    # Read image bytes → OpenCV
    # -------------------------
    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )

    # -------------------------
    # Run your segmentation
    # -------------------------
    segmented = predict_sclera_and_vessels(
        img_bgr,
        plot=False   # ❗ important for API
    )

    # -------------------------
    # Store result
    # -------------------------
    ref_id = f"{user_id}_{eye_side}"
    save_path = os.path.join(STORAGE_DIR, f"{ref_id}.png")

    cv2.imwrite(save_path, segmented)

    # -------------------------
    # API response
    # -------------------------
    return {
        "reference_id": ref_id,
        "user_id": user_id,
        "eye_side": eye_side,
        "image_path": save_path,
        "message": "Segmentation successful"
    }


@app.post("/compare")
async def compare_endpoint(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    eye_side: str = Form(...)
):
    """
    Compare a new eye image against a stored processed eye image
    using the Siamese network.
    """

    # -------------------------------------------------
    # Validate eye side
    # -------------------------------------------------
    eye_side = eye_side.capitalize()
    if eye_side not in ["Left", "Right"]:
        raise HTTPException(status_code=400, detail="eye_side must be 'Left' or 'Right'")

    # -------------------------------------------------
    # Load uploaded image
    # -------------------------------------------------
    contents = await image.read()
    img_np = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------
    # Run segmentation + skeletonization
    # -------------------------------------------------
    try:
        processed_img = predict_sclera_and_vessels(
            img,
            plot=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

    # -------------------------------------------------
    # Siamese comparison
    # -------------------------------------------------
    try:
        result = compare_processed_eye(
            processed_img=processed_img,
            user_id=user_id,
            eye_side=eye_side,
            base_dir=STORAGE_DIR,
            threshold=0.58
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Siamese comparison failed: {str(e)}")

    return {
        "user_id": user_id,
        "eye_side": eye_side,
        "distance": result["distance"],
        "similarity": result["similarity"],
        "label": result["label"],
    }


@app.post("/identify")
async def identify_eye(
    image: UploadFile = File(...)
):
    """
    Identify a processed sclera image against all stored processed images.
    """

    # -----------------------------
    # Read uploaded processed image
    # -----------------------------
    img_bytes = await image.read()
    query = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if query is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image upload"}
        )

    query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

    try:
        query = predict_sclera_and_vessels(
            query,
            plot=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

    # Match training normalization
    try:
        result = identify_processed_eye_across_database(
            processed_query_img=query,
            processed_dir=STORAGE_DIR,
            threshold=0.75
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")
    
    return {
        "matches": result["matches"]
    }

    
