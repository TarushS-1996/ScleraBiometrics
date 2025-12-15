import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt 
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize
from skimage import img_as_float

# --------------------------------------------
# Load Models
# --------------------------------------------
SEG_MODEL_PATH = "Model/sclera_iris_segmentation_model.h5"
seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)


def sclera_roi(img, height_ratio=0.35):
    """
    Crops the central sclera band, avoiding top/bottom lashes.
    height_ratio defines how tall the middle region is.
    """
    h, w = img.shape[:2]
    band_h = int(h * height_ratio)
    top = (h - band_h) // 2
    bottom = top + band_h
    return img[top:bottom, :]

def extract_focused_veins(img_rgb):
    """
    Extracts only the prominent veins from the clean central sclera region.
    """
    h, w = img_rgb.shape[:2]
    roi = sclera_roi(img_rgb, height_ratio=0.35)

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    gray_norm = cv2.normalize(gray, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Vesselness + gamma boost
    v = frangi(img_as_float(gray_norm), sigmas=range(1,6))
    v = np.nan_to_num(v, nan=0.0)
    v = cv2.normalize(v, None, 0, 1.0, cv2.NORM_MINMAX)
    v = cv2.pow(v, 0.6)
    v = (v * 255).astype(np.uint8)

    # Sensitive threshold
    mean_val, std_val = cv2.meanStdDev(v)
    mean_val, std_val = float(mean_val), float(std_val)
    _, mask = cv2.threshold(v, mean_val + 0.15 * std_val, 255, cv2.THRESH_BINARY)

    # Cleanup + thin veins
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    skel = skeletonize(mask > 0)
    veins = (skel * 255).astype(np.uint8)

    # Re-embed into full frame
    full = np.zeros((h, w), dtype=np.uint8)
    top = (h - roi.shape[0]) // 2
    full[top:top + roi.shape[0], :] = veins
    return full


def predict_sclera_and_vessels(image_path, save_dir="results",
                               seg_size=(128,128), full_size=(512,512),
                               sat_boost=1.4, clip_high=4.0, clip_low=2.0, plot=True):
    """
    Returns a 256×256×3 sclera-enhanced image compatible with the Siamese network.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Load image ---
    img_bgr = image_path
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_full = cv2.resize(img_rgb, full_size)

    # --- Segment sclera (low-res) ---
    img_small = cv2.resize(img_rgb, seg_size)
    inp = np.expand_dims(img_small / 255.0, axis=0)
    pred = seg_model.predict(inp, verbose=0)
    pred_mask = np.argmax(pred[0], axis=-1)

    # --- Stretch mask to full resolution ---
    mask_resized = cv2.resize(pred_mask.astype(np.uint8), full_size,
                              interpolation=cv2.INTER_NEAREST)

    # --- Extract sclera ---
    sclera_mask = (mask_resized == 1).astype(np.uint8) * 255
    sclera_only = cv2.bitwise_and(img_full, img_full, mask=sclera_mask)

    # --- CLAHE locally ---
    gray = cv2.cvtColor(sclera_only, cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(gray)
    clip = clip_low if mean_intensity > 110 else clip_high
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    L = clahe.apply(gray)

    lab = cv2.cvtColor(sclera_only, cv2.COLOR_RGB2LAB)
    L_orig, A, B = cv2.split(lab)
    lab_enh = cv2.merge([L, A, B])
    sclera_clahe = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB)

    # --- Saturation boost ---
    hsv = cv2.cvtColor(sclera_clahe, cv2.COLOR_RGB2HSV)
    hsv[...,1] = cv2.multiply(hsv[...,1], sat_boost)
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    sclera_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # --- Extract veins (your function) ---
    veinmap = extract_focused_veins(sclera_sat)   # ← 1-channel

    # ---------------------------------------------------------
    # 🔥 FIX 1 — Resize everything to 256×256
    # ---------------------------------------------------------
    sclera_sat_256 = cv2.resize(sclera_sat, (256,256))
    veinmap_256 = cv2.resize(veinmap, (256,256))

    # ---------------------------------------------------------
    # 🔥 FIX 2 — Convert veinmap (1-channel) → RGB
    # ---------------------------------------------------------
    veinmap_rgb = cv2.cvtColor(veinmap_256, cv2.COLOR_GRAY2RGB)

    # ---------------------------------------------------------
    # 🔥 Siamese expects (256,256,3)
    # ---------------------------------------------------------
    final_out = veinmap_rgb.astype(np.uint8)


    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    if plot:
        plt.figure(figsize=(14,4))
        plt.subplot(1,3,1); plt.imshow(sclera_only); plt.title("Sclera Only"); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(veinmap_256); plt.title("Vein Map (256×256)"); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(final_out.astype(np.uint8)); plt.title("Final Output (RGB)"); plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return final_out
