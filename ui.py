import streamlit as st
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
SIAMESE_WEIGHTS_PATH = "Model/siamese_model_trained.weights.h5"

# load segmentation model
seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)

# rebuild siamese and load weights
def build_sclera_siamese(input_shape=(256,256,3)):
    inpA = tf.keras.layers.Input(shape=input_shape)
    inpB = tf.keras.layers.Input(shape=input_shape)

    def enc_block():
        inp = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(128, activation=None)(x)
        out = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(out)
        return tf.keras.Model(inp, out)

    encoder = enc_block()
    featA = encoder(inpA)
    featB = encoder(inpB)

    dist = tf.keras.layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]),
                                        axis=1,
                                        keepdims=True) + 1e-6)
    )([featA, featB])

    return tf.keras.Model([inpA, inpB], dist)

siamese_model = build_sclera_siamese()
siamese_model.load_weights(SIAMESE_WEIGHTS_PATH)

# --------------------------------------------
# Utility Functions
# --------------------------------------------

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


def ssim_similarity(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(g1, g2, full=True)
    return float(score)


# folder where processed images are saved
BASE_DIR = "data/users"
os.makedirs(BASE_DIR, exist_ok=True)

# --------------------------------------------
# Streamlit UI
# --------------------------------------------

st.title("👁️ Sclera Vein Matching System")

tab1, tab2 = st.tabs(["➕ Add New User", "🔍 Verify / Similarity Check"])


# -----------------------------------------------------
#  TAB 1 — Add New User
# -----------------------------------------------------
with tab1:
    st.header("Add a New User to Database")

    username = st.text_input("Enter user name:")
    eye_side = st.selectbox("Eye side", ["Left", "Right"])

    uploaded = st.file_uploader("Upload eye image", type=["jpg", "jpeg", "png"])

    if uploaded and username.strip():
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, caption="Original uploaded image", width=250)

        if st.button("Process & Save"):
            seg = predict_sclera_and_vessels(image)

            save_dir = os.path.join(BASE_DIR, username)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{eye_side}.png")
            cv2.imwrite(save_path, cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))

            st.success(f"Saved segmented eye to {save_path}")
            st.image(seg, caption="Segmented sclera")


# -----------------------------------------------------
#  TAB 2 — Verify User
# -----------------------------------------------------
with tab2:
    st.header("Verify Identity using Siamese Model")

    users = sorted(os.listdir(BASE_DIR))
    if not users:
        st.warning("No users added yet.")
        st.stop()

    user_sel = st.selectbox("Select stored user", users)

    user_path = os.path.join(BASE_DIR, user_sel)
    available_eyes = [f for f in os.listdir(user_path) if f.endswith(".png")]

    if not available_eyes:
        st.warning("This user has no stored eyes.")
        st.stop()

    stored_eye = st.selectbox("Select stored eye", available_eyes)
    stored_img = cv2.cvtColor(cv2.imread(os.path.join(user_path, stored_eye)), cv2.COLOR_BGR2RGB)

    new_img_file = st.file_uploader("Upload image to verify", type=["jpg","jpeg","png"])

    if new_img_file:
        new = cv2.imdecode(np.frombuffer(new_img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        new = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)

        st.image(new, caption="New image", width=250)

        if st.button("Compute Similarity"):
            seg_new = predict_sclera_and_vessels(new)
            X1 = np.expand_dims(seg_new, axis=0)
            X2 = np.expand_dims(stored_img, axis=0)

            # -----------------------------
            #  Siamese distance prediction
            # -----------------------------
            dist = float(siamese_model.predict([X1, X2])[0][0])
            similarity = 1.0 - dist

            # Same threshold as notebook
            threshold = 0.79
            label = "SAME" if similarity >= threshold else "DIFFERENT"

            # -----------------------------
            #  SSIM between raw resized images
            # -----------------------------
            img1_r = cv2.resize(stored_img, (128,128))
            img2_r = cv2.resize(new, (128,128))
            ssim_score = ssim_similarity(img1_r, img2_r)

            # Display metrics
            st.subheader("🔎 Results")
            st.write(f"**Distance:** {dist:.4f}")
            st.write(f"**Similarity (1 - distance):** {similarity:.4f}")
            st.write(f"**SSIM:** {ssim_score:.4f}")

            if label == "SAME":
                st.success("🎉 MATCH: SAME EYE")
            else:
                st.error("❌ DIFFERENT EYES")

            # -----------------------------
            #  Comparison plot (Matplotlib)
            # -----------------------------
            fig, axes = plt.subplots(1, 2, figsize=(10,5))

            axes[0].imshow(seg_new)
            axes[0].set_title("New Image (Sclera Extracted)")
            axes[0].axis('off')

            axes[1].imshow(stored_img)
            axes[1].set_title("Stored Image (Sclera Extracted)")
            axes[1].axis('off')

            fig.suptitle(
                f"Dist: {dist:.4f} | Sim: {similarity:.4f} | {label}",
                fontsize=14,
                color=('green' if label=="SAME" else 'red')
            )

            st.pyplot(fig)

