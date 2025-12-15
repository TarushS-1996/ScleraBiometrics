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
SIAMESE_WEIGHTS_PATH = "Model/siamese_model_trained.weights.h5"

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



def compare_processed_eye(
    processed_img: np.ndarray,
    user_id: str,
    eye_side: str,
    base_dir: str,
    threshold: float = 0.6
):
    """
    Compare TWO ALREADY-PROCESSED sclera images.
    NO re-running predict_sclera_and_vessels.
    """

    # -------------------------------------------------
    # Load STORED PROCESSED image
    # -------------------------------------------------
    stored_path = os.path.join(base_dir, f"{user_id}_{eye_side}.png")

    if not os.path.exists(stored_path):
        raise FileNotFoundError(f"Stored processed image not found: {stored_path}")

    stored_img = cv2.imread(stored_path)
    stored_img = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------
    # Enforce SHAPE + TYPE CONSISTENCY
    # -------------------------------------------------
    processed_img = cv2.resize(processed_img, (256, 256))
    stored_img = cv2.resize(stored_img, (256, 256))

    # 🔥 THIS IS THE CRITICAL LINE
    # Match training EXACTLY
    processed_img = processed_img.astype(np.float32)
    stored_img = stored_img.astype(np.float32)

    # ❌ DO NOT divide by 255 if training didn’t
    # processed_img /= 255.0
    # stored_img /= 255.0

    # -------------------------------------------------
    # Siamese inference
    # -------------------------------------------------
    X1 = np.expand_dims(processed_img, axis=0)
    X2 = np.expand_dims(stored_img, axis=0)

    dist = float(siamese_model.predict([X1, X2], verbose=0)[0][0])
    similarity = 1.0 - dist
    label = "SAME" if similarity >= 0.6 else "DIFFERENT"

    return {
        "distance": dist,
        "similarity": similarity,
        "label": label
    }

def identify_processed_eye_across_database(
    processed_query_img: np.ndarray,
    processed_dir: str,
    threshold: float = 0.58
):
    """
    Compare a processed sclera image against ALL stored processed images.
    """
    processed_query_img = cv2.resize(processed_query_img, (256, 256))
    query = processed_query_img.astype(np.float32)
    Xq = np.expand_dims(query, axis=0)

    results = []

    for fname in sorted(os.listdir(processed_dir)):
        if not fname.endswith(".png"):
            continue

        path = os.path.join(processed_dir, fname)

        stored_img = cv2.imread(path)
        stored_img = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)
        stored_img = cv2.resize(stored_img, (256, 256))

        if stored_img.shape != (256, 256, 3):
            continue

        stored = stored_img.astype(np.float32)
        Xs = np.expand_dims(stored, axis=0)

        dist = float(
            siamese_model.predict([Xq, Xs], verbose=0)[0][0]
        )

        similarity = 1.0 - dist

        print(
            f"{fname} → dist={dist:.4f}, sim={similarity:.4f}"
        )

        results.append({
            "name": fname.replace(".png", ""),
            "distance": dist,
            "similarity": similarity,
            "label": "SAME" if similarity >= threshold else "DIFFERENT"
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "threshold": threshold,
        "best_match": results[0] if results else None,
        "matches": [r for r in results if r["label"] == "SAME"],
        "all_results": results
    }
