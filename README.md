# 👁️ Sclera Vein Biometric Verification
### *Deep Learning–based Siamese Network for Eye Identity Matching*

This project implements a **sclera-based biometric identity verification system** using:

- A **U-Net–based sclera/iris segmentation model**
- A **custom Siamese network** trained with contrastive loss
- A **vessel-enhanced preprocessing pipeline** (Frangi filter + skeletonization)
- A **Streamlit web application** for:
  - Adding new users
  - Extracting and storing sclera patterns
  - Verifying identity using similarity scoring

The system extracts fine-grained scleral vein patterns and uses a Siamese model to determine whether two samples belong to the same person.

---

## 🚀 Features

- **Sclera segmentation** (U-Net at 128×128 resolution)
- **Vein enhancement pipeline** using:
  - CLAHE
  - Frangi vesselness
  - Skeletonization
- **Siamese CNN** with L2 embeddings for comparison
- **Distance + similarity-based classification**
- **Streamlit interface** with:
  - “Add User” tab
  - “Verify User” tab
- **Local folder database** (no external DB needed)

---

## 🧠 Technologies

| Component         | Technology              |
|-------------------|-------------------------|
| Deep Learning     | TensorFlow / Keras      |
| Segmentation      | U-Net                   |
| Feature Extraction| Frangi filter (scikit-image), CLAHE |
| Similarity Model  | Siamese Network (contrastive loss) |
| Frontend          | Streamlit               |
| Image Processing  | OpenCV                  |

---

# 🛠️ Setup Instructions

Follow these steps to run the project locally.

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python3 -m venv tf_env
source tf_env/bin/activate
```

### Windows

```bash
python -m venv tf_env
tf_env\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install streamlit tensorflow opencv-python scikit-image matplotlib numpy tqdm
```

## 4. Add Model Weights

Place the following files inside the `Model/` directory:

```plaintext
Model/sclera_iris_segmentation_model.h5
Model/siamese_model_trained.weights.h5
```

## 5. Launch Streamlit UI

```bash
streamlit run ui.py
```

Then open the local URL (usually http://localhost:8501).

