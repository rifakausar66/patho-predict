from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
# Note: Ensure you have pushed the fix for 'tensorflow==2.15.0' in requirements.txt
model = load_model('models/eye_disease_model.h5')
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]


# ---------------------- FUNCTION TO CHECK IF IMAGE IS EYE ----------------------

def looks_like_eye(img):
    w, h = img.size
    
    # This check is weak, but keeps very small/corrupt files out
    if w < 100 or h < 100:
        return False
    
    return True


# ---------------------- ROUTES ----------------------

@app.route("/")
def login():
    return render_template("login.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)

    path = os.path.join("uploads", file.filename)
    file.save(path)

    # Load and preprocess the image
    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
    except Exception:
        # Handle case where file is not a valid image
        return redirect("/result?disease=Invalid Image File&confidence=0%")


    # Check if uploaded image is eye or not (basic size check)
    if not looks_like_eye(img):
        return redirect("/result?disease=Not an Eye Image&confidence=0%")

    x = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = model.predict(x)
    max_conf = float(np.max(pred))

    # *** IMPORTANT CHANGE: Increased Confidence Threshold ***
    # Model is overconfident on random images. Raising this threshold 
    # forces the model to be extremely sure it's one of the four classes.
    # Set to 95% (0.95) to filter out most random images.
    CONFIDENCE_THRESHOLD = 0.95
    
    if max_conf < CONFIDENCE_THRESHOLD:
        # If confidence is below the high threshold, treat it as an unknown or non-eye image
        disease = "Not a Reliable Prediction (Low Confidence)"
        confidence = round(max_conf * 100, 2)
    else:
        # If confidence is high, use the model's top prediction
        disease = CLASS_NAMES[np.argmax(pred)]
        confidence = round(max_conf * 100, 2)

    return redirect(f"/result?disease={disease}&confidence={confidence}%")


@app.route("/result")
def result():
    return render_template("result.html")


# ---------------------- RUN FLASK ----------------------

if __name__ == "__main__":
    # Allows public access (important for deployment)
    app.run(host="0.0.0.0", port=5000, debug=True)