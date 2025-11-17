from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = load_model("eye_disease_model.h5")
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]


# ---------------------- FUNCTION TO CHECK IF IMAGE IS EYE ----------------------

def looks_like_eye(img):
    w, h = img.size
    
    # Very small or weird images are not eye images
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
    img = Image.open(path).convert("RGB").resize((224, 224))

    # Check if uploaded image is eye or not
    if not looks_like_eye(img):
        return redirect("/result?disease=Not an Eye Image&confidence=0%")

    x = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = model.predict(x)
    max_conf = float(np.max(pred))

    # Confidence threshold (60%) → if below, we say NOT AN EYE IMAGE
    if max_conf < 0.60:
        disease = "Not an Eye Image"
        confidence = round(max_conf * 100, 2)
    else:
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
