import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = FastAPI()
model = load_model("Finger_signs.keras")

classes = {
    0: "Fist",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five"
}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = image.resize((64,64))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "class_id": class_id,
        "class_name": classes[class_id],
        "confidence": confidence
    }