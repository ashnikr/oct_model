from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = FastAPI()

# Define class labels
vgg8_classes = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']

# Load ONNX model
session = ort.InferenceSession(r"C:\Users\Antino\OneDrive\Desktop\oct_eye\model\ensemble_model.onnx")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    input_tensor = preprocess(img_bytes)
    outputs = session.run(None, {"input": input_tensor})[0]
    pred_idx = int(np.argmax(outputs))
    confidence = float(outputs[0][pred_idx])
    return JSONResponse({
        "prediction": vgg8_classes[pred_idx],
        "confidence": round(confidence, 4)
    })
