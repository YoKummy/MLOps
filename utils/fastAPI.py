from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Load your model once at startup
model = YOLO("yolo11n.pt") # or your custom best.pt

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image from the upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run YOLO Inference
    results = model.predict(img)
    
    # 3. Format the results as JSON
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })
    
    return {"filename": file.filename, "detections": detections}