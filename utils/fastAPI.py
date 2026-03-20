from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load your model once at startup
model = YOLO("C:/Users/1003380/Mine/MLOps/runs/detect/train/weights/best.pt") # Please change this to your working directory

# Changed 'async def' to standard 'def' 
# This tells FastAPI to run this heavy ML task in a separate thread!
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # 1. Quick validation to ensure it's an image type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # 2. Read image synchronously (using file.file.read() instead of await file.read())
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. Check if OpenCV successfully decoded the image
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image data.")

    # 4. Run YOLO Inference
    results = model.predict(img)
    
    # 5. Format the results as JSON
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist() 
            })
    
    return {"filename": file.filename, "detections": detections}