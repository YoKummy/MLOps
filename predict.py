import os

from ultralytics import YOLO
import time
model = YOLO("C:/Users/1003380/Mine/MLOps/runs/detect/train/weights/best.pt")
a = time.time()

model.predict(source="C:/Users/1003380/Mine/MLOps/catvdog/test/images", imgsz=640, conf=0.40, save=True)
b = time.time()
total = b - a
print("Time taken for prediction:", f"{total:.2f} seconds",
      f"({total/len(os.listdir('C:/Users/1003380/Mine/MLOps/catvdog/test/images')):.5f} seconds per image)")

# model.train(data="catvdog.yaml", epochs=300, imgsz=640, batch=16, name="yolo26s-catvdog", workers=0)
# s: 69.6018 / 1746 = 0.03985 s/img
# m: 105.5772 / 1746 = 0.06045 s/img