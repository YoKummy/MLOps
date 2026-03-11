from ultralytics import YOLO
import time
model = YOLO(r"C:\Users\1003380\ultrayolo\kb_V11_l_0649.pt")
a = time.time()

model.predict(source="C:/Users/1003380/ultrayolo/dataset/KB/KBDataset_20250904/images/val", imgsz=1600, conf=0.40, save=True)
b = time.time()
total = b - a
print("Time taken for prediction:", f"{total:.2f} seconds",
      f"({total/1746:.5f} seconds per image)")

# model.train(data="catvdog.yaml", epochs=300, imgsz=640, batch=16, name="yolo26s-catvdog", workers=0)
# s: 69.6018 / 1746 = 0.03985 s/img
# m: 105.5772 / 1746 = 0.06045 s/img