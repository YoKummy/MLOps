from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

model.export(format='onnx')
# import onnx

# onnx_model = onnx.load("kb_v11_l_0649.onnx")
# print("Opset version:", onnx_model.opset_import[0].version)