from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")

model = YOLO("yolo11l.pt")
# model.train(data="kb.yaml", epochs=250, imgsz=1600, cfg='evo.yaml',
#             device=[0], project="KB_EVO", name="KB_evo", workers=39, batch=8,
#             optimizer='SGD', save=True, auto_augment=None, erasing=0.0,
#             cos_lr=True, plots=True, val=True)
# model.train(data="kb.yaml", epochs=250, imgsz=1600, cls=1.0, copy_paste=0.4,
            # optimizer='SGD', close_mosaic=20, patience=0, dropout=0.1,
            # batch=8, name="kb_1600p_l", mixup=0.05, lr0=0.01, cos_lr=True,
            # workers=39, device=[0], project='KB_highres_l', auto_augment=None,
            # hsv_h=0.01, hsv_s=0.0, hsv_v=0.1, scale=0.5, erasing=0.0, degrees=10.0,)


model.train(data="catvdog.yaml", epochs=100, imgsz=640, 
            device=[0], workers=0, batch=4, optimizer='SGD', 
            save=True, cos_lr=True, plots=True, val=True, exist_ok=True)

# model = YOLO("yolo26l.pt")
# model.tune(data="kb.yaml", epochs=50, imgsz=1600, optimizer='SGD',
#            close_mosaic=10, batch=8, name="kb_1600p_evo_extraLong", plots=True,
#            save=False, val=True, device=[1], iterations=100, workers=39)

# model.train(data="tcl.yaml", epochs=500, imgsz=1024,
#             optimizer='SGD', close_mosaic=450,
#             batch=64, name="tcl_s",
#             workers=39, device=[0], project='TCL')


# model = YOLO("runs/detect/lgp4/weights/best.pt",)
# model.train(cfg='cfg/mosaic.yaml', data="kb.yaml", epochs=500, imgsz=1000,
#             batch=48, name="kb_Mosaic_SGD_s",
#             workers=39, device=[0], project='Mosaic_SGD')
# model.val(
#     plots=True,
#     data="lgp.yaml",
# )
