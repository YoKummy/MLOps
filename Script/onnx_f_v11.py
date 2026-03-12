import os
import cv2
import numpy as np
import onnxruntime as ort
from glob import glob


class YOLO26ONNX:
    def __init__(
        self,
        model_path: str,
        img_size: int = 1600,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres 

        # Force CPU to avoid CUDA dependency issues
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name


    def preprocess(self, image):
        h, w = image.shape[:2]

        scale = self.img_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (nw, nh))

        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        top = (self.img_size - nh) // 2
        left = (self.img_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized

        img = canvas[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, scale, left, top

    def postprocess(self, preds, scale, left, top):
        preds = preds[preds[:, 4] > self.conf_thres]
        if len(preds) == 0:
            return []

        boxes = preds[:, :4]
        scores = preds[:, 4]
        classes = preds[:, 5].astype(int)

        # Undo letterbox
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= scale

        # indices = cv2.dnn.NMSBoxes(
            # boxes.tolist(),
            # scores.tolist(),
            # self.conf_thres,
            # self.iou_thres,
        # )
        def fast_nms(boxes, scores, iou_thres):
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort()[::-1]

            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)

                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= iou_thres)[0]
                order = order[inds + 1]

            return keep
        indices = fast_nms(boxes, scores, self.iou_thres)

        results = []
        if len(indices) > 0:
            for i in indices:
                results.append(
                    {
                        "bbox": boxes[i].tolist(),
                        "score": float(scores[i]),
                        "class": int(classes[i]),
                    }
                )
        return results

    def infer(self, image):
        img, scale, left, top = self.preprocess(image)

        outputs = self.session.run(None, {self.input_name: img})
        preds = outputs[0]

        # YOLO11 ONNX output is usually (1, 84, N)
        preds = np.squeeze(preds)

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # convert to (N, 84)

        boxes = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences > self.conf_thres

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert center format → corner format
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        preds_final = np.column_stack([boxes, confidences, class_ids])

        return self.postprocess(preds_final, scale, left, top)


def draw_results(image, results):
    for r in results:
        x1, y1, x2, y2 = map(int, r["bbox"])
        cls = r["class"]
        conf = r["score"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{cls}:{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return image


if __name__ == "__main__":
    model_path = "kb_v11_l_0649.onnx"
    input_path = r"C:\Users\1003380\ultrayolo\dataset\KB\KBDataset_20250904\images\val"   # can be folder or single image
    output_folder = "v11_output_images"

    os.makedirs(output_folder, exist_ok=True)

    detector = YOLO26ONNX(model_path)

    # If single image
    if os.path.isfile(input_path):
        img = cv2.imread(input_path)
        results = detector.infer(img)
        annotated = draw_results(img.copy(), results)

        save_path = os.path.join(output_folder, os.path.basename(input_path))
        cv2.imwrite(save_path, annotated)
        print("Saved:", save_path)

    # If folder
    else:
        image_paths = glob(os.path.join(input_path, "*.*"))

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            results = detector.infer(img)
            annotated = draw_results(img.copy(), results)

            save_path = os.path.join(output_folder, os.path.basename(path))
            cv2.imwrite(save_path, annotated)

            print("Processed:", path)

    print("Done.")