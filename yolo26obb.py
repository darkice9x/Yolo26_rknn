import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import cv2,math
import copy
from math import ceil
import logging
from copy import copy
from utils import *
from rknnlite.api import RKNNLite

class Yolo26OBB:
    def __init__(self, 
                 RK3588_RKNN_MODEL: str,
                 input_size: str | int | None = None ,
                 score_thresh=0.45, 
                 nms_thresh=0.25
                 ) -> None:
        self.input_size = parse_imgsz(input_size) if input_size else detect_imgsz_from_path(RK3588_RKNN_MODEL)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.infertime = 0

        self.rknn_lite = RKNNLite()

        # load RKNN model
        #print('--> Load RKNN model')
        print_info(f'--> Load YOLO26 model')
        ret = self.rknn_lite.load_rknn(RK3588_RKNN_MODEL)
        if ret != 0:
            #print('Load RKNN model failed')
            print_info(f'Load RKNN model failed')
            exit(ret)
        print('done')

        # init runtime environment
        #print('--> Init runtime environment')
        print_info(f'--> Init runtime environment YOLO26')
        # run on RK356x/RK3588 with Debian OS, do not need specify target.

        ret = self.rknn_lite.init_runtime()

        if ret != 0:
            #print('Init runtime environment failed')
            print_info(f'Init runtime environment failed')
            exit(ret)
        print('done')

    def __call__(self, input, box_vis=False, angle=False):
        """
        Call the detect method to perform inference on the input image.
        :param input: Input image, which can be a NumPy array or file path.
        :return: Processed image with detected keypoints and bounding boxes.
        """
        if isinstance(input, str):
            self.img_org = cv2.imread(input)
        else:
            self.img_org = input

        self.img_result = self.img_org.copy()

        #boxes, classes, scores = self.detect(self.img_org)
        objects = self.infer(self.img_org)
        
        #return boxes, classes, scores
        return objects
    
    def _letterbox(self, im, color=(0, 0, 0)):
        shape = im.shape[:2]
        self.ratio = min(self.input_size[1] / shape[0], self.input_size[0]/ shape[1])
        new_unpad = int(round(shape[1] * self.ratio)), int(round(shape[0] * self.ratio))

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 양쪽 분할 대신 오른쪽에만/아래에만 패딩을 넣음 (top=0, left=0)
        bottom = self.input_size[1] - new_unpad[1]
        right = self.input_size[0]- new_unpad[0]
        im = cv2.copyMakeBorder(im, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=color)

        return im

    def _preprocess(self, image: np.ndarray):
        input_img = self._letterbox(image)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_img, axis=0)

        return input_data

    def _decode_outputs(self, outputs):
        feat_outputs = outputs[:3]
        angle_tensor = outputs[3].squeeze()  # (8400,)

        strides = [8, 16, 32]
        raw_detections = []

        global_box_idx = 0

        for idx, feat in enumerate(feat_outputs):
            stride = strides[idx]
            if feat.ndim == 4:
                feat = feat[0]  # (19, H, W)

            channels, height, width = feat.shape

            cls_probs = feat[4:, :, :]  # (15, H, W)
            max_cls_scores = np.max(cls_probs, axis=0)  # (H, W)
            max_cls_ids = np.argmax(cls_probs, axis=0)    # (H, W)

            mask = max_cls_scores > self.score_thresh
            ys, xs = np.where(mask)

            box_preds = feat[:4, :, :]  # (4, H, W)

            for y, x in zip(ys, xs):
                score = max_cls_scores[y, x]
                cls_id = max_cls_ids[y, x]

                cx_model = (x + 0.5) * stride
                cy_model = (y + 0.5) * stride

                b_raw = box_preds[:, y, x]
                l, t, r, b = b_raw[0], b_raw[1], b_raw[2], b_raw[3]

                w_model = (l + r) * stride
                h_model = (t + b) * stride

                if w_model <= 0 or h_model <= 0 or w_model > self.input_size[0] * 2:
                    w_model = np.exp(np.clip(r, -10, 10)) * stride
                    h_model = np.exp(np.clip(b, -10, 10)) * stride

                cx_orig = cx_model / self.ratio
                cy_orig = cy_model / self.ratio
                w_orig = w_model / self.ratio
                h_orig = h_model / self.ratio

                flat_offset = y * width + x
                current_idx = global_box_idx + flat_offset
                angle_val = angle_tensor[current_idx]

                angle_deg = np.degrees(angle_val) if abs(angle_val) <= np.pi else angle_val

                raw_detections.append({
                    "box": ((float(cx_orig), float(cy_orig)), (float(w_orig), float(h_orig)), float(angle_deg)),
                    "score": float(score),
                    "class_id": int(cls_id)
                })

            global_box_idx += height * width

        if not raw_detections:
            return []

        r_boxes = [item["box"] for item in raw_detections]
        scores = [item["score"] for item in raw_detections]

        indices = cv2.dnn.NMSBoxesRotated(
            r_boxes, scores, score_threshold=self.score_thresh, nms_threshold=self.nms_thresh
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        return [raw_detections[idx] for idx in indices]

    def infer(self, image_path):
        # 전처리 메서드 호출
        input_data = self._preprocess(image_path)
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[input_data])
        self.infertime = (time.time() - start_time)*1000

        results = self._decode_outputs(
            outputs
        )
        return results

    def draw(self, results):
        for det in results:
            box = det["box"]
            score = det["score"]
            class_id = det["class_id"]

            pts = cv2.boxPoints(box)
            pts = np.int32(pts)

            cv2.polylines(
                self.img_result, [pts], isClosed=True, color=(0, 255, 0), thickness=2
            )

            label = f"ID:{class_id} {score:.2f}"
            cv2.putText(
                self.img_result,
                label,
                (int(pts[0][0]), int(pts[0][1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        #cv2.imwrite(output_path, self.img_result)
        #print(f"추론 완료: {output_path} 저장됨")

    def info(self):
        print_info(f'Inference time: {self.infertime} ms')

    def release(self):
            self.rknn_lite.release()