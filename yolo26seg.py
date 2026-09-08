import cv2
import numpy as np
from numpy import dot, sqrt
import time
from copy import copy
from utils import *
from rknnlite.api import RKNNLite

COCO = 0
FIRE = 1
CRACK = 2
LICENSE = 3
GARBAGE = 4
PCB = 5

def hex2rgb(h):  # rgb order (PIL)
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

class Yolo26Seg:
    def __init__(self,
                #yolo_model: str,
                RK3588_RKNN_MODEL: str,
                input_size: str | int | None = None ,
                DATASET = COCO,
                CONF_THRESH = 0.25,
                IOU_THRESH = 0.45,
                ) -> None:
        self.conf_thresh = CONF_THRESH
        self.iou_thresh = IOU_THRESH
        self.input_size = parse_imgsz(input_size) if input_size else detect_imgsz_from_path(RK3588_RKNN_MODEL)
        self.DATASET = DATASET 
        self.colors = Colors()
        self.num_classes = 80
        self.proto_channel = 32
        self.rknn_lite = RKNNLite()

        if self.DATASET == COCO :
            self.CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
                    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                    "oven ", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
        elif self.DATASET == CRACK :
            self.CLASSES = ("crack", "")
        elif self.DATASET == GARBAGE :
            self.CLASSES = ("Aluminium foil", "Battery", "Aluminium blister pack", "Carded blister pack", "Other plastic bottle",
                    "Plastic bottle", "Clear plastic bottle", "Glass bottle", "Plastic bottle cap", "Metal bottle cap", "Broken glass",
                    "Food Can", "Aerosol", "Drink can", "Toilet tube", "Other carton", "Egg carton", "Drink carton", "Corrugated carton",
                    "Meal carton", "Pizza box", "Paper cup", "Disposable plastic cup", "Foam cup", "Glass cup", "Other plastic cup",
                    "Food waste", "Glass jar", "Plastic lid", "Metal lid", "Other plastic", "Magazine paper", "Tissues", "Wrapping paper",
                    "Normal paper", "Paper bag", "Plastified paper bag", "Plastic film", "Six pack rings", "Garbage bag", "Other plastic wrapper",
                    "Single-use carrier bag", "Polypropylene bag", "Crisp packet", "Spread tub", "Tupperware", "Disposable food container",
                    "Foam food container", "Other plastic container", "Plastic glooves", "Plastic utensils", "Pop tab", "Rope & strings",
                    "Scrap metal", "Shoe", "Squeezable tube", "Plastic straw", "Paper straw", "Styrofoam piece", "Unlabeled litter", "Cigarette")

        #print('--> Load YOLO model')
        print_info(f'--> Load YOLO26 model')
        ret = self.rknn_lite.load_rknn(RK3588_RKNN_MODEL)
        if ret != 0:
            #print('Load RKNN model failed')
            print_info(f'Load RKNN model failed')
            exit(ret)
        print('done')

        #print('--> Init runtime environment YOLO')
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
        img_letterbox = self._letterbox(image)
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img_in = np.expand_dims(img_rgb, axis=0)
        
        return img_in

    def _postprocess(self, outputs: list, orig_shape: tuple = None) -> list[dict]:
        proto = outputs[6]
        if proto.ndim == 4:
            proto = proto[0]

        if proto.shape[-1] == self.proto_channel:
            proto = np.transpose(proto, (2, 0, 1))
        
        proto_h, proto_w = proto.shape[1], proto.shape[2]

        head_outputs = [
            (outputs[0], outputs[1]),  # Stride 8  (80x80)
            (outputs[2], outputs[3]),  # Stride 16 (40x40)
            (outputs[4], outputs[5])   # Stride 32 (20x20)
        ]

        filter_boxes = []
        obj_probs = []
        class_ids = []
        filter_segments = []

        for det_out, seg_out in head_outputs:
            det = det_out[0]
            seg = seg_out[0]

            if det.shape[-1] != det.shape[0] and det.ndim == 3:
                if det.shape[0] != 84 and det.shape[-1] == 84:
                    det = np.transpose(det, (2, 0, 1))
                if seg.shape[0] != 32 and seg.shape[-1] == 32:
                    seg = np.transpose(seg, (2, 0, 1))

            grid_h, grid_w = det.shape[1], det.shape[2]
            stride = self.input_size[1] // grid_h

            for i in range(grid_h):
                for j in range(grid_w):
                    cls_scores = det[4:4 + self.num_classes, i, j]
                    max_cls_id = int(np.argmax(cls_scores))
                    box_conf = float(cls_scores[max_cls_id])

                    if box_conf >= self.conf_thresh:
                        loc = det[:4, i, j]
                        x1 = (j + 0.5 - loc[0]) * stride
                        y1 = (i + 0.5 - loc[1]) * stride
                        w_ = (loc[0] + loc[2]) * stride
                        h_ = (loc[1] + loc[3]) * stride

                        filter_boxes.append([x1, y1, w_, h_])
                        obj_probs.append(box_conf)
                        class_ids.append(max_cls_id)
                        filter_segments.append(seg[:, i, j])

        if not filter_boxes:
            return []

        boxes_for_nms = [[b[0], b[1], b[2], b[3]] for b in filter_boxes]
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, obj_probs, self.conf_thresh, self.iou_thresh)
        if len(indices) == 0:
            return []

        indices = indices.flatten()
        boxes_num = len(indices)

        selected_segments = []
        crop_boxes = []
        nms_boxes = []
        nms_scores = []
        nms_class_ids = []

        for idx in indices:
            bx, by, bw, bh = filter_boxes[idx]
            x1, y1 = bx, by
            x2, y2 = bx + bw, by + bh

            crop_boxes.append([x1, y1, x2, y2])

            real_x1 = int(x1 / self.ratio)
            real_y1 = int(y1 / self.ratio)
            real_x2 = int(x2 / self.ratio)
            real_y2 = int(y2 / self.ratio)

            nms_boxes.append([real_x1, real_y1, real_x2, real_y2])
            nms_scores.append(obj_probs[idx])
            nms_class_ids.append(class_ids[idx])
            selected_segments.append(filter_segments[idx])

        selected_segments = np.array(selected_segments)
        proto_flat = proto.reshape(self.proto_channel, -1)
        matmul_out = np.dot(selected_segments, proto_flat).reshape(boxes_num, proto_h, proto_w)

        if orig_shape is not None:
            orig_h, orig_w = orig_shape
            cropped_w = int(round(orig_w * self.ratio))
            cropped_h = int(round(orig_h * self.ratio))
        else:
            max_x2 = max([b[2] for b in crop_boxes]) if crop_boxes else self.input_size[0]
            max_y2 = max([b[3] for b in crop_boxes]) if crop_boxes else self.input_size[1]
            cropped_w = int(min(self.input_size[0], max_x2))
            cropped_h = int(min(self.input_size[1], max_y2))
            orig_w = int(cropped_w / self.ratio)
            orig_h = int(cropped_h / self.ratio)

        results = []
        for b in range(boxes_num):
            mask_full = cv2.resize(matmul_out[b], (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_LINEAR)

            cx1, cy1, cx2, cy2 = [int(v) for v in crop_boxes[b]]
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(self.input_size[0], cx2), min(self.input_size[1], cy2)

            cropped_inst_mask = np.zeros((self.input_size[0], self.input_size[1]), dtype=bool)
            if cx2 > cx1 and cy2 > cy1:
                cropped_inst_mask[cy1:cy2, cx1:cx2] = mask_full[cy1:cy2, cx1:cx2] > 0

            valid_crop_mask = cropped_inst_mask[:cropped_h, :cropped_w].astype(np.uint8)
            real_mask = cv2.resize(valid_crop_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)

            real_box = [
                int(np.clip(nms_boxes[b][0], 0, orig_w)),
                int(np.clip(nms_boxes[b][1], 0, orig_h)),
                int(np.clip(nms_boxes[b][2], 0, orig_w)),
                int(np.clip(nms_boxes[b][3], 0, orig_h))
            ]

            results.append({
                'box': real_box,
                'score': nms_scores[b],
                'class_id': nms_class_ids[b],
                'mask': real_mask
            })

        return results

    def infer(self, image: np.ndarray) -> list[dict]:
        img_in = self._preprocess(image)
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[img_in])
        self.infertime = (time.time() - start_time)*1000
        print_info(f'Inference time: {self.infertime} ms')
        return self._postprocess(outputs, orig_shape=image.shape[:2])

    def draw(self, results: list[dict]):
        img_h, img_w = self.img_result.shape[:2]

        for res in results:
            box = res['box']
            score = res['score']
            cid = res['class_id']
            mask = res['mask']

            # Colors 팔레트에서 BGR 색상 가져오기
            color = self.colors(cid, bgr=True)

            if mask is not None and np.any(mask):
                if mask.shape[0] != img_h or mask.shape[1] != img_w:
                    mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)

                colored_mask = np.zeros_like(self.img_result, dtype=np.uint8)
                colored_mask[mask] = color
                overlay = cv2.addWeighted(self.img_result, 0.5, colored_mask, 0.5, 0)
                self.img_result[mask] = overlay[mask]

            x1, y1, x2, y2 = box
            cv2.rectangle(self.img_result, (x1, y1), (x2, y2), color, 2)
            label = f"{self.CLASSES[cid]}: {score:.2f}"
            cv2.putText(self.img_result, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return self.img_result

    def info(self):
            print_info(f'Inference time: {self.infertime} ms')

    def release(self):
        self.rknn_lite.release()