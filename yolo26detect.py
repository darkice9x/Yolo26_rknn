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

class Yolo26Detect(object):
    def __init__(self,
                 #yolo_model: str,
                 RK3588_RKNN_MODEL: str,
                 input_size: str | int | None = None ,
                 DATASET = COCO,
                 OBJ_THRESH = 0.25
                 ) -> None:
        #self.yolo_model = yolo_model
        self.rknn_lite = RKNNLite()
        self.input_size = parse_imgsz(input_size) if input_size else detect_imgsz_from_path(RK3588_RKNN_MODEL)
        self.OBJ_THRESH = OBJ_THRESH
        self.DATASET = DATASET
        self.infertime = 0
        #self.logger = logging.getLogger("YOLO")
        #self.logger.setLevel(logging.DEBUG)
        #logging.basicConfig(format='%(name)s : %(message)s', level=logging.DEBUG)

        if self.DATASET == COCO :
            self.CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
                    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                    "oven ", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
        elif self.DATASET == FIRE :
            self.CLASSES = ("fire", "", "smoke")
        elif self.DATASET == CRACK :
            self.CLASSES = ("crack", "")
        elif self.DATASET == LICENSE :
            self.CLASSES = ("license", "")
        elif self.DATASET == PCB :
            self.CLASSES = ("missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper")

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
        objects = self.detect(self.img_org)
        
        #return boxes, classes, scores
        return objects

    def _letterbox(self, im, color=(0, 0, 0)):
        shape = im.shape[:2]
        self.ratio  = min(self.input_size[1] / shape[0], self.input_size[0]/ shape[1])
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
    
    def _postprocess(self, input_data):
        """
        后处理 - 三尺度输出解码
        
        参数:
            outputs: (1, 84, 80, 80), (1, 84, 40, 40), (1, 84, 20, 20)
        
        返回:
            boxes: (N, 4) - [x1, y1, x2, y2] 归一化到原图
            scores: (N,) - 置信度
            classes: (N,) - 类别索引
        """

        all_boxes, all_scores, all_classes = [], [], []
        outputs = [input_data[0], input_data[1], input_data[2]]
        # strides for 3 scales
        strides = [8, 16, 32]
        
        for i, output in enumerate(outputs):
            # 번호판의 경우 5로 나온다 그래서 5만 꺼내서 사용
            # output shape: (1, x, h, w) -> (x, h*w) output[0].shape[0] = x
            pred = output[0].reshape(output[0].shape[0], -1)
            
            h, w = output.shape[2], output.shape[3]
            stride = strides[i]
            
            # anchor_points
            y = np.arange(h) * stride + stride // 2
            x = np.arange(w) * stride + stride // 2
            xx, yy = np.meshgrid(x, y)
            anchor_points = np.stack([xx.ravel(), yy.ravel()], axis=0)  # (2, N)
            
            #box cls_scores
            box_dist = pred[:4, :]  # (4, N)
            cls_scores = pred[4:, :]  # (80, N)
            
            # dist2bbox
            x1y1 = anchor_points - box_dist[:2, :] * stride
            x2y2 = anchor_points + box_dist[2:, :] * stride
            boxes = np.concatenate([x1y1, x2y2], axis=0)  # (4, N)
            
            # max_cls_scores
            max_cls_scores = cls_scores.max(axis=0)  # (N,)
            
            mask = max_cls_scores > self.OBJ_THRESH
            if not mask.any():
                continue
            
            # classes
            classes = cls_scores.argmax(axis=0)

            all_boxes.append(boxes[:, mask])
            all_scores.append(max_cls_scores[mask])
            all_classes.append(classes[mask])
        
        if not all_boxes:
            return np.empty((0, 4)), np.empty(0), np.empty(0)
        
        boxes = np.concatenate(all_boxes, axis=1).T  # (N, 4)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)
        boxes = (boxes / self.ratio).astype(np.int32)

        # [수정] 리스트 형태의 딕셔너리 구조로 재구성
        objects = [
            {
                "box": box.tolist(),         # numpy array -> python list [x1, y1, x2, y2]
                "score": float(score),       # float 형변환 (JSON 직렬화 시 용이)
                "class": int(cls)            # int 형변환
            }
            for box, score, cls in zip(boxes, scores, classes)
        ]
        #return self.boxes, self.scores, self.classes
        return objects

    def draw(self, objects):
            """Draw the boxes on the image.
    
            # Argument:
                image: original image.
                boxes: ndarray, boxes of objects.
                classes: ndarray, classes of objects.
                scores: ndarray, scores of objects.
                all_classes: all classes name.
            """
            #print(len(objects))
            for obj in objects:
                #print(f"Class: {obj['class']}, Score: {obj['score']:.2f}, Box: {obj['box']}")
                left, top, right, bottom = obj['box']
                cv2.rectangle(self.img_result, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(self.img_result, '{0} {1:.2f}'.format(self.CLASSES[obj['class']], obj['score']),
                            (left, top - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                #print("{:^12} {:^12.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(self.CLASSES[obj['class']], obj['score'], top, left, right, bottom))

    def detect(self, img_org):
        #letterbox_img, self.ratio, ( self.dw, self.dh ) = self.letterbox(img_org)  # letterbox缩放
        infer_img = self._preprocess(img_org)
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[infer_img])
        self.infertime = (time.time() - start_time)*1000
        input_data = [outputs[0], outputs[1], outputs[2]]
        objects = self._postprocess(outputs)
        print_info(f'Inference time: {self.infertime} ms')
        #return boxes, classes, scores
        return objects
    
    def info(self):
        print_info(f'Inference time: {self.infertime} ms')
        
    def release(self):
        self.rknn_lite.release()
        print_info(f'RKNN Detect Release!!')
                            
