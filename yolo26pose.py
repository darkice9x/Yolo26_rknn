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

# COCO 17개 관절 연결 정보 (어깨, 팔꿈치, 무릎 등 연결선)
SKELETON =  [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
             [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], 
             [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
             [2, 4], [3, 5], [4, 6], [5, 7]]

# 관절 및 뼈대 색상 (BGR)
POINT_COLOR = (0, 255, 0)   # 초록색
LINE_COLOR = (255, 0, 0)    # 파란색
BOX_COLOR = (0, 165, 255)   # 주황색

class Yolo26Pose(object):
    def __init__(self,
                 RK3588_RKNN_MODEL: str,
                 input_size: str | int | None = None ,
                 OBJ_THRESH=0.5,
                 calc_angle = True
                 ) -> None:

        self.rknn_lite = RKNNLite()
        self.input_size = parse_imgsz(input_size) if input_size else detect_imgsz_from_path(RK3588_RKNN_MODEL)
        self.objectThresh = OBJ_THRESH
        self.calc_angle = calc_angle
        self.infertime = 0
        self.left_knee_angle = 0 
        self.right_knee_angle = 0
        self.left_elbow_angle = 0
        self.right_elbow_angle = 0
        self.left_knee_list = []
        self.left_knee_list_sum = []
        self.right_knee_list = []
        self.right_knee_list_sum = []
        self.left_elbow_list = []
        self.left_elbow_list_sum = []
        self.right_elbow_list = []
        self.right_elbow_list_sum = []
        self.sum_len = 0
        #self.logger = logging.getLogger("YOLO")
        #logging.basicConfig(format='%(name)s : %(message)s', level=logging.DEBUG)

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
        input_data = np.expand_dims(input_img, axis=0)  # (1, 320, 320, 3)
        
        return input_data

    def infer(self, img_path):
        # 2. 이미지 전처리
        input_data = self._preprocess(img_path)
        
        # 3. NPU 추론 실행
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[input_data])
        self.infertime = (time.time() - start_time)*1000
        print_info(f'Inference time: {self.infertime} ms')
        
        # 4. 후처리 진행 (dw, dh 전달 없음)
        objects = self._postprocess(outputs)

        for obj in objects:
            kpts = obj["keypoints"]  # [[x, y, conf], ...] (17개)
            if self.calc_angle :
                # 12, 14, 16
                #self.left_knee_angle = self._calculateAngle((int( kpts[12][0]), int(kpts[12][1])), (int( kpts[14][0]), int(kpts[14][1])), (int( kpts[16][0]), int(kpts[16][1])))
                self.left_knee_angle = self._calculateAngle((kpts[12][0], kpts[12][1]), (kpts[14][0], kpts[14][1]), (kpts[16][0], kpts[16][1]))
                self.left_knee_list_sum.append(self.left_knee_angle)
                self.left_knee_list_sum = self.left_knee_list_sum[-10:]
                self.sum_len = len( self.left_knee_list_sum  )
                self.left_knee_angle = int(sum(self.left_knee_list_sum)/self.sum_len)
                self.left_knee_list.append(self.left_knee_angle)
                self.left_knee_list = self.left_knee_list[-50:]
                # 11, 13, 15
                #self.right_knee_angle = self._calculateAngle((int( kpts[11][0]), int(kpts[11][1])), (int( kpts[13][0]), int(kpts[13][1])), (int( kpts[15][0]), int(kpts[15][1])))
                self.right_knee_angle = self._calculateAngle((kpts[11][0], kpts[11][1]), (kpts[13][0], kpts[13][1]), (kpts[15][0], kpts[15][1]))
                self.right_knee_list_sum.append(self.right_knee_angle)
                self.right_knee_list_sum = self.right_knee_list_sum[-10:]
                self.sum_len = len( self.right_knee_list_sum  )
                self.right_knee_angle = int(sum(self.right_knee_list_sum)/self.sum_len)
                self.right_knee_list.append(self.right_knee_angle)
                self.right_knee_list = self.right_knee_list[-50:]
                # 6, 8, 10
                #self.left_elbow_angle = self._calculateAngle((int( kpts[6][0]), int(kpts[6][1])), (int( kpts[8][0]), int(kpts[8][1])), (int( kpts[10][0]), int(kpts[10][1])))
                self.left_elbow_angle = self._calculateAngle((kpts[6][0], kpts[6][1]), (kpts[8][0], kpts[8][1]), (kpts[10][0], kpts[10][1]))
                self.left_elbow_list_sum.append(self.left_elbow_angle)
                self.left_elbow_list_sum = self.left_elbow_list_sum[-10:]
                self.sum_len = len( self.left_elbow_list_sum  )
                self.left_elbow_angle = int(sum(self.left_elbow_list_sum)/self.sum_len)
                self.left_elbow_list.append(self.left_elbow_angle)
                self.left_elbow_list = self.left_elbow_list[-50:]
                # 5, 7, 9
                #self.right_elbow_angle = self._calculateAngle((int( kpts[5][0]), int(kpts[5][1])), (int( kpts[7][0]), int(kpts[7][1])), (int( kpts[9][0]), int(kpts[9][1])))
                self.right_elbow_angle = self._calculateAngle((kpts[5][0], kpts[5][1]), (kpts[7][0], kpts[7][1]), (kpts[9][0], kpts[9][1]))
                self.right_elbow_list_sum.append(self.right_elbow_angle)
                self.right_elbow_list_sum = self.right_elbow_list_sum[-10:]
                self.sum_len = len( self.right_elbow_list_sum  )
                self.right_elbow_angle = int(sum(self.right_elbow_list_sum)/self.sum_len)
                self.right_elbow_list.append(self.right_elbow_angle)
                self.right_elbow_list = self.right_elbow_list[-50:]

        return objects

    def _postprocess(self, outputs):
        """후처리: ratio만으로 좌표 복원"""
        box_outputs = outputs[:3]
        kpt_output = outputs[3][0]       # (17, 3, 2100)

        src_h, src_w = self.img_org.shape[:2]
        strides = [8, 16, 32]
        all_boxes, all_scores, kpt_indices = [], [], []

        anchor_count = 0
        for i, output in enumerate(box_outputs):
            pred = output[0].reshape(5, -1)  # (5, H*W)
            h, w = output.shape[2], output.shape[3]
            stride = strides[i]

            # Anchor Point 생성
            y = np.arange(h) * stride + stride // 2
            x = np.arange(w) * stride + stride // 2
            xx, yy = np.meshgrid(x, y)
            anchor_points = np.stack([xx.ravel(), yy.ravel()], axis=0)

            box_dist = pred[:4, :]
            cls_scores = pred[4, :]

            # BBox 디코딩
            x1y1 = anchor_points - box_dist[:2, :] * stride
            x2y2 = anchor_points + box_dist[2:, :] * stride
            boxes = np.concatenate([x1y1, x2y2], axis=0)

            mask = cls_scores > self.objectThresh
            if mask.any():
                all_boxes.append(boxes[:, mask])
                all_scores.append(cls_scores[mask])
                
                valid_indices = np.where(mask)[0] + anchor_count
                kpt_indices.append(valid_indices)

            anchor_count += (h * w)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=1).T
        scores = np.concatenate(all_scores)
        kpt_idx = np.concatenate(kpt_indices)

        objects = []
        for i in range(len(scores)):
            # ratio 연산만으로 BBox 좌표 복원
            b = boxes[i]
            x1 = np.clip(b[0] / self.ratio, 0, src_w).astype(int)
            y1 = np.clip(b[1] / self.ratio, 0, src_h).astype(int)
            x2 = np.clip(b[2] / self.ratio, 0, src_w).astype(int)
            y2 = np.clip(b[3] / self.ratio, 0, src_h).astype(int)

            # ratio 연산만으로 Keypoints 좌표 복원
            kpts = kpt_output[:, :, kpt_idx[i]].copy()
            kpts[:, 0] = np.clip(kpts[:, 0] / self.ratio, 0, src_w).astype(int)
            kpts[:, 1] = np.clip(kpts[:, 1] / self.ratio, 0, src_h).astype(int)

            objects.append({
                "box": [x1, y1, x2, y2],
                "score": float(scores[i]),
                "keypoints": kpts.tolist()
            })

        return objects
    
    def draw(self, objects, kpt_thresh=0.3):
        """
        image: cv2.imread()로 읽은 원본 이미지 (수정본 반환)
        objects: infer() 결과 리스트
        kpt_thresh: 표시할 관절의 신뢰도(Confidence) 임계값
        """
        for obj in objects:
            # 1. Bounding Box 그리기
            x1, y1, x2, y2 = obj["box"]
            cv2.rectangle(self.img_result, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.putText(self.img_result, f"{obj['score']:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)

            keypoints = obj["keypoints"]  # [[x, y, conf], ...] (17개)

            # 2. 관절 간 뼈대(Skeleton) 그리기
            for p1_idx, p2_idx in SKELETON:
                # 1-indexed -> 0-indexed 변환
                kpt1 = keypoints[p1_idx - 1]
                kpt2 = keypoints[p2_idx - 1]

                x1_kpt, y1_kpt, conf1 = kpt1
                x2_kpt, y2_kpt, conf2 = kpt2

                # 두 관절 모두 신뢰도가 임계값 이상일 때만 선 연결
                if conf1 > kpt_thresh and conf2 > kpt_thresh:
                    pt1 = (int(x1_kpt), int(y1_kpt))
                    pt2 = (int(x2_kpt), int(y2_kpt))
                    cv2.line(self.img_result, pt1, pt2, LINE_COLOR, 2)

            # 3. 관절 점(Keypoints) 그리기
            for kpt in keypoints:
                x_kpt, y_kpt, conf = kpt
                if conf > kpt_thresh:
                    cv2.circle(self.img_result, (int(x_kpt), int(y_kpt)), 4, POINT_COLOR, -1)

        return self.img_result

    def _calculateAngle(self, landmark1, landmark2, landmark3):
        '''
        This function calculates angle between three different landmarks.
        Args:
            landmark1: The first landmark containing the x,y and z coordinates.
            landmark2: The second landmark containing the x,y and z coordinates.
            landmark3: The third landmark containing the x,y and z coordinates.
        Returns:
            angle: The calculated angle between the three landmarks.

        '''

        # Get the required landmarks coordinates.
        x1, y1 = landmark1
        x2, y2 = landmark2
        x3, y3 = landmark3

        # Calculate the angle between the three points
        angle = abs( math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)))

        # Check if the angle is less than zero.
        if angle > 180.0:

            # Add 360 to the found angle.
            angle = 360 - angle
        
        # Return the calculated angle.
        return angle

    def info(self):
            print_info(f'Inference time: {self.infertime} ms')

    def release(self):
        self.rknn_lite.release()