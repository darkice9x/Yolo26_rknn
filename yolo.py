from utils import *
from yolo26seg import *
from yolo26detect import *
from yolo26pose import *    
from yolo26obb import *
from yolo26depth import *


COCO = 0
FIRE = 1
CRACK = 2
LICENSE = 3
GARBAGE = 4
PCB = 5

class Yolo(object):
    def __init__(self,
                 #yolo_model: str,
                 TASK: str,
                 RK3588_RKNN_MODEL: str,
                 DATASET = COCO,
                 S_IOU_THRESH = 0.45,
                 DS_OBJ_THRESH = 0.25,
                 O_NMS_THRESH = 0.25,
                 O_SCORE_THRESH = 0.45,
                 P_OBJ_THRESH = 0.5,
                 CALC_ANGLE = False
                 ) -> None:
        self.task = TASK
        self.detect = None
        self.pose = None
        self.seg = None
        self.obb = None
        self.depth = None
        
        if TASK == "detect" :
            self.detect = Yolo26Detect(RK3588_RKNN_MODEL, 
                                    DATASET = DATASET, 
                                    OBJ_THRESH = DS_OBJ_THRESH
                                    )
        elif TASK == "pose" :
            self.pose = Yolo26Pose(RK3588_RKNN_MODEL, 
                                OBJ_THRESH = P_OBJ_THRESH,
                                calc_angle = CALC_ANGLE
                                )
        elif TASK == "seg" :
            self.seg = Yolo26Seg(RK3588_RKNN_MODEL, 
                            DATASET = DATASET,
                            IOU_THRESH = S_IOU_THRESH,
                            CONF_THRESH = DS_OBJ_THRESH
                            )
        elif TASK == "obb" :
            self.obb = Yolo26OBB(RK3588_RKNN_MODEL, 
                            score_thresh = O_SCORE_THRESH,
                            nms_thresh = O_NMS_THRESH
                            )
        elif TASK == "depth" :
            self.depth = Yolo26Depth(RK3588_RKNN_MODEL, 
                            )

    def release(self):
        if self.detect is not None :
            self.detect.release()
        if self.pose is not None :
            self.pose.release()
        if self.seg is not None :
            self.seg.release()
        if self.obb is not None :
            self.obb.release()
