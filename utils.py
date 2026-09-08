
import cv2
import numpy as np
import math
import platform
import re
import matplotlib.pyplot as plt

name = "YOLO"
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                elif 'rk3576' in device_compatible_str:
                    host = 'RK3576'
                elif 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

def print_info(messages: str):
    print( f"{name} : {messages}")

def letterbox(image, target_width, target_height, bg_color):
    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        raise ValueError("Input image is None")

    image_height, image_width = image.shape[:2]

    aspect_ratio = min(target_width / image_width,
                       target_height / image_height)

    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    image = cv2.resize(image, (new_width, new_height),
                       interpolation=cv2.INTER_AREA)

    result_image = np.ones(
        (target_height, target_width, 3),
        dtype=np.uint8
    ) * bg_color

    result_image[0:new_height, 0:new_width] = image
    return result_image, aspect_ratio

def just( image ) :
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 700, 350, apertureSize = 5, L2gradient = True)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength = 3, maxLineGap = 150)
    
    angle = 0
    maxdim = 0
    angles = []
    if not (lines is None):
        for line in lines:
            x1, y1, x2, y2 = line
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            angles.append(angle_deg)

    # 각도들을 히스토그램으로 표현하여 가장 많이 등장하는 각도를 찾습니다.
    histogram = cv2.calcHist([np.array(angles).astype(np.float32)], [0], None, [180], [-90, 90])
    max_angle_index = np.argmax(histogram)
    most_frequent_angle = max_angle_index - 90  # 각도 범위는 -90도부터 90도까지입니다.

    #print("가장 빈번하게 등장하는 각도:", most_frequent_angle)
    
    roih, roiw, roic = image.shape
    matrix = cv2.getRotationMatrix2D((roiw/2, roih/2), most_frequent_angle, 1)
    roi = cv2.warpAffine(image, matrix, (roiw, roih))
    #roi_org = cv2.warpAffine(image_org, matrix, (roiw, roih))
    return  roi

def plt_img_show(src_img, title, percent=100, axis='off'):
    height, width, _ = src_img.shape
    scale = percent / 100
    fig_w = (width / 100) * scale
    fig_h = (height / 100) * scale

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    plt.axis(axis)
    plt.title(title)
    plt.show()

def parse_imgsz(value: str | int) -> tuple[int, int]:
    """Parse an imgsz spec into (H, W).

    Examples:
        640       -> (640, 640)
        "640"     -> (640, 640)
        "640x480" -> (640, 480)   # H x W, rect model
    """
    s = str(value)
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    return int(s), int(s)

def detect_imgsz_from_path(model_path: str) -> tuple[int, int]:
    """Auto-detect input size (H, W) from model filename.

    Examples:
        yolo26n-depth-float.rknn           -> (640, 640)
        yolo26n-depth_768-float.rknn       -> (768, 768)
        yolo26x-depth_1280.onnx            -> (1280, 1280)
        yolo26n-depth_640x480-float.rknn   -> (640, 480)
    """
    m = re.search(r'_(\d+x\d+|\d+)[-_.]', model_path)
    return parse_imgsz(m.group(1)) if m else (640, 640)
