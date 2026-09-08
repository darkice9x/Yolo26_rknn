#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import time
import cv2
import numpy as np
import re
import warnings
import os
import sys
import time
from utils import *

from rknnlite.api import RKNNLite  # noqa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))

STRIDE = 32  # YOLO network stride — all input dims must be multiples of 32

def colorize_depth_opencv(depth_map, min_depth=None, max_depth=None):
    # 1. 무효값(0 이하 또는 NaN/Inf) 마스킹
    valid_mask = (depth_map > 0) & np.isfinite(depth_map)
    
    if min_depth is None:
        min_depth = depth_map[valid_mask].min() if np.any(valid_mask) else 0.0
    if max_depth is None:
        max_depth = depth_map[valid_mask].max() if np.any(valid_mask) else 10.0

    # 2. 지정된 최소/최대 깊이로 클리핑 후 0~255 범위로 정규화
    depth_clamped = np.clip(depth_map, min_depth, max_depth)
    depth_norm = ((depth_clamped - min_depth) / (max_depth - min_depth) * 255.0).astype(np.uint8)

    # 3. 컬러맵 적용 (INFERNO, TURBO, JET 등 선택 가능)
    # OpenCV는 BGR 출력이므로 RGB로 변환
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    #depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    # 4. (선택) 무효 영역을 검은색으로 처리
    depth_colored[~valid_mask] = 0

    return depth_colored

def _align32(x: int) -> int:
    """Round an integer to the nearest multiple of STRIDE, clamped to >= 32."""
    return max(int(round(x / STRIDE) * STRIDE), STRIDE)


def compute_rect_input(src_h: int, src_w: int, model_max_dim: int) -> tuple[int, int]:
    """Compute rect (aspect-ratio-preserving) input size, matching ultralytics PT predict.

    Ultralytics' rect inference scales the image so that the longer side equals
    *model_max_dim* while keeping the aspect ratio, then rounds each dimension
    to the nearest stride-32 multiple.

    Args:
        src_h: Source image height.
        src_w: Source image width.
        model_max_dim: The reference dimension (usually the larger of model's H/W
                       for square models this is just imgsz).

    Returns:
        (input_h, input_w) stride-32 aligned.
    """
    scale = model_max_dim / max(src_h, src_w)
    new_h = _align32(src_h * scale)
    new_w = _align32(src_w * scale)
    return new_h, new_w


def derive_model_max_dim(imgsz: tuple[int, int]) -> int:
    """Derive the effective 'max_dim' from a model's fixed input size (H, W).

    For square models (640,640) this is 640.
    For rect models (640,480) this is 640 (the larger dimension).
    """
    return max(imgsz)


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
    m = re.search(r'_(\d+x\d+|\d+)[-.]', model_path)
    return parse_imgsz(m.group(1)) if m else (640, 640)


def detect_imgsz_from_onnx(onnx_path: str) -> tuple[int, int]:
    """Read input size (H, W) from ONNX model metadata."""
    import onnx
    m = onnx.load(onnx_path)
    shape = m.graph.input[0].type.tensor_type.shape.dim
    return int(shape[2].dim_value), int(shape[3].dim_value)  # H, W in N,C,H,W


def prepare_input_rect(image: np.ndarray, imgsz: tuple[int, int],
                       normalize: bool = True) -> np.ndarray:
    """Preprocess image with rect (aspect-ratio-preserving) inference, matching ultralytics PT predict.

    For each input image, the rect input size is computed from the image's aspect
    ratio. If it matches the model's fixed input size (H, W), the image is resized
    directly. Otherwise a warning is emitted and the image is stretched (fallback).

    Args:
        image: BGR source image (H, W, 3) uint8.
        imgsz: Model input size (H, W) — fixed for ONNX/RKNN models.
        normalize: If True, convert to range [0, 1] float32 NCHW (for ONNX).
                   If False, return uint8 NHWC (for RKNN which handles /255 internally).

    Returns:
        Preprocessed input tensor.
    """
    src_h, src_w = image.shape[:2]
    model_max_dim = derive_model_max_dim(imgsz)
    rect_h, rect_w = compute_rect_input(src_h, src_w, model_max_dim)
    mh, mw = imgsz

    if rect_h != mh or rect_w != mw:
        warnings.warn(
            f"Image aspect ratio ({src_w}:{src_h}) does not match model aspect ratio "
            f"({mw}:{mh}). Rect input would be {rect_h}x{rect_w} but model expects "
            f"{mh}x{mw}. Results will differ from ultralytics PT predict. "
            f"Export a rect model with --imgsz {rect_h}x{rect_w} for best accuracy.",
            UserWarning, stacklevel=2
        )
        # Fallback: stretch to model size
        rh, rw = mh, mw
    else:
        rh, rw = mh, mw

    inp = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    if normalize:
        # ONNX: float32 NCHW, /255
        inp_np = rgb.astype(np.float32) / 255.0
        return inp_np.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    else:
        # RKNN: uint8 NHWC (internal /255)
        return rgb[np.newaxis]


def colorize_depth(depth: np.ndarray, mode: str = "disparity") -> np.ndarray:
    """Convert metric depth map to a BGR heatmap image.

    Args:
        depth: (H, W) float32 depth map in meters.
        mode: "disparity" (1/d with percentile clipping) or "metric" (linear).

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if mode == "disparity":
        v = np.zeros_like(depth)
        v[valid] = 1.0 / np.maximum(depth[valid], 1e-6)
        lo, hi = np.percentile(v[valid], (2, 98))
    else:
        v = depth
        lo = float(depth[valid].min())
        hi = float(depth[valid].max())

    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((v - lo) / (hi - lo), 0, 1)
    idx = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(idx, cv2.COLORMAP_JET)
    heat[~valid] = 0
    return heat


def save_outputs(depth: np.ndarray, image: np.ndarray, save_path: str | None = None,
                 save_depth: str | None = None, save_heat: str | None = None,
                 mode: str = "disparity", overlay_alpha: float = 0.5):
    """Save inference results (overlay, heatmap, raw depth).

    Args:
        depth: (H, W) float32 depth map.
        image: Original BGR image (H, W, 3) uint8.
        save_path: If set, save overlay (image + heatmap blended).
        save_depth: If set, save raw depth as .npy.
        save_heat:  If set, save standalone heatmap image.
        mode:       "disparity" or "metric" for colorization.
        overlay_alpha: Blend weight for heatmap (0-1).
    """
    if save_path or save_heat:
        heat = colorize_depth(depth, mode=mode)

    if save_heat:
        cv2.imwrite(save_heat, heat)
        print(f"  Heatmap saved: {save_heat}")

    if save_path:
        overlay = cv2.addWeighted(image, 1 - overlay_alpha, heat, overlay_alpha, 0)
        cv2.imwrite(save_path, overlay)
        print(f"  Overlay saved: {save_path}")

    if save_depth:
        np.save(save_depth, depth.astype(np.float32))
        print(f"  Depth saved: {save_depth}")

class Yolo26Depth:
    """RKNN depth estimation model wrapper."""

    # Maps --core choice to RKNNLite core mask (RK3588 has 3 NPU cores)
    CORE_MASKS = {
        "auto": 0,  # NPU_CORE_AUTO
        "0": 1,     # NPU_CORE_0
        "1": 2,     # NPU_CORE_1
        "2": 4,     # NPU_CORE_2
        "012": 7,   # NPU_CORE_0_1_2
    }

    def __init__(self, 
                 RK3588_RKNN_MODEL: str, 
                 imgsz: str | int | None = None, 
                 core: str = "auto"
                 ) -> None:
        # (H, W): from --imgsz (e.g. "640" or "640x480") or the model filename
        self.imgsz = parse_imgsz(imgsz) if imgsz else detect_imgsz_from_path(RK3588_RKNN_MODEL)
        #print('--> Load YOLO model')
        self.rknn_lite = RKNNLite()
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
        print(f"RKNN model: {RK3588_RKNN_MODEL} (imgsz={self.imgsz}, core={core})")

    def __call__(self, input, overlay_alpha=0.5):
        """
        Call the detect method to perform inference on the input image.
        :param input: Input image, which can be a NumPy array or file path.
        :return: Processed image with detected keypoints and bounding boxes.
        """
        if isinstance(input, str):
            self.img_org = cv2.imread(input)
        else:
            self.img_org = input

        #self.img_result = self.img_org.copy()

        self.depth_map = self.predict(self.img_org)
        self.color_depth_map = colorize_depth(self.depth_map)
        self.img_result = cv2.addWeighted(self.img_org, 1 - overlay_alpha, self.color_depth_map, overlay_alpha, 0)
        
        #return boxes, classes, scores
        return self.img_result 
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image into model input tensor (uint8 NHWC).

        Can be called once and reused across benchmark iterations.
        """
        return prepare_input_rect(image, self.imgsz, normalize=False)

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run NPU inference on a preprocessed input tensor.

        Returns (H, W) float32 depth at model output resolution (not original size).
        """
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[input_tensor])
        self.infertime = (time.time() - start_time)*1000
        # Output is already float32 — no redundant astype()
        return np.squeeze(outputs[0])

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference, return (H, W) float32 depth at original image size.

        Preprocessing matches ultralytics PT predict: the image is resized with
        aspect-ratio-preserving rect scaling.  If the image aspect ratio differs
        from the model's, a warning is emitted.
        """
        src_h, src_w = image.shape[:2]
        tensor = self.preprocess(image)
        depth = self.infer(tensor)
        # Resize depth back to original image size
        depth = cv2.resize(depth, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def __del__(self):
        try:
            self.rknn_lite.release()
        except Exception:
            pass


def print_depth_stats(depth: np.ndarray):
    """Print depth statistics to stdout."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        print("  No valid depth values!")
        return
    print(f"  Median: {np.median(valid):.2f} m")
    print(f"  5%-95%: {np.percentile(valid, 5):.1f} ~ {np.percentile(valid, 95):.1f} m")
