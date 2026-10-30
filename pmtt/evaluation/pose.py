"""OpenCV essential-matrix estimation with explicit failure records."""
import ctypes as C
import numpy as np
from pmtt.settings import native_library
_lib=None

def estimate(p1, p2, threshold, seed=12345):
    global _lib
    if len(p1) < 5:
        return dict(status='failure', reason='insufficient_matches', inliers=[False] * len(p1))
    if _lib is None:
        _lib = C.CDLL(str(native_library()))
        arr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        mask = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
        _lib.pmtt_pose.argtypes = [
            C.c_int,
            arr,
            arr,
            C.c_double,
            C.c_int,
            arr,
            arr,
            arr,
            mask,
        ]
        _lib.pmtt_pose.restype = C.c_int
    p1 = np.ascontiguousarray(p1, dtype=np.float64)
    p2 = np.ascontiguousarray(p2, dtype=np.float64)
    E = np.zeros((3, 3))
    R = np.zeros((3, 3))
    t = np.zeros(3)
    mask = np.zeros(len(p1), np.uint8)
    if not np.isfinite(p1).all() or not np.isfinite(p2).all():
        raise ValueError('Non-finite points')
    code = _lib.pmtt_pose(len(p1), p1, p2, threshold, seed, E, R, t, mask)
    if code < 0:
        return dict(status='failure', reason='opencv_' + str(code), inliers=[False] * len(p1))
    return dict(status='success', reason=None, E=E.tolist(), R=R.tolist(), t=t.tolist(), inliers=(mask > 0).tolist())
