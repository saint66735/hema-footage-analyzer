import cv2
import numpy as np

_MAX_CORNERS = 200
_QUALITY     = 0.01
_MIN_DIST    = 30
_MIN_POINTS  = 5


def estimate_camera_motion(prev_gray, curr_gray):
    """
    Estimate per-frame camera translation (pan/tilt) using sparse Lucas-Kanade
    optical flow. Majority of tracked points are background, so the median
    displacement gives a robust global motion estimate even when fighters move.

    Returns (dx, dy) in pixels, or (0.0, 0.0) if estimation fails.
    """
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=_MAX_CORNERS,
        qualityLevel=_QUALITY,
        minDistance=_MIN_DIST,
    )
    if prev_pts is None or len(prev_pts) < _MIN_POINTS:
        return 0.0, 0.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    good_prev = prev_pts[status.ravel() == 1]
    good_curr = curr_pts[status.ravel() == 1]

    if len(good_prev) < _MIN_POINTS:
        return 0.0, 0.0

    dx = float(np.median(good_curr[:, 0, 0] - good_prev[:, 0, 0]))
    dy = float(np.median(good_curr[:, 0, 1] - good_prev[:, 0, 1]))
    return dx, dy
