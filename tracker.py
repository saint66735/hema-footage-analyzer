import cv2
import numpy as np


def _iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _hist(frame_bgr, box):
    x1 = max(0, box[0]);  y1 = max(0, box[1])
    x2 = min(frame_bgr.shape[1], box[2]);  y2 = min(frame_bgr.shape[0], box[3])
    if x2 <= x1 or y2 <= y1:
        return None
    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


def _color_corr(h1, h2):
    if h1 is None or h2 is None:
        return 0.5
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


class FighterTracker:
    """
    Tracks two fighters by following ByteTrack-assigned person IDs.

    Pass 1 — follow known track IDs directly.  ByteTrack's internal Kalman
    filter handles motion; no additional check needed.

    Pass 2 — re-assign when a track ID is missing (ByteTrack dropped the
    track after a gap, and will have assigned a new ID on reappearance).
    Re-assignment requires BOTH spatial overlap (IOU) AND colour match against
    the seed-frame histogram.  This prevents accidentally grabbing a nearby
    bystander who happens to overlap the last known position.
    """

    _REID_IOU   = 0.20
    _REID_COLOR = 0.40

    def __init__(self, box_a, box_b, seed_frame):
        self._boxes     = [list(box_a), list(box_b)]
        self._track_ids = [None, None]
        self._hists     = [_hist(seed_frame, box_a), _hist(seed_frame, box_b)]

    def update(self, track_detections, frame):
        """
        track_detections: dict {track_id: [x1, y1, x2, y2]} from PersonDetector.track().
        frame: current BGR frame (needed to compute candidate colour histograms for re-ID).
        Returns (boxes, matched).
        """
        matched = [False, False]

        if not track_detections:
            return list(self._boxes), matched

        claimed      = set()
        cand_hists   = {tid: _hist(frame, box) for tid, box in track_detections.items()}

        # Pass 1: follow known track IDs — trust ByteTrack, no colour check
        for i in range(2):
            tid = self._track_ids[i]
            if tid is not None and tid in track_detections:
                self._boxes[i] = list(track_detections[tid])
                matched[i] = True
                claimed.add(tid)

        # Pass 2: re-assign for fighters whose track was dropped.
        # Require IOU overlap + colour match so we don't grab a bystander.
        for i in range(2):
            if matched[i]:
                continue
            best_score, best_id = 0.0, None
            for tid, box in track_detections.items():
                if tid in claimed:
                    continue
                iou   = _iou(self._boxes[i], box)
                if iou < self._REID_IOU:
                    continue
                color = _color_corr(self._hists[i], cand_hists[tid])
                if color < self._REID_COLOR:
                    continue
                score = iou + 0.5 * color
                if score > best_score:
                    best_score, best_id = score, tid
            if best_id is not None:
                self._track_ids[i] = best_id
                self._boxes[i]     = list(track_detections[best_id])
                matched[i]         = True
                claimed.add(best_id)

        return list(self._boxes), matched

    @property
    def boxes(self):
        return list(self._boxes)
