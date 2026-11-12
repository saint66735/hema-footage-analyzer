import cv2
import numpy as np


def _iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


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

    Pass 2 — short gap: re-assign when a track ID is missing (ByteTrack dropped
    the track and assigned a new ID on reappearance).  Requires spatial overlap
    (IOU) with the last known box AND a colour match.

    Pass 3 — long gap: the fighter has been missing long enough that the last
    known box is meaningless (e.g. the camera panned and they left the frame,
    then came back somewhere else).  Overlap is no longer required; the best
    colour match among the remaining people wins, with a stricter colour bar
    and a small preference for candidates near the last known position / the
    other fighter.  This is what stops a fighter being lost for the rest of
    the video after walking out of shot.

    Appearance model — each fighter's colour histogram starts from the seed
    frame and is slowly updated while that fighter is confidently tracked, so
    lighting changes (and seed frames with logos/overlays on them) don't
    permanently break re-identification.
    """

    _REID_IOU        = 0.20
    _REID_COLOR      = 0.40
    _LONG_GAP_FRAMES = 8      # after this many missed frames, Pass 3 is allowed
    _LONG_GAP_COLOR  = 0.50   # stricter colour bar when position can't help
    _HIST_RATE       = 0.05   # rolling appearance update per confident frame
    _STALE_FRAMES    = 5      # lost longer than this → box is considered stale

    def __init__(self, box_a, box_b, seed_frame):
        self._boxes     = [list(box_a), list(box_b)]
        self._track_ids = [None, None]
        self._hists     = [_hist(seed_frame, box_a), _hist(seed_frame, box_b)]
        self._lost      = [0, 0]   # consecutive frames without a match
        self._diag      = None     # frame diagonal, set on first update

    # ------------------------------------------------------------------

    def update(self, track_detections, frame):
        """
        track_detections: dict {track_id: [x1, y1, x2, y2]} from PersonDetector.track().
        frame: current BGR frame (needed to compute candidate colour histograms for re-ID).
        Returns (boxes, matched).
        """
        if self._diag is None:
            self._diag = float(np.hypot(frame.shape[0], frame.shape[1]))

        matched = [False, False]

        if not track_detections:
            self._bump_lost(matched)
            return list(self._boxes), matched

        claimed    = set()
        cand_hists = {tid: _hist(frame, box) for tid, box in track_detections.items()}

        # Pass 1: follow known track IDs — trust ByteTrack, no colour check
        for i in range(2):
            tid = self._track_ids[i]
            if tid is not None and tid in track_detections:
                self._boxes[i] = list(track_detections[tid])
                matched[i] = True
                claimed.add(tid)

        # Pass 2: short gap — require IOU overlap + colour match
        for i in range(2):
            if matched[i]:
                continue
            best_score, best_id = 0.0, None
            for tid, box in track_detections.items():
                if tid in claimed:
                    continue
                iou = _iou(self._boxes[i], box)
                if iou < self._REID_IOU:
                    continue
                color = _color_corr(self._hists[i], cand_hists[tid])
                if color < self._REID_COLOR:
                    continue
                score = iou + 0.5 * color
                if score > best_score:
                    best_score, best_id = score, tid
            if best_id is not None:
                self._assign(i, best_id, track_detections, claimed, matched)

        # Pass 3: long gap — position no longer trusted; colour decides
        for i in range(2):
            if matched[i] or self._lost[i] < self._LONG_GAP_FRAMES:
                continue
            other = 1 - i
            best_score, best_id = -1e9, None
            for tid, box in track_detections.items():
                if tid in claimed:
                    continue
                color = _color_corr(self._hists[i], cand_hists[tid])
                if color < self._LONG_GAP_COLOR:
                    continue
                # Fighters stay near each other; prefer candidates close to the
                # other fighter (if tracked) or to the last known position.
                ref = self._boxes[other] if matched[other] else self._boxes[i]
                dist = np.hypot(*np.subtract(_center(box), _center(ref))) / self._diag
                score = color - 0.5 * dist
                if score > best_score:
                    best_score, best_id = score, tid
            if best_id is not None:
                self._assign(i, best_id, track_detections, claimed, matched)

        # Rolling appearance update — only when both are tracked and the two
        # boxes don't overlap (otherwise the histogram mixes both fighters).
        if matched[0] and matched[1] and _iou(self._boxes[0], self._boxes[1]) < 0.05:
            for i in range(2):
                h = cand_hists.get(self._track_ids[i])
                if h is not None and self._hists[i] is not None:
                    self._hists[i] = (1 - self._HIST_RATE) * self._hists[i] + self._HIST_RATE * h
                elif h is not None:
                    self._hists[i] = h

        self._bump_lost(matched)
        return list(self._boxes), matched

    # ------------------------------------------------------------------

    def _assign(self, i, tid, track_detections, claimed, matched):
        self._track_ids[i] = tid
        self._boxes[i]     = list(track_detections[tid])
        matched[i]         = True
        claimed.add(tid)

    def _bump_lost(self, matched):
        for i in range(2):
            self._lost[i] = 0 if matched[i] else self._lost[i] + 1

    @property
    def boxes(self):
        return list(self._boxes)

    @property
    def stale(self):
        """Per fighter: True when the box is an old position, not a live detection."""
        return [n > self._STALE_FRAMES for n in self._lost]
