import cv2
import math
import numpy as np

# MediaPipe landmark indices
_LM_LEFT_ELBOW  = 13
_LM_LEFT_WRIST  = 15
_LM_RIGHT_ELBOW = 14
_LM_RIGHT_WRIST = 16

# Low threshold so uncertain wrists (mittens, dark gambesons) still anchor the strip.
_MIN_VIS_ANCHOR = 0.20

# Motion mask
_DIFF_THRESH   = 15
_MOTION_DILATE = 3

# Hough
_HOUGH_THRESH  = 15
_MAX_GAP       = 25

# Two-wrist strip geometry
_EXTEND_FRONT  = 4.0
_GRIP_PAD      = 0.5
_MIN_LINE_GRIP = 1.0
_WRIST_PROX    = 0.45
_ANGLE_TOL     = 20

# Single-wrist forearm extension (blade-tip search)
# A longsword blade is ~3.5x the hand-separation past the front wrist.
# We extend 4.5x the forearm length past each wrist so the tip strip is
# long enough even when the fighter is close to the camera.
_FOREARM_EXTEND = 4.5
_FA_PAD         = 0.6
_FA_MIN_LINE    = 0.7
_FA_WRIST_PROX  = 0.9
_FA_ANGLE_TOL   = 20

# Temporal smoothing
_TEMP_ANGLE_TOL = 35    # degrees — within this of prev detection → get score bonus
_TEMP_POS_TOL   = 200   # pixels — line midpoint within this of prev → get score bonus
_TEMP_BONUS     = 1.5   # score multiplier for temporally consistent candidates
_STALE_LIMIT    = 6     # frames to hold a stale (not freshly detected) line


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def _angle_diff(a1, a2):
    d = abs(a1 - a2) % 180
    return min(d, 180 - d)


def _line_dist(x1, y1, x2, y2, px, py):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / math.hypot(dx, dy)


def _line_mid(line):
    x1, y1, x2, y2 = line
    return (x1 + x2) / 2, (y1 + y2) / 2


def _line_angle(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _search_rect(pts, pad, fw, fh):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = int(max(0,  min(xs) - pad))
    y1 = int(max(0,  min(ys) - pad))
    x2 = int(min(fw, max(xs) + pad))
    y2 = int(min(fh, max(ys) + pad))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _motion_edges(curr_gray, prev_gray, sbox):
    """Canny edges AND-masked with frame-diff motion mask inside sbox."""
    sx1, sy1, sx2, sy2 = sbox
    crop_c = curr_gray[sy1:sy2, sx1:sx2]
    edges  = cv2.Canny(crop_c, 50, 150)
    if prev_gray is not None:
        crop_p    = prev_gray[sy1:sy2, sx1:sx2]
        diff      = cv2.absdiff(crop_c, crop_p)
        _, motion = cv2.threshold(diff, _DIFF_THRESH, 255, cv2.THRESH_BINARY)
        motion    = cv2.dilate(motion, np.ones((3, 3), np.uint8),
                               iterations=_MOTION_DILATE)
        edges = cv2.bitwise_and(edges, motion)
    return edges


def _hough(edge_img, offset, min_len):
    ox, oy = offset
    lines  = cv2.HoughLinesP(
        edge_img,
        rho=1, theta=np.pi / 180,
        threshold=_HOUGH_THRESH,
        minLineLength=int(min_len),
        maxLineGap=_MAX_GAP,
    )
    if lines is None:
        return []
    out = []
    for seg in lines:
        cx1, cy1, cx2, cy2 = seg[0]
        out.append((
            math.hypot(cx2 - cx1, cy2 - cy1),
            (cx1 + ox, cy1 + oy, cx2 + ox, cy2 + oy),
        ))
    out.sort(reverse=True)
    return out


# ---------------------------------------------------------------------------
# Candidate finders — return ALL qualifying lines, not just the first
# ---------------------------------------------------------------------------

def _candidates_two_wrists(lw, rw, curr_gray, prev_gray, fw, fh):
    """
    Build the wrist-to-wrist axis strip and return all lines that pass both
    the angle filter and the two-wrist proximity gate.
    Tries motion-masked Canny first; falls back to plain Canny if it yields
    nothing (covers held guards where the sword barely moves).
    Returns (search_box, [(length, line), ...]).
    """
    grip_len = math.hypot(rw[0] - lw[0], rw[1] - lw[1])
    if grip_len < 15:
        return None, []

    dx = (rw[0] - lw[0]) / grip_len
    dy = (rw[1] - lw[1]) / grip_len
    end_a = (lw[0] - dx * grip_len * _EXTEND_FRONT,
             lw[1] - dy * grip_len * _EXTEND_FRONT)
    end_b = (rw[0] + dx * grip_len * _EXTEND_FRONT,
             rw[1] + dy * grip_len * _EXTEND_FRONT)

    sbox = _search_rect([end_a, end_b], grip_len * _GRIP_PAD, fw, fh)
    if sbox is None:
        return None, []

    sword_angle = math.degrees(math.atan2(dy, dx))
    max_wrist_d = grip_len * _WRIST_PROX
    min_len     = grip_len * _MIN_LINE_GRIP

    for use_motion in (True, False):
        if use_motion:
            edges = _motion_edges(curr_gray, prev_gray, sbox)
        else:
            sx1, sy1, sx2, sy2 = sbox
            edges = cv2.Canny(curr_gray[sy1:sy2, sx1:sx2], 50, 150)

        qualifiers = []
        for length, line in _hough(edges, (sbox[0], sbox[1]), min_len):
            x1, y1, x2, y2 = line
            if _angle_diff(_line_angle(line), sword_angle) > _ANGLE_TOL:
                continue
            if (_line_dist(x1, y1, x2, y2, lw[0], lw[1]) > max_wrist_d or
                    _line_dist(x1, y1, x2, y2, rw[0], rw[1]) > max_wrist_d):
                continue
            qualifiers.append((length, line))

        if qualifiers:
            return sbox, qualifiers   # motion pass succeeded — skip static

    return sbox, []


def _candidates_single_wrist(wrist_px, elbow_px, curr_gray, prev_gray, fw, fh):
    """
    Forearm-direction strip from a single wrist.  Returns all qualifying lines.
    """
    wx, wy = wrist_px
    ex, ey = elbow_px
    flen   = math.hypot(wx - ex, wy - ey)
    if flen < 15:
        return None, []

    dx    = (wx - ex) / flen
    dy    = (wy - ey) / flen
    tip_x = wx + dx * flen * _FOREARM_EXTEND
    tip_y = wy + dy * flen * _FOREARM_EXTEND

    sbox = _search_rect([(wx, wy), (tip_x, tip_y)], flen * _FA_PAD, fw, fh)
    if sbox is None:
        return None, []

    fa_angle  = math.degrees(math.atan2(dy, dx))
    max_wrist = flen * _FA_WRIST_PROX
    min_len   = flen * _FA_MIN_LINE

    for use_motion in (True, False):
        if use_motion:
            edges = _motion_edges(curr_gray, prev_gray, sbox)
        else:
            sx1, sy1, sx2, sy2 = sbox
            edges = cv2.Canny(curr_gray[sy1:sy2, sx1:sx2], 50, 150)

        qualifiers = []
        for length, line in _hough(edges, (sbox[0], sbox[1]), min_len):
            x1, y1, x2, y2 = line
            if _angle_diff(_line_angle(line), fa_angle) > _FA_ANGLE_TOL:
                continue
            if _line_dist(x1, y1, x2, y2, wx, wy) > max_wrist:
                continue
            qualifiers.append((length, line))

        if qualifiers:
            return sbox, qualifiers

    return sbox, []


# ---------------------------------------------------------------------------
# Stateful per-fighter detector with temporal consistency memory
# ---------------------------------------------------------------------------

class WeaponDetector:
    """
    Per-fighter weapon detector.  Must be instantiated once per fighter so
    that temporal state is not shared between the two tracks.

    Each call to detect() returns a list with one dict:
        {"search_box": (x1,y1,x2,y2) | None, "line": (x1,y1,x2,y2) | None,
         "stale": bool}

    "stale" is True when the returned line is carried over from a previous
    frame (no fresh detection this frame).  Overlay draws stale lines dimmer.
    """

    def __init__(self):
        self._prev_line   = None   # last confirmed line (x1,y1,x2,y2)
        self._stale_count = 0

    def _score(self, length, line):
        """
        Score = line length, boosted if the candidate is temporally consistent
        with the previous detection.  This makes stable tracking prefer
        candidates near the known sword position over equally long spurious edges.
        """
        if self._prev_line is None:
            return length
        prev_angle = _line_angle(self._prev_line)
        pmx, pmy   = _line_mid(self._prev_line)
        mx,  my    = _line_mid(line)
        angle_ok   = _angle_diff(_line_angle(line), prev_angle) <= _TEMP_ANGLE_TOL
        pos_ok     = math.hypot(mx - pmx, my - pmy)          <= _TEMP_POS_TOL
        return length * (_TEMP_BONUS if (angle_ok and pos_ok) else 1.0)

    def detect(self, landmarks, frame_bgr, curr_gray, prev_gray, fighter_box):
        fh, fw = frame_bgr.shape[:2]

        def _lm(idx):
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            return lm if (lm.visibility or 0) >= _MIN_VIS_ANCHOR else None

        lw_lm = _lm(_LM_LEFT_WRIST)
        rw_lm = _lm(_LM_RIGHT_WRIST)
        le_lm = _lm(_LM_LEFT_ELBOW)
        re_lm = _lm(_LM_RIGHT_ELBOW)

        sbox      = None
        all_cands = []

        # Primary: two-wrist grip axis (covers the handle / grip region)
        if lw_lm and rw_lm:
            lw = _px(lw_lm, fw, fh)
            rw = _px(rw_lm, fw, fh)
            sbox, cands = _candidates_two_wrists(lw, rw, curr_gray, prev_gray, fw, fh)
            all_cands.extend(cands)

        # Blade-tip extension: always run past each wrist in the forearm direction,
        # not just as a fallback.  The grip-axis search centres on the hands;
        # the forearm extension strips cover where the blade tip actually ends up
        # during cuts and thrusts, which is well beyond the wrist landmarks.
        for elbow_lm, wrist_lm in [(le_lm, lw_lm), (re_lm, rw_lm)]:
            if not wrist_lm or not elbow_lm:
                continue
            s, tip_cands = _candidates_single_wrist(
                _px(wrist_lm, fw, fh), _px(elbow_lm, fw, fh),
                curr_gray, prev_gray, fw, fh,
            )
            if sbox is None:
                sbox = s
            all_cands.extend(tip_cands)

        # Pick best candidate using temporal scoring
        best_line = None
        if all_cands:
            best_line = max(all_cands, key=lambda t: self._score(t[0], t[1]))[1]

        if best_line is not None:
            self._prev_line   = best_line
            self._stale_count = 0
            return [{"search_box": sbox, "line": best_line, "stale": False}]

        if self._prev_line is not None and self._stale_count < _STALE_LIMIT:
            self._stale_count += 1
            return [{"search_box": sbox, "line": self._prev_line, "stale": True}]

        # No detection and no usable prior — clear state
        self._prev_line   = None
        self._stale_count = 0
        if sbox is None:
            return []
        return [{"search_box": sbox, "line": None, "stale": False}]
