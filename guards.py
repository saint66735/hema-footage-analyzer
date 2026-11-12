"""
Guard recognition for Liechtenauer longsword.

Classifies a fighter's current pose into one of the primary Huten using
geometric rules on MediaPipe landmarks.  Designed for single-frame
classification; the caller is responsible for temporal smoothing if needed.

Guards recognised
-----------------
  VOM_TAG   Roof guard — hands high, blade pointing up or back
  OCHS      Ox guard   — hands high, blade horizontal toward opponent
  PFLUG     Plow guard — hands low/mid, blade forward-upward
  ALBER     Fool guard — hands low, blade hanging downward
  LANGORT   Longpoint  — arms extended, blade straight at opponent
  UNKNOWN   Transitional or unclear position

Handedness
----------
  "left" / "right" / "center" — which side of the FIGHTER's body the
  sword hand is on.  Inferred from the fighter's facing direction (left
  shoulder vs. right shoulder position in the frame).
"""

import math
from dataclasses import dataclass

# Guard name constants
VOM_TAG = "vom_tag"
OCHS    = "ochs"
PFLUG   = "pflug"
ALBER   = "alber"
LANGORT = "langort"
UNKNOWN = "unknown"

# MediaPipe 33-point indices
_LS, _RS = 11, 12   # shoulders
_LH, _RH = 23, 24   # hips
_LW, _RW = 15, 16   # wrists
_LE, _RE = 13, 14   # elbows

_MIN_VIS = 0.25


@dataclass
class GuardResult:
    name:       str    # one of the constants above
    side:       str    # "left", "right", "center"
    confidence: float  # 0.0 – 1.0

    @property
    def label(self):
        """Short display string, e.g. 'vom_tag_r', 'pflug_l', 'langort'."""
        if self.name == UNKNOWN:
            return "?"
        if self.side == "center":
            return self.name
        return f"{self.name}_{self.side[0]}"   # vom_tag_r / ochs_l / …


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _lm(lms, idx):
    if idx >= len(lms):
        return None
    lm = lms[idx]
    return lm if (lm.visibility or 0) >= _MIN_VIS else None


def _px(lm, fw, fh):
    return lm.x * fw, lm.y * fh


def _mid(a, b):
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _body_metrics(lms, fw, fh):
    """
    Compute normalised body geometry.  Returns a dict or None if core
    torso landmarks are missing.

    All distances are normalised to spine_px (shoulder-midpoint →
    hip-midpoint distance), making the values camera-distance-independent.

    Height convention: positive = above shoulder midpoint (world up),
    achieved by inverting image-y (which increases downward).

    Facing convention: if left-shoulder x < right-shoulder x in the image,
    the fighter is approximately facing to the right.
    """
    ls_lm = _lm(lms, _LS);  rs_lm = _lm(lms, _RS)
    lh_lm = _lm(lms, _LH);  rh_lm = _lm(lms, _RH)
    if not (ls_lm and rs_lm and lh_lm and rh_lm):
        return None

    ls = _px(ls_lm, fw, fh);  rs = _px(rs_lm, fw, fh)
    lh = _px(lh_lm, fw, fh);  rh = _px(rh_lm, fw, fh)

    smid    = _mid(ls, rs)          # shoulder midpoint
    hmid    = _mid(lh, rh)          # hip midpoint
    spine   = _dist(smid, hmid)
    sh_w    = _dist(ls, rs)         # shoulder width
    if spine < 5 or sh_w < 1:
        return None

    # Fighter facing direction: left-shoulder image-x < right-shoulder image-x
    # means the fighter is facing to the image-right (typical side-on view)
    facing_right = ls[0] < rs[0]

    lw_lm = _lm(lms, _LW);  rw_lm = _lm(lms, _RW)
    if not lw_lm and not rw_lm:
        return None

    if lw_lm and rw_lm:
        lw = _px(lw_lm, fw, fh);  rw = _px(rw_lm, fw, fh)
        wmid       = _mid(lw, rw)
        bdx        = rw[0] - lw[0]
        bdy        = rw[1] - lw[1]          # image coords (down = +)
        blade_len  = math.hypot(bdx, bdy)
        # How vertical is the blade?  0 = horizontal, 1 = vertical
        blade_vert = abs(bdy) / blade_len if blade_len > 0 else 0.0
        # Blade elevation in world coords: +1 = pointing straight up
        # Image up = -y, so elevation = -(bdy / blade_len)
        blade_elev = (-bdy / blade_len) if blade_len > 0 else 0.0
        have_blade = True
    elif lw_lm:
        lw = _px(lw_lm, fw, fh);  rw = None;  wmid = lw
        blade_vert = None;  blade_elev = None;  have_blade = False
    else:
        rw = _px(rw_lm, fw, fh);  lw = None;  wmid = rw
        blade_vert = None;  blade_elev = None;  have_blade = False

    # Forearm direction: how much each wrist is below its own elbow, in spine units.
    # Positive = wrist is below elbow (world down) → forearm pointing downward.
    # This is the key Alber vs. Pflug discriminator at low hand positions.
    le_lm = _lm(lms, _LE);  re_lm = _lm(lms, _RE)
    forearm_dips = []
    if lw_lm and le_lm:
        le = _px(le_lm, fw, fh)
        lw_p = lw if lw_lm and lw is not None else _px(lw_lm, fw, fh)
        forearm_dips.append((lw_p[1] - le[1]) / spine)   # + = wrist below elbow
    if rw_lm and re_lm:
        re = _px(re_lm, fw, fh)
        rw_p = rw if rw_lm and rw is not None else _px(rw_lm, fw, fh)
        forearm_dips.append((rw_p[1] - re[1]) / spine)
    forearm_down = sum(forearm_dips) / len(forearm_dips) if forearm_dips else 0.0

    # Wrist midpoint height above shoulders (spine units, world up = +)
    wrist_h = (smid[1] - wmid[1]) / spine

    # Lateral offset of wrist midpoint from body centre, in shoulder-width units
    # Positive = image-right of body centre
    wrist_lat_img = (wmid[0] - smid[0]) / sh_w

    # Convert to fighter's own left/right
    # If facing right: image-right  = fighter's right
    # If facing left:  image-right  = fighter's left
    wrist_lat_fighter = wrist_lat_img if facing_right else -wrist_lat_img

    if wrist_lat_fighter > 0.55:
        side = "right"
    elif wrist_lat_fighter < -0.55:
        side = "left"
    else:
        side = "center"

    # Arm extension: distance smid → wmid in spine units
    wrist_ext = _dist(wmid, smid) / spine

    return {
        "wrist_h":      wrist_h,
        "blade_vert":   blade_vert,
        "blade_elev":   blade_elev,
        "wrist_ext":    wrist_ext,
        "side":         side,
        "have_blade":   have_blade,
        "forearm_down": forearm_down,
    }


def _scale_conf(base, amount, scale):
    """Interpolate confidence from base toward 0.90 as amount/scale → 1."""
    return round(min(0.90, base + (0.90 - base) * min(amount / max(scale, 1e-6), 1.0)), 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Adaptive switch thresholds — how many consecutive frames of a new guard
# are required before committing to it, based on how fast the wrists are moving.
#
# High motion (cutting, transitioning): require many frames — a briefly low
# hand position during a cut should NOT lock in Alber.
# Low motion (standing still in a guard): commit quickly — the fighter has settled.
_SWITCH_STILL  =  4   # px/frame wrist speed below which we consider "settled"
_SWITCH_FAST   = 35   # px/frame wrist speed above which we consider "transitioning"
_FRAMES_STILL  = 20   # frames required when hands are still (resist noise in held guards)
_FRAMES_FAST   =  4   # frames required when hands are moving (capture deliberate transitions)

# MediaPipe wrist indices (duplicated locally to keep GuardTracker self-contained)
_W_IDX = (15, 16)


class GuardTracker:
    """
    Wraps classify() with temporal smoothing so the displayed guard doesn't
    flip frame-by-frame.

    The number of consecutive frames required to confirm a guard change
    adapts to wrist speed: slow hands (settled guard) commit fast; fast
    hands (cut in progress) require many frames so brief intermediate
    positions don't accidentally lock in as guards.

    UNKNOWN frames are treated as "no information" and never reset the
    countdown or change the displayed guard.
    """

    def __init__(self):
        self._confirmed  = None   # currently displayed GuardResult
        self._candidate  = None   # guard being auditioned
        self._count      = 0      # consecutive frames of candidate
        self._prev_lms   = None   # landmarks from the previous frame

    def _wrist_speed(self, lms, fw, fh):
        """Average wrist pixel movement since the previous frame."""
        if self._prev_lms is None:
            return 0.0
        dists = []
        for idx in _W_IDX:
            if idx >= len(lms) or idx >= len(self._prev_lms):
                continue
            c = lms[idx];  p = self._prev_lms[idx]
            if (c.visibility or 0) < _MIN_VIS or (p.visibility or 0) < _MIN_VIS:
                continue
            dists.append(math.hypot((c.x - p.x) * fw, (c.y - p.y) * fh))
        return sum(dists) / len(dists) if dists else 0.0

    def _switch_frames(self, speed):
        """Required consecutive frames to confirm a new guard, given wrist speed.
        Fast hands = deliberate transition → commit quickly.
        Still hands = settled in current guard → resist noise."""
        if speed <= _SWITCH_STILL:
            return _FRAMES_STILL
        if speed >= _SWITCH_FAST:
            return _FRAMES_FAST
        t = (speed - _SWITCH_STILL) / (_SWITCH_FAST - _SWITCH_STILL)
        return int(_FRAMES_STILL - t * (_FRAMES_STILL - _FRAMES_FAST))

    def update(self, lms, fw, fh, weapon_line=None):
        """
        Call once per frame.  Returns the current confirmed GuardResult
        or None if no guard has been confirmed yet.

        weapon_line : (x1,y1,x2,y2) or None
            Fresh (non-stale) sword line from WeaponDetector.  When provided
            the blade angle from the line takes priority over the wrist-to-wrist
            estimate inside classify().
        """
        speed = self._wrist_speed(lms, fw, fh)
        self._prev_lms = lms

        raw = classify(lms, fw, fh, weapon_line=weapon_line)

        # No pose landmarks available this frame — hold current state
        if raw is None:
            return self._confirmed

        # UNKNOWN = transitional noise; don't let it interrupt anything
        if raw.name == UNKNOWN:
            return self._confirmed

        # First detection ever — accept immediately
        if self._confirmed is None:
            self._confirmed = raw
            return self._confirmed

        # Still in the same guard — refresh confidence, reset candidate
        if raw.name == self._confirmed.name and raw.side == self._confirmed.side:
            self._confirmed = raw
            self._candidate = None
            self._count     = 0
            return self._confirmed

        # Different guard seen — accumulate candidate frames
        if (self._candidate is not None and
                raw.name == self._candidate.name and
                raw.side  == self._candidate.side):
            self._count += 1
        else:
            self._candidate = raw
            self._count     = 1

        # Promote candidate once it's held long enough for the current speed
        if self._count >= self._switch_frames(speed):
            self._confirmed = self._candidate
            self._candidate = None
            self._count     = 0

        return self._confirmed


def _blade_from_line(line):
    """
    Compute blade_vert and blade_elev from a detected weapon line segment.
    More reliable than the wrist-to-wrist proxy when sword detection is fresh.
    Returns (bv, be) or (None, None).
    """
    if line is None:
        return None, None
    x1, y1, x2, y2 = line
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return None, None
    bv = abs(y2 - y1) / length          # 0 = horizontal, 1 = vertical
    be = -(y2 - y1) / length            # +1 = pointing up in world coords
    return bv, be


def classify(lms, fw, fh, weapon_line=None):
    """
    Classify the guard from pose landmarks, optionally refined by a detected
    weapon line from weapon.py.

    Parameters
    ----------
    lms : list of _Landmark
        Full-frame normalised landmarks returned by pose.PoseDetector.
    fw, fh : int
        Frame width and height in pixels.
    weapon_line : (x1, y1, x2, y2) or None
        Fresh (non-stale) sword line from WeaponDetector.detect().  When
        provided its angle overrides the wrist-to-wrist blade estimate,
        giving a much more reliable blade direction signal.

    Returns
    -------
    GuardResult or None
        None if the required torso landmarks are not visible.
    """
    m = _body_metrics(lms, fw, fh)
    if m is None:
        return None

    wh   = m["wrist_h"]
    ext  = m["wrist_ext"]
    side = m["side"]
    fwd  = m["forearm_down"]  # + = wrists clearly below elbows

    # Prefer weapon line angle when available (more reliable than wrist proxy)
    if weapon_line is not None:
        bv, be = _blade_from_line(weapon_line)
    else:
        bv = m["blade_vert"]   # None if only one wrist visible
        be = m["blade_elev"]

    # ── Langort ──────────────────────────────────────────────────────────────
    # Arms extended, blade STRICTLY horizontal (bv < 0.25 ≈ sin(15°)).
    if ext > 0.75 and (bv is None or bv < 0.25) and side == "center":
        return GuardResult(LANGORT, "center",
                           _scale_conf(0.60, ext - 0.75, 0.65))

    # ── High guards ───────────────────────────────────────────────────────
    if wh > 0.20:
        if bv is not None:
            if bv > 0.55:
                return GuardResult(VOM_TAG, side,
                                   _scale_conf(0.60, bv - 0.55, 0.45))
            elif bv < 0.40 and wh > 0.35:
                return GuardResult(OCHS, side,
                                   _scale_conf(0.60, wh - 0.35, 0.50))
        return GuardResult(VOM_TAG, side, 0.40)

    # ── Low guards (wrists below chest level) ────────────────────────────
    # wh = -0.45 ≈ chest/sternum; wh = -1.0 = hip midpoint.
    if wh < -0.45:
        # Alber: blade pointing CLEARLY downward.
        # With a weapon line: be < -0.45 means the blade tilts downward.
        # Without: require both a strongly downward forearm (fwd > 0.22,
        # i.e. wrists >22% of spine below elbows) AND a near-vertical
        # wrist-to-wrist axis (bv > 0.70) to avoid triggering on any
        # slight wrist dip present in almost every fighting stance.
        if weapon_line is not None:
            alber = (be is not None and be < -0.45)
        else:
            alber = (fwd > 0.22 and (bv is None or bv > 0.50)) or \
                    (bv is not None and bv > 0.70)

        if alber:
            conf = _scale_conf(0.60, max(fwd - 0.22, 0), 0.25)
            return GuardResult(ALBER, side, conf)

        # Pflug: blade angling forward/upward from low hands.
        if bv is not None and be is not None and be > 0.15 and bv < 0.65:
            return GuardResult(PFLUG, side,
                               _scale_conf(0.60, -wh - 0.45, 0.55))

        # Low hands, unclear orientation — Pflug is a safer default than Alber
        return GuardResult(PFLUG, side, 0.35)

    # ── Mid-torso range: transitional ────────────────────────────────────
    if ext > 0.65 and (bv is None or bv < 0.40) and side == "center":
        return GuardResult(LANGORT, "center", 0.35)

    return GuardResult(UNKNOWN, side, 0.25)
