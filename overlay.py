import cv2

# Landmark connections for MediaPipe's 33-point body model
POSE_CONNECTIONS = frozenset([
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Shoulders / jaw
    (9, 10),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
])

_HIGH = 0.70
_MED  = 0.40

_GREEN       = (0,   220,   0)   # high-confidence landmark/edge
_YELLOW      = (0,   200, 255)   # medium confidence (BGR: orange-yellow)
_RED         = (0,     0, 255)   # low confidence
_CYAN        = (255, 220,   0)   # weapon line — fresh detection (bright cyan in BGR)
_CYAN_STALE  = (120, 100,   0)   # weapon line — stale / carried over from prior frame
_SEARCH_GRAY = (70,   70,  70)   # weapon search-region outline

# One colour per tracked fighter (BGR)
_FIGHTER_COLORS = [
    (0,  140, 255),   # Fighter A — orange
    (255, 80,   0),   # Fighter B — blue
]

_DOT_RADIUS = 4
_LINE_WIDTH = 2
_BOX_WIDTH  = 2


def _color(visibility):
    if visibility >= _HIGH:
        return _GREEN
    if visibility >= _MED:
        return _YELLOW
    return _RED


# ---------------------------------------------------------------------------
# Pose skeleton
# ---------------------------------------------------------------------------

def draw_poses(frame, pose_landmarks_list):
    """Overlay skeletons for all detected poses, colour-coded by landmark visibility."""
    h, w = frame.shape[:2]
    for landmarks in pose_landmarks_list:
        pts = []
        for lm in landmarks:
            vis = lm.visibility if lm.visibility is not None else 0.0
            pts.append((int(lm.x * w), int(lm.y * h), vis))

        for (a, b) in POSE_CONNECTIONS:
            if a >= len(pts) or b >= len(pts):
                continue
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            cv2.line(frame, (xa, ya), (xb, yb), _color(min(va, vb)), _LINE_WIDTH)

        for (x, y, vis) in pts:
            cv2.circle(frame, (x, y), _DOT_RADIUS, _color(vis), -1)


# ---------------------------------------------------------------------------
# Fighter tracker boxes
# ---------------------------------------------------------------------------

def draw_tracker_boxes(frame, boxes, guard_results=None, stale=None):
    """
    Draw labelled bounding boxes for the two tracked fighters.
    boxes: list of two [x1, y1, x2, y2].
    guard_results: optional list of two GuardResult (or None) from guards.py.
    stale: optional list of two bools; a stale fighter (not seen for a while)
           gets a "LOST" note in the corner instead of a box at an old position.
    """
    labels = ["Fighter A", "Fighter B"]
    for i, (box, color) in enumerate(zip(boxes, _FIGHTER_COLORS)):
        if stale and stale[i]:
            cv2.putText(frame, f"{labels[i]}: LOST", (12, 30 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, _BOX_WIDTH)
        cv2.putText(frame, labels[i], (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if guard_results and guard_results[i] is not None:
            g = guard_results[i]
            guard_text = g.label
            # Dim the colour slightly for lower-confidence labels
            alpha = max(0.45, g.confidence)
            dim   = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, guard_text, (x1 + 4, y1 + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, dim, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Weapon: search regions + detected lines
# ---------------------------------------------------------------------------

def draw_weapon_results(frame, weapon_results):
    """
    Draw weapon search regions and detected line segments.
    weapon_results: list of dicts from weapon.detect_weapon().
      Each dict: {"search_box": (x1,y1,x2,y2), "line": (x1,y1,x2,y2) or None}
    """
    for det in weapon_results:
        if det.get("search_box") is not None:
            bx1, by1, bx2, by2 = det["search_box"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), _SEARCH_GRAY, 1)

        if det["line"] is not None:
            lx1, ly1, lx2, ly2 = det["line"]
            color = _CYAN_STALE if det.get("stale") else _CYAN
            cv2.line(frame, (lx1, ly1), (lx2, ly2), color, 3)
            cv2.circle(frame, (lx1, ly1), 5, color, -1)
