import csv
import math
import os


class StatsCollector:
    """
    Accumulates per-frame diagnostic data and writes it to a CSV file.

    Columns
    -------
    frame, time_s
        Frame index and wall-clock position.
    cam_dx, cam_dy
        Estimated camera pan/tilt in pixels for this frame.
    a_matched, b_matched
        1 = tracker got a fresh YOLO detection this frame; 0 = coasting on last box.
    a_pose, b_pose
        1 = MediaPipe found a pose in the fighter's crop.
    a_vis_mean, b_vis_mean
        Mean landmark visibility (0–1) across all 33 landmarks.
    a_vis_min, b_vis_min
        Minimum landmark visibility.
    a_jitter, b_jitter
        Average pixel distance each landmark moved vs the previous detected frame.
    a_wrist_vis, b_wrist_vis
        Lower of the two wrist visibilities (0–1). Low = at least one hand lost.
    a_wrist_jump, b_wrist_jump
        Largest wrist movement since the previous pose frame, in torso lengths
        (shoulder-to-hip). Values above ~0.6 in a single frame are physically
        implausible and almost always mean the wrist landmark jumped to the
        wrong place.
    a_arm_ok, b_arm_ok
        1 = both wrists and both elbows have visibility >= 0.5 AND neither wrist
        jumped implausibly. Empty when no pose. This is the "are the arms usable
        for blade work" signal summarised at the end of a run.
    a_weapon, b_weapon
        1 = at least one weapon line was detected near this fighter's wrist.
    a_guard, b_guard
        Best-guess guard label for this frame, e.g. "vom_tag_r", "pflug_l",
        "langort", "?".  Empty when the pose is not available.
    fighter_dist
        Pixel distance between the two fighter box centroids.
        Low = fighters are close together (engaged or binding).
    wrist_dist
        Pixel distance between the nearest visible wrist pair across the two fighters.
        Better proxy for sword-tip proximity than box centroid distance.
        Empty when either fighter has no visible wrists.
    sword_dist
        Minimum pixel distance between the two detected sword line segments.
        0 = segments are crossing (bind). Empty when either fighter has no
        detected sword line this frame.
    """

    # Wrist landmark indices in MediaPipe's 33-point model
    _LM_WRISTS = (15, 16)   # left wrist, right wrist
    _MIN_WRIST_VIS = 0.20
    _LM_ELBOWS = (13, 14)
    _ARM_VIS_OK = 0.50
    _WRIST_JUMP_MAX = 0.60   # torso lengths per frame

    def __init__(self, input_path, output_dir, fps, frame_w, frame_h):
        stem = os.path.splitext(os.path.basename(input_path))[0]
        self._path  = os.path.join(output_dir, f"{stem}_stats.csv")
        self._fps   = fps
        self._fw    = frame_w
        self._fh    = frame_h
        self._rows  = []
        self._prev  = [None, None]   # last detected landmarks per fighter

    def record(self, frame_idx, matched, per_fighter_lms, weapon_per_fighter,
               cam_motion, fighter_boxes=None, guard_results=None):
        """Call once per processed frame."""
        row = {
            "frame":  frame_idx,
            "time_s": round(frame_idx / self._fps, 3),
            "cam_dx": round(cam_motion[0], 2),
            "cam_dy": round(cam_motion[1], 2),
        }

        for i, label in enumerate(["a", "b"]):
            lms = per_fighter_lms[i]
            row[f"{label}_matched"] = int(matched[i])
            row[f"{label}_pose"]    = int(lms is not None)

            if lms:
                vis = [lm.visibility for lm in lms]
                row[f"{label}_vis_mean"] = round(sum(vis) / len(vis), 3)
                row[f"{label}_vis_min"]  = round(min(vis), 3)
                row[f"{label}_jitter"]   = self._jitter(lms, self._prev[i])
                wv, wj, ok = self._arm_quality(lms, self._prev[i])
                row[f"{label}_wrist_vis"]  = wv
                row[f"{label}_wrist_jump"] = wj
                row[f"{label}_arm_ok"]     = ok
                self._prev[i] = lms
            else:
                row[f"{label}_vis_mean"] = ""
                row[f"{label}_vis_min"]  = ""
                row[f"{label}_jitter"]   = ""
                row[f"{label}_wrist_vis"]  = ""
                row[f"{label}_wrist_jump"] = ""
                row[f"{label}_arm_ok"]     = ""

            row[f"{label}_weapon"] = int(
                any(r.get("line") is not None for r in weapon_per_fighter[i])
            )

        # Proximity signals
        row["fighter_dist"] = self._fighter_dist(fighter_boxes)
        row["wrist_dist"]   = self._wrist_dist(per_fighter_lms)
        row["sword_dist"]   = self._sword_dist(weapon_per_fighter)

        # Guard labels
        if guard_results:
            for i, label in enumerate(["a", "b"]):
                g = guard_results[i]
                row[f"{label}_guard"] = g.label if g is not None else ""
        else:
            row["a_guard"] = ""
            row["b_guard"] = ""

        self._rows.append(row)

    # ------------------------------------------------------------------
    # Proximity helpers
    # ------------------------------------------------------------------

    def _fighter_dist(self, boxes):
        if not boxes or len(boxes) < 2:
            return ""
        a, b = boxes[0], boxes[1]
        ax = (a[0] + a[2]) / 2;  ay = (a[1] + a[3]) / 2
        bx = (b[0] + b[2]) / 2;  by = (b[1] + b[3]) / 2
        return round(math.hypot(bx - ax, by - ay), 1)

    def _wrist_dist(self, per_fighter_lms):
        lms_a, lms_b = per_fighter_lms[0], per_fighter_lms[1]
        if lms_a is None or lms_b is None:
            return ""
        wrists_a = self._visible_wrists(lms_a)
        wrists_b = self._visible_wrists(lms_b)
        if not wrists_a or not wrists_b:
            return ""
        min_d = min(
            math.hypot(ax - bx, ay - by)
            for ax, ay in wrists_a
            for bx, by in wrists_b
        )
        return round(min_d, 1)

    def _visible_wrists(self, lms):
        pts = []
        for idx in self._LM_WRISTS:
            if idx < len(lms) and (lms[idx].visibility or 0) >= self._MIN_WRIST_VIS:
                pts.append((lms[idx].x * self._fw, lms[idx].y * self._fh))
        return pts

    def _sword_dist(self, weapon_per_fighter):
        line_a = self._first_line(weapon_per_fighter[0])
        line_b = self._first_line(weapon_per_fighter[1])
        if line_a is None or line_b is None:
            return ""
        return round(_seg_dist(line_a, line_b), 1)

    @staticmethod
    def _first_line(dets):
        for det in dets:
            if det.get("line") is not None:
                return det["line"]
        return None

    # ------------------------------------------------------------------

    def _arm_quality(self, lms, prev_lms):
        """Return (min wrist visibility, max wrist jump in torso lengths, arm_ok 0/1)."""
        wrist_vis = min((lms[i].visibility or 0) for i in self._LM_WRISTS)
        elbow_vis = min((lms[i].visibility or 0) for i in self._LM_ELBOWS)
        # Torso length: mid-shoulder to mid-hip, in pixels
        sx = (lms[11].x + lms[12].x) / 2 * self._fw;  sy = (lms[11].y + lms[12].y) / 2 * self._fh
        hx = (lms[23].x + lms[24].x) / 2 * self._fw;  hy = (lms[23].y + lms[24].y) / 2 * self._fh
        torso = max(math.hypot(sx - hx, sy - hy), 1.0)
        jump = ""
        if prev_lms is not None:
            jump = max(
                math.hypot((lms[i].x - prev_lms[i].x) * self._fw, (lms[i].y - prev_lms[i].y) * self._fh)
                for i in self._LM_WRISTS
            ) / torso
        ok = int(wrist_vis >= self._ARM_VIS_OK and elbow_vis >= self._ARM_VIS_OK
                 and (jump == "" or jump <= self._WRIST_JUMP_MAX))
        return round(wrist_vis, 3), (round(jump, 3) if jump != "" else ""), ok

    def _jitter(self, new_lms, prev_lms):
        if prev_lms is None:
            return ""
        total = 0.0
        for n, p in zip(new_lms, prev_lms):
            dx = (n.x - p.x) * self._fw
            dy = (n.y - p.y) * self._fh
            total += (dx ** 2 + dy ** 2) ** 0.5
        return round(total / len(new_lms), 2)

    def write(self):
        if not self._rows:
            return
        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)

    @property
    def path(self):
        return self._path

    def summary(self):
        """Return a dict of aggregate metrics for the end-of-run printout."""
        n = len(self._rows)
        if n == 0:
            return {}

        def mean(vals):
            v = [x for x in vals if x != "" and x is not None]
            return round(sum(v) / len(v), 3) if v else None

        def pct(vals):
            v = [x for x in vals if x != "" and x is not None]
            return round(sum(v) / len(v) * 100, 1) if v else None

        return {
            "a_matched_%":    pct(r["a_matched"]    for r in self._rows),
            "b_matched_%":    pct(r["b_matched"]    for r in self._rows),
            "a_pose_%":       pct(r["a_pose"]       for r in self._rows),
            "b_pose_%":       pct(r["b_pose"]       for r in self._rows),
            "a_vis_mean":     mean(r["a_vis_mean"]  for r in self._rows),
            "b_vis_mean":     mean(r["b_vis_mean"]  for r in self._rows),
            "a_jitter_mean":  mean(r["a_jitter"]    for r in self._rows),
            "b_jitter_mean":  mean(r["b_jitter"]    for r in self._rows),
            "a_weapon_%":     pct(r["a_weapon"]     for r in self._rows),
            "b_weapon_%":     pct(r["b_weapon"]     for r in self._rows),
            # % of frames WITH a pose where the arms are usable
            "a_arm_ok_%":     pct(r.get("a_arm_ok", "") for r in self._rows),
            "b_arm_ok_%":     pct(r.get("b_arm_ok", "") for r in self._rows),
        }


# ---------------------------------------------------------------------------
# Segment geometry — used for sword_dist
# ---------------------------------------------------------------------------

def _pt_to_seg(px, py, x1, y1, x2, y2):
    """Closest distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _segs_intersect(s1, s2):
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2

    def cross2d(ax, ay, bx, by):
        return ax * by - ay * bx

    d1 = cross2d(x4 - x3, y4 - y3, x1 - x3, y1 - y3)
    d2 = cross2d(x4 - x3, y4 - y3, x2 - x3, y2 - y3)
    d3 = cross2d(x2 - x1, y2 - y1, x3 - x1, y3 - y1)
    d4 = cross2d(x2 - x1, y2 - y1, x4 - x1, y4 - y1)
    return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))


def _seg_dist(s1, s2):
    """Minimum pixel distance between two line segments. 0 if they cross."""
    if _segs_intersect(s1, s2):
        return 0.0
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2
    return min(
        _pt_to_seg(x1, y1, x3, y3, x4, y4),
        _pt_to_seg(x2, y2, x3, y3, x4, y4),
        _pt_to_seg(x3, y3, x1, y1, x2, y2),
        _pt_to_seg(x4, y4, x1, y1, x2, y2),
    )
