import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading MediaPipe pose model (~26 MB) to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")


class _Landmark:
    """
    Full-frame NormalizedLandmark. Mirrors the mediapipe NormalizedLandmark duck type
    so overlay.py and weapon.py can treat both the same way.
    """
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class PoseDetector:
    """
    Wraps MediaPipe PoseLandmarker for a single tracked subject.

    Runs in VIDEO mode so MediaPipe's internal Kalman-like temporal filter
    smooths landmark positions across frames. One instance per fighter —
    each maintains its own temporal state so the two streams don't interfere.
    """

    def __init__(self, num_poses=2):
        _ensure_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker   = mp_vision.PoseLandmarker.create_from_options(options)
        self._prev_centroid = None  # full-frame pixel (x, y) of the last returned pose

    def detect_in_box(self, frame_bgr, box, timestamp_ms):
        """
        Run pose detection inside the bounding box [x1, y1, x2, y2].

        timestamp_ms must be a strictly monotonically increasing integer
        (milliseconds) — required by VIDEO mode for temporal smoothing.

        Crops the frame to the box (after a 10% inward shrink to reduce edge
        inclusion of adjacent people), runs the landmarker, then translates
        normalised crop coordinates back to full-frame coordinates.

        Returns the best detected pose as a list of 33 _Landmark objects, or []
        if no pose is found.
        """
        bx1, by1, bx2, by2 = box
        fh, fw = frame_bgr.shape[:2]

        # Fixed-size crop centered on the tracker box centroid.
        # Consistent geometry is critical for VIDEO-mode temporal smoothing —
        # a changing crop size looks like landmark movement to the Kalman filter.
        # Height = 70% of frame height; width = 55% of frame height (portrait ratio).
        cx    = (bx1 + bx2) // 2
        cy    = (by1 + by2) // 2
        half_h = int(fh * 0.35)   # half of 70%
        half_w = int(fh * 0.275)  # half of 55%

        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(fw, cx + half_w)
        y2 = min(fh, cy + half_h)
        if x2 <= x1 or y2 <= y1:
            return []

        crop    = frame_bgr[y1:y2, x1:x2]
        rgb     = np.ascontiguousarray(crop[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        except Exception:
            return []

        if not result.pose_landmarks:
            return []

        cw = x2 - x1
        ch = y2 - y1

        # When multiple poses detected in the crop, pick the right one.
        #
        # Strategy:
        #   - If we have a previous centroid (all frames after frame 1): use
        #     continuity — pick the pose whose centroid is nearest to where this
        #     fighter was last frame. An occluder walking into the crop always
        #     enters from a distance; Fighter B's centroid is nearby.
        #   - On the very first multi-pose frame (no history): fall back to
        #     spatial scoring with an out-of-box penalty.
        if len(result.pose_landmarks) == 1:
            chosen = result.pose_landmarks[0]
        else:
            # If prev_centroid is outside the current crop the tracker box has
            # jumped (identity swap or re-assignment).  Reset it so spatial
            # scoring re-anchors to whoever is inside the new tracker box.
            if self._prev_centroid is not None:
                px, py = self._prev_centroid
                if not (x1 <= px <= x2 and y1 <= py <= y2):
                    self._prev_centroid = None

            if self._prev_centroid is not None:
                # Continuity: nearest to last frame's centroid.
                # An occluder entering from outside always starts far away.
                chosen, best_dist = None, float('inf')
                for pose in result.pose_landmarks:
                    ff_xs = [lm.x * cw + x1 for lm in pose]
                    ff_ys = [lm.y * ch + y1 for lm in pose]
                    pcx   = sum(ff_xs) / len(ff_xs)
                    pcy   = sum(ff_ys) / len(ff_ys)
                    d     = ((pcx - self._prev_centroid[0]) ** 2 +
                             (pcy - self._prev_centroid[1]) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist, chosen = d, pose
            else:
                # No history: spatial score + out-of-box penalty
                diag  = (fw ** 2 + fh ** 2) ** 0.5
                tb_cx = (bx1 + bx2) / 2.0
                tb_cy = (by1 + by2) / 2.0
                chosen, best_score = None, float('-inf')
                for pose in result.pose_landmarks:
                    ff_xs = [lm.x * cw + x1 for lm in pose]
                    ff_ys = [lm.y * ch + y1 for lm in pose]
                    span  = (max(ff_xs) - min(ff_xs)) + (max(ff_ys) - min(ff_ys))
                    pcx   = sum(ff_xs) / len(ff_xs)
                    pcy   = sum(ff_ys) / len(ff_ys)
                    dist  = ((pcx - tb_cx) ** 2 + (pcy - tb_cy) ** 2) ** 0.5
                    in_box  = (bx1 <= pcx <= bx2) and (by1 <= pcy <= by2)
                    penalty = 0.0 if in_box else diag * 0.5
                    score   = (span - dist - penalty) / diag
                    if score > best_score:
                        best_score, chosen = score, pose

        translated = []
        for lm in chosen:
            translated.append(_Landmark(
                x=(lm.x * cw + x1) / fw,
                y=(lm.y * ch + y1) / fh,
                z=lm.z,
                visibility=lm.visibility if lm.visibility is not None else 0.0,
            ))

        # Update centroid for next frame's continuity check
        self._prev_centroid = (
            sum(lm.x * fw for lm in translated) / len(translated),
            sum(lm.y * fh for lm in translated) / len(translated),
        )
        return translated

    def close(self):
        self._landmarker.close()
