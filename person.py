from ultralytics import YOLO

_PERSON_CLASS   = 0     # COCO class 0 = 'person'
_CONF_THRESHOLD = 0.50


class PersonDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self._model = YOLO(model_name)

    def detect(self, frame_bgr):
        """
        Plain detection, no tracking state. Used for the seed-frame scan.
        Returns list of [x1, y1, x2, y2].
        """
        results = self._model(frame_bgr, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            if int(box.cls.item()) != _PERSON_CLASS:
                continue
            if float(box.conf.item()) < _CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            boxes.append([x1, y1, x2, y2])
        return boxes

    def track(self, frame_bgr):
        """
        ByteTrack-based tracking. Used in the main processing loop.
        Returns {track_id: [x1, y1, x2, y2]} for all detected persons.
        Track IDs are persistent across frames — the same physical person keeps
        the same ID through movement and brief occlusions.
        """
        results = self._model.track(
            frame_bgr,
            persist=True,
            classes=[_PERSON_CLASS],
            conf=_CONF_THRESHOLD,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        out = {}
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            ids   = results[0].boxes.id.int().tolist()
            boxes = results[0].boxes.xyxy.tolist()
            for tid, box in zip(ids, boxes):
                out[tid] = [int(v) for v in box]
        return out
