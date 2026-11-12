import cv2


class VideoReader:
    def __init__(self, path):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self):
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def reset(self):
        """Seek back to frame 0 so frames() can be iterated again."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class VideoWriter:
    def __init__(self, path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise ValueError(f"Cannot open output video: {path}")

    def write(self, frame):
        self._writer.write(frame)

    def close(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
