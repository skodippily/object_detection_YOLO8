from ultralytics import YOLO
from collections import defaultdict, deque
import cv2
import numpy as np


class ApproachDetector:
    def __init__(self,
                 history_len=5,
                 growth_threshold=0.04,
                 min_consecutive=3):

        self.area_history = defaultdict(
            lambda: deque(maxlen=history_len)
        )

        self.growth_count = defaultdict(int)

        self.growth_threshold = growth_threshold
        self.min_consecutive = min_consecutive

    def area(self, box):
        x1, y1, x2, y2 = box
        return max(1, (x2-x1)*(y2-y1))

    def update_track(self, track_id, box):

        A = self.area(box)
        hist = self.area_history[track_id]

        approaching = False

        if len(hist) > 0:

            prevA = hist[-1]

            growth = (A - prevA)/prevA

            if growth > self.growth_threshold:
                self.growth_count[track_id] += 1
            else:
                self.growth_count[track_id] = max(
                    0,
                    self.growth_count[track_id]-1
                )

            if self.growth_count[track_id] >= self.min_consecutive:
                approaching = True

        hist.append(A)

        return approaching


class YOLOTracker:
    def __init__(
        self,
        model_path="yolov8n.pt",
        source=0,
        confidence=0.5,
        target_classes=None,
    ):
        # Config
        self.model_path = model_path
        self.source = source
        self.confidence = confidence
        self.target_classes = target_classes or [0, 1, 2, 3, 5, 7]

        # Colors
        self.colors = {
            "person": (0, 255, 0),
            "bicycle": (255, 165, 0),
            "car": (0, 165, 255),
            "motorcycle": (255, 0, 255),
            "bus": (0, 0, 255),
            "truck": (255, 0, 0),
        }

        # Load model
        self.model = YOLO(self.model_path)
        self.approach_detector = ApproachDetector()

        # Load camera
        self.cap = self.load_camera_source(self.source)

        self.crowded_threshold = 5

    def gstreamer_pipeline(
        self,
        sensor_id=0,
        capture_width=1920,
        capture_height=1080,
        display_width=640,
        display_height=320,
        framerate=60,
        flip_method=0,
    ):
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

    def load_camera_source(self, source):
        if source == "JETSON":
            return cv2.VideoCapture(
                self.gstreamer_pipeline(flip_method=2),
                cv2.CAP_GSTREAMER
            )
        return cv2.VideoCapture(source)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 360))

        results = self.model.track(
            frame,
            classes=self.target_classes,
            conf=self.confidence,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            stream=False
        )[0]

        return frame, results

    def draw_detections(self, frame, results):
        if results.boxes.id is None:
            return frame

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            track_id = int(box.id[0])

            approaching = self.approach_detector.update_track(
                track_id,
                (x1, y1, x2, y2)
            )
            if approaching:
                label += " APPROACHING"

            color = self.colors.get(label, (0, 255, 0))
            display = f"{label} {track_id} ({conf:.0%})"

            print(f"Detected: {display}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                display,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return frame

    def draw_density_heatmap(
            self,
            frame,
            results,
            alpha=0.35,
            center_weight=True,
            sigma=8,
            radius=80):
        """
        Density heatmap using Gaussian blobs at object centers.
        """

        h, w = frame.shape[:2]

        density = np.zeros(
            (h, w),
            dtype=np.float32
        )

        img_center = w/2

        for (x1, y1, x2, y2) in results.boxes.xyxy.int().tolist():

            # ---------------------
            # choose center point
            # bottom-center works well for road objects
            # ---------------------
            bx = int((x1+x2)/2)
            by = int(y2)
            bwidth = x2-x1
            bheight = y2-y1

            if (
                bx < 0 or bx >= w or
                by < 0 or by >= h
            ):
                continue

            # ---------------------
            # weight
            # ---------------------
            area = (x2-x1)*(y2-y1)

            weight = area/(w*h)

            if center_weight:

                center_factor = 1 / (
                    1 + abs(
                        bx-img_center
                    )/150
                )

                weight *= center_factor

            # ---------------------
            # Gaussian blob
            # ---------------------
            for yy in range(-bheight, bheight+1):
                for xx in range(-bwidth, bwidth+1):

                    px = bx + xx
                    py = by + yy

                    if (
                        0 <= px < w and
                        0 <= py < h
                    ):

                        g = np.exp(
                            -(
                                xx**2 + yy**2
                            )/(2*sigma**2)
                        )

                        density[py, px] += (
                            weight*g
                        )

        # ---------------------
        # normalize
        # ---------------------
        if density.max() > 0:
            density /= density.max()

        heat = (
            density*255
        ).astype(np.uint8)

        heat_color = cv2.applyColorMap(
            heat,
            cv2.COLORMAP_JET
        )

        blended = cv2.addWeighted(
            frame,
            1-alpha,
            heat_color,
            alpha,
            0
        )

        return blended

    def getResults(self):
        if self.results.boxes.id is None:
            return None

        results_dict = []

        for box in self.results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            track_id = int(box.id[0])

            approaching = self.approach_detector.update_track(
                track_id,
                (x1, y1, x2, y2)
            )

            results_dict.append({
                "id": track_id,
                "class": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "approaching": approaching
            })

        return results_dict

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, self.results = self.process_frame(frame)
            frame = self.draw_detections(frame, self.results)
            frame = self.draw_density_heatmap(frame, self.results)

            cv2.imshow("Detection - Person & Vehicles", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()


if __name__ == "__main__":
    tracker = YOLOTracker(
        model_path="yolov8n.pt",
        source=0,  # or "JETSON"
        confidence=0.7
    )
    tracker.run()
