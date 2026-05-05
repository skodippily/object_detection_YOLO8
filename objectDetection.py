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
        model_path="yolov8n.engine",
        source="JETSON",  # 0 or "JETSON"
        confidence=0.5,
        target_classes=None,
        imgsz=320,
        heatmap_enabled=True,
        heatmap_every_n=1,
        heatmap_sigma=18,
        jetson_mode=False,
    ):
        # Config
        self.model_path = model_path
        self.source = source
        self.confidence = confidence
        self.target_classes = target_classes or [0, 1, 2, 3, 5, 7]
        self.imgsz = imgsz
        self.heatmap_enabled = heatmap_enabled
        self.heatmap_every_n = max(1, heatmap_every_n)
        self.heatmap_sigma = heatmap_sigma
        self.jetson_mode = jetson_mode or source == "JETSON"
        self.frame_count = 0
        self.densities = {"total_density": 0.0, "avg_density": 0.0}

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

        self.results = None

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
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 360))

        results = self.model.track(
            frame,
            classes=self.target_classes,
            conf=self.confidence,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            stream=False,
            imgsz=self.imgsz,
            half=self.jetson_mode
        )[0]

        return frame, results

    def draw_density_heatmap(
            self,
            frame,
            results,
            alpha=0.35,
            center_weight=True,
            sigma=18):
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

            # choose center point but bottom-center works well for road objects
            bx = int((x1+x2)/2)
            by = int((y1+y2)/2)
            bwidth = x2-x1
            bheight = y2-y1

            if (
                bx < 0 or bx >= w or
                by < 0 or by >= h
            ):
                continue

            # weight
            area = (x2-x1)*(y2-y1)
            weight = area/(w*h)

            if center_weight:
                center_factor = 1 / (
                    1 + abs(
                        bx-img_center
                    )/(0.25*w)
                )
                weight *= center_factor

            density[by, bx] += weight

        # Blur once instead of doing a Gaussian loop per pixel per object.
        density = cv2.GaussianBlur(
            density,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE
        )

        total_density = density.sum()
        avg_density = density.mean()

        # normalize
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

        return blended, {
            "total_density": total_density,
            "avg_density": avg_density
        }

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
            color = self.colors.get(label, (0, 255, 0))
            if approaching:
                label += " APPROACHING"

            display = f"{label} {track_id} ({conf:.0%}) Approching:{approaching}"

            # print(f"Detected: {display}")

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

    def getResults(self, frame, crowded_threshold=0.0005):
        frame, results = self.process_frame(frame)

        if results.boxes.id is None:
            return frame, None

        bbox_dict = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            track_id = int(box.id[0])

            approaching = self.approach_detector.update_track(
                track_id,
                (x1, y1, x2, y2)
            )
            color = self.colors.get(label, (0, 255, 0))
            if approaching:
                label += " APPROACHING"

            bbox_dict.append({
                "id": track_id,
                "class": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "approaching": approaching
            })

            display = f"{label} {track_id} ({conf:.0%}) Approching:{approaching}"

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

        if self.heatmap_enabled and (
            self.frame_count % self.heatmap_every_n == 0
        ):
            frame, self.densities = self.draw_density_heatmap(
                frame,
                results,
                sigma=self.heatmap_sigma
            )
        # traffic 150
        results_dict = {
            "objects": bbox_dict,
            "approches": any(obj["approaching"] for obj in bbox_dict),
            "traffic_density": self.densities["total_density"],
            "avg_density": self.densities["avg_density"],
            "crowded": self.densities["avg_density"] > crowded_threshold
        }

        return frame, results_dict

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def getFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame_count += 1

        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            frame, self.results = self.process_frame(frame)
            frame = self.draw_detections(frame, self.results)

            if self.heatmap_enabled and (
                self.frame_count % self.heatmap_every_n == 0
            ):
                frame, self.densities = self.draw_density_heatmap(
                    frame,
                    self.results,
                    sigma=self.heatmap_sigma
                )

            cv2.imshow("Camera frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()


if __name__ == "__main__":
    tracker = YOLOTracker(
        model_path="yolov8n.engine",
        source="JETSON",  # 0 or "JETSON"
        confidence=0.7,
        jetson_mode=False,
        imgsz=320,
        heatmap_enabled=True,
        heatmap_every_n=1,
        heatmap_sigma=14
    )
    tracker.run()
