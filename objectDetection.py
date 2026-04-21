import cv2
from ultralytics import YOLO


MODEL = "yolov8n.engine"
CONFIDENCE = 0.7
# SOURCE = "131232-749706873.mp4"  # 0 = webcam, or JETSON for gstreamer camera source
SOURCE = "JETSON"

# person, bicycle, car, motorcycle, bus, truck
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

COLORS = {
    "person": (0, 255, 0),
    "bicycle": (255, 165, 0),
    "car": (0, 165, 255),
    "motorcycle": (255, 0, 255),
    "bus": (0, 0, 255),
    "truck": (255, 0, 0),
}


def gstreamer_pipeline(
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


def load_camera_source(source=0):
    if source == "JETSON":
        video_capture = cv2.VideoCapture(
            gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    else:
        video_capture = cv2.VideoCapture(source)
    return video_capture


model = YOLO(MODEL)
cap = load_camera_source(SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 360))

    results = model.track(
        frame, 
        classes=TARGET_CLASSES, 
        conf=CONFIDENCE, 
        persist=True, 
        verbose=False,
    	stream=False 
    )[0]

    if results.boxes.id is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            track_id = int(box.id[0])
            color = COLORS.get(label, (0, 255, 0))

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

    cv2.imshow("Detection - Person & Vehicles", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
