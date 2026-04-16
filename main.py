import cv2
from ultralytics import YOLO


MODEL = "yolov8n.pt"
CONFIDENCE = 0.5
# SOURCE = "131232-749706873.mp4"  # 0 = webcam, or video file path
SOURCE = 0

TARGET_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

COLORS = {
    "person": (0, 255, 0),
    "bicycle": (255, 165, 0),
    "car": (0, 165, 255),
    "motorcycle": (255, 0, 255),
    "bus": (0, 0, 255),
    "truck": (255, 0, 0),
}

model = YOLO(MODEL)
cap = cv2.VideoCapture(SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame, classes=TARGET_CLASSES, conf=CONFIDENCE, persist=True, verbose=False
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
