
import cv2
import numpy as np
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกแล้ว
model = YOLO("runs/detect/litterbox_detect2/weights/best.pt")  # เปลี่ยน path ให้ตรงกับโมเดลของคุณ


# ฟังก์ชันสำหรับตรวจจับสีปลอกคอ
COLORS = {
    "Pink": [(145, 50, 50), (170, 255, 255)],
    "Blue": [(100, 50, 50), (130, 255, 255)],
    "Green": [(40, 50, 50), (80, 255, 255)],
    "Red": [(0, 50, 50), (10, 255, 255)],
    "Orange": [(10, 100, 100), (25, 255, 255)],
    "Yellow": [(25, 100, 100), (35, 255, 255)]
}

def detect_collar_color(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in COLORS.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 50:
            return color
    return "Unknown"

def is_near(box1, box2, threshold=50):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    return distance < threshold

def detect_behavior(frame, results):
    litter_boxes = []
    cat_boxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == 1:  # Litter Box
                litter_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Litter Box", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if cls_id == 0:  # Cat
                collar_color = detect_collar_color(frame, box)
                cat_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Cat ({collar_color})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ตรวจสอบว่ามีแมวอยู่ใกล้กะบะทรายหรือไม่
    for litter_box in litter_boxes:
        for cat in cat_boxes:
            if is_near(litter_box, cat, threshold=100):
                cv2.putText(frame, "Cat near Litter Box!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture("cats.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# สร้าง VideoWriter สำหรับบันทึกผลลัพธ์
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ใช้โมเดล YOLO ตรวจจับวัตถุ
    results = model(frame, conf=0.4)
    detect_behavior(frame, results)

    # บันทึกวิดีโอ
    out.write(frame)

    # แสดงผลลัพธ์
    frame_resized = cv2.resize(frame, (640, 480))
    cv2.imshow("Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
