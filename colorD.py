import cv2
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLO ที่ฝึกใหม่
model = YOLO("runs/detect/cat_characteristics5/weights/best.pt")  # เปลี่ยนเป็นโมเดลที่คุณฝึกเอง

# ฟังก์ชันสำหรับตรวจจับสีปลอกคอ
COLORS = {
    "Pink": [(145, 50, 50), (170, 255, 255)],
    "Blue": [(100, 50, 50), (130, 255, 255)],
    "Green": [(40, 50, 50), (80, 255, 255)],
    "Red": [(0, 50, 50), (10, 255, 255)],
    "Orange": [(10, 100, 100), (25, 255, 255)],
    "Yellow": [(25, 100, 100), (35, 255, 255)]
}

# ตำแหน่งเฟรมก่อนหน้าเพื่อคำนวณการเคลื่อนไหว
prev_positions = {}

def detect_collar_color(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in COLORS.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 50:
            return color
    return "Unknown"

def detect_behavior(cls_id, x1, y1, x2, y2, prev_positions):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    if cls_id in prev_positions:
        prev_x, prev_y = prev_positions[cls_id]
        movement = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
        prev_positions[cls_id] = (center_x, center_y)
        if movement > 60:  # ปรับค่าการเคลื่อนไหวตามความเหมาะสม
            return "Playing"
        else:
            return "Stay"
    else:
        prev_positions[cls_id] = (center_x, center_y)
        return "Stay"

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture("play.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# สร้าง VideoWriter สำหรับบันทึกผลลัพธ์
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video_behavior.mp4", fourcc, fps, (width, height))

# ตัวแปรสำหรับข้ามช่วงที่ไม่มีการตรวจจับ
frames_without_detection = 0
MAX_FRAMES_SKIP = fps * 5  # ข้ามไม่เกิน 5 วินาที

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ใช้โมเดล YOLO ตรวจจับวัตถุ
    results = model(frame)
    detections_made = False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ตรวจสอบว่าคลาสคืออะไร
            if cls_id == 0:  # แมว
                detections_made = True
                collar_color = detect_collar_color(frame, box)
                behavior = detect_behavior(cls_id, x1, y1, x2, y2, prev_positions)

                # วาดกรอบรอบแมวและระบุสีปลอกคอ + พฤติกรรม
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Cat ({collar_color}, {behavior})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif cls_id == 1:  # ถาดทราย (litter box)
                detections_made = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Litter Box", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            elif cls_id == 2:  # ชามข้าว (food bowl)
                detections_made = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Food Bowl", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if detections_made:
        frames_without_detection = 0
        out.write(frame)  # บันทึกเฟรม
    else:
        frames_without_detection += 1
        if frames_without_detection <= MAX_FRAMES_SKIP:
            out.write(frame)  # บันทึกเฟรมว่างต่อเนื่องในช่วงเวลาที่กำหนด

    # แสดงผลลัพธ์
    cv2.imshow("Cat Detection and Behavior", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
