import cv2
import numpy as np
import time
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกแล้ว
model = YOLO("runs/detect/food_detect2/weights/best.pt")

# สีปลอกคอและเสื้อที่ต้องตรวจจับ
COLORS = {
    "Pink": [(155, 50, 50), (180, 255, 255)],
    "Green": [(40, 50, 50), (80, 255, 255)],
    "Red": [(5, 50, 30), (25, 255, 200)],
    "Purple": [(130, 50, 50), (155, 255, 255)]
}

# ตัวแปรล็อคตำแหน่งชามข้าวและแมวที่กำลังกิน
locked_food_bowl = None
locked_cats = {}
cat_last_seen = {}  # บันทึกเวลาเห็นแมวครั้งสุดท้าย

def detect_collar_color(frame, box):
    """ ตรวจจับสีปลอกคอ/เสื้อของแมว """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = frame[y1:y2, x1:x2]

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return "Unknown"

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.GaussianBlur(hsv_roi, (5, 5), 0)

    for color, (lower, upper) in COLORS.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 50:
            return color
    return "Unknown"

def is_near(box1, box2, threshold=180):  # ปรับ threshold เป็น 180
    """ ตรวจสอบระยะห่างของวัตถุทั้งสอง """
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    return distance < threshold

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture("color3.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# ตรวจสอบการหมุนวิดีโอ
rotate_video = width < height

# สร้าง VideoWriter สำหรับบันทึกผลลัพธ์
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video_behavior.mp4", fourcc, fps, (max(width, height), min(width, height)))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # หมุนวิดีโอถ้าจำเป็น
    if rotate_video:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # ตรวจจับวัตถุในเฟรม
    results = model(frame, conf=0.5, iou=0.5)
    detected_cats = []
    detected_food_bowls = []
    food_bowl_confidence = {}

    # ตรวจจับแมวและชามข้าว
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == 2:  # ชามอาหาร
                detected_food_bowls.append((x1, y1, x2, y2))
                food_bowl_confidence[(x1, y1, x2, y2)] = conf

            if cls_id == 0:  # แมว
                collar_color = detect_collar_color(frame, box)
                detected_cats.append((x1, y1, x2, y2, collar_color))

    # ล็อคชามข้าวที่มีค่า confidence สูงสุด
    if detected_food_bowls:
        highest_confidence_bowl = max(food_bowl_confidence, key=food_bowl_confidence.get)
        locked_food_bowl = highest_confidence_bowl

    # แสดงกรอบของชามข้าวที่ล็อคไว้
    if locked_food_bowl:
        x1, y1, x2, y2 = locked_food_bowl
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Food Bowl (Locked)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ตรวจจับแมวที่เข้าไปกินข้าว
    for x1, y1, x2, y2, collar_color in detected_cats:
        cat_box = (x1, y1, x2, y2)

        # ตรวจสอบว่าแมวอยู่ในระยะกินข้าวหรือไม่
        if locked_food_bowl and is_near(cat_box, locked_food_bowl, threshold=300):
            if cat_box not in locked_cats:
                locked_cats[cat_box] = {"start_time": time.time(), "color": collar_color}
            cat_last_seen[cat_box] = time.time()  # อัปเดตเวลาที่แมวถูกเห็นล่าสุด

        # ถ้าแมวถูกล็อค ให้แสดงข้อความ
        if cat_box in locked_cats:
            cv2.putText(frame, f"Cat ({locked_cats[cat_box]['color']}) Eating!", 
                        (locked_food_bowl[0], locked_food_bowl[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # แสดงกรอบของแมว
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Cat ({collar_color})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ตรวจสอบว่าแมวออกจากชามข้าวแล้วหรือยัง
    current_time = time.time()
    to_remove = []
    for cat_box in locked_cats:
        if cat_box not in cat_last_seen or (current_time - cat_last_seen[cat_box] > 2):  # ล็อคแมวไว้อีก 2 วิ
            to_remove.append(cat_box)

    for cat_box in to_remove:
        del locked_cats[cat_box]

    # บันทึกวิดีโอผลลัพธ์
    out.write(frame)

    # แสดงผลลัพธ์
    frame = cv2.resize(frame, (1000, 600))
    cv2.imshow("Cat Detection and Behavior", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
