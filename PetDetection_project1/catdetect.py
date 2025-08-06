import cv2
import numpy as np
import time
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกแล้ว
model = YOLO("runs/detect/food_detect3/weights/best.pt")

# สีของเสื้อ (เฉพาะสีเขียว, ม่วง, ชมพู)
SHIRT_COLORS = {
    "Green Shirt": [(40, 50, 50), (80, 255, 255)],  
    "Purple Shirt": [(125, 50, 50), (145, 255, 255)],  
    "Pink Shirt": [(150, 50, 50), (170, 255, 255)]  
}

# สีปลอกคอ
COLLAR_COLORS = {
    "Pink Collar": [(140, 30, 30), (180, 255, 255)],
    "Blue Collar": [(90, 30, 30), (140, 255, 255)],
    "Green Collar": [(30, 30, 30), (90, 255, 255)],
    "Red Collar": [(0, 30, 30), (10, 255, 255)],
    "Orange Collar": [(10, 50, 50), (25, 255, 255)],
    "Yellow Collar": [(20, 50, 50), (40, 255, 255)]
}

# ตัวแปรล็อคตำแหน่งของชามข้าว
locked_food_bowl = None  

def get_center_mask(hsv_roi):
    """ ใช้เฉพาะส่วนกลางของ bounding box เพื่อลดผลกระทบจากแสงเงา """
    h, w, _ = hsv_roi.shape
    center_x, center_y = w // 2, h // 2
    mask_size = 10  
    return hsv_roi[center_y - mask_size:center_y + mask_size, center_x - mask_size:center_x + mask_size]

def detect_color(hsv_roi, color_ranges):
    """ ตรวจจับสีจาก ROI ตามช่วงสีที่กำหนด """
    detected_color = "Unknown"
    max_pixels = 0

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > max_pixels:
            detected_color = color
            max_pixels = pixel_count

    return detected_color

def detect_shirt_or_collar(frame, box):
    """ ตรวจจับสีเสื้อก่อน ถ้าไม่มีเสื้อให้ตรวจจับสีปลอกคอ """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = frame[y1:y2, x1:x2]

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return "Unknown"

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.GaussianBlur(hsv_roi, (5, 5), 0)
    hsv_center = get_center_mask(hsv_roi)

    # ตรวจจับสีเสื้อก่อน
    shirt_color = detect_color(hsv_center, SHIRT_COLORS)
    
    if shirt_color != "Unknown":
        return shirt_color  

    # ถ้าไม่มีเสื้อ ให้ตรวจจับปลอกคอแทน
    collar_color = detect_color(hsv_center, COLLAR_COLORS)
    return collar_color

def detect_behavior(frame, results):
    global locked_food_bowl

    detected_food_bowls = []
    detected_cats = []
    food_bowl_confidence = {}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])  
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == 2:  # ชามข้าว
                detected_food_bowls.append((x1, y1, x2, y2))
                food_bowl_confidence[(x1, y1, x2, y2)] = conf

            if cls_id == 0:  # แมว
                detected_cats.append((x1, y1, x2, y2, detect_shirt_or_collar(frame, box)))

    # ล็อคตำแหน่งชามข้าวที่มีค่า confidence สูงสุด
    if detected_food_bowls:
        highest_confidence_box = max(food_bowl_confidence, key=food_bowl_confidence.get)
        locked_food_bowl = highest_confidence_box  

    # แสดงกรอบของชามข้าว
    if locked_food_bowl:
        x1, y1, x2, y2 = locked_food_bowl
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Food Bowl", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ตรวจจับแมว
    for x1, y1, x2, y2, detected_color in detected_cats:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Cat ({detected_color})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture("catfront.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# ตรวจสอบแนววิดีโอ
rotate_video = width < height

# สร้าง VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # หมุนวิดีโอถ้าเป็นแนวตั้ง
    #if rotate_video:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    results = model(frame, conf=0.5, iou=0.5)
    detect_behavior(frame, results)

    out.write(frame)
    cv2.imshow("Detection", cv2.resize(frame, (400, 800)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
