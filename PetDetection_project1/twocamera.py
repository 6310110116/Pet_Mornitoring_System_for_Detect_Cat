import cv2
import torch
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLO ที่เทรนแล้ว
model = YOLO('runs/detect/food_detect/weights/best.pt')
model.eval()

# ฟังก์ชันตรวจจับสีของปลอกคอ (ขึ้น "Null" ถ้ามีหลายสี)
def detect_collar_color(image, bbox):
    x1, y1, x2, y2 = bbox
    collar_region = image[int(y1 + (y2 - y1) * 0.7):int(y1 + (y2 - y1) * 0.85), x1:x2]

    if collar_region.size == 0:
        return "Null"

    hsv = cv2.cvtColor(collar_region, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv, axis=(0, 1))

    # ตรวจจับสี
    color_dict = {
        "Red": (avg_color[0] < 10 or avg_color[0] > 170),
        "Yellow": (10 < avg_color[0] < 35),
        "Green": (35 < avg_color[0] < 85),
        "Blue": (85 < avg_color[0] < 135),
        "Purple": (135 < avg_color[0] < 170)
    }

    detected_colors = [color for color, condition in color_dict.items() if condition]

    # ถ้าพบมากกว่าหนึ่งสี ให้ขึ้น "Null"
    return detected_colors[0] if len(detected_colors) == 1 else "Null"

# ฟังก์ชันคำนวณระดับความมั่นใจ
def determine_confidence(cats_cam1, cats_cam2):
    confidence_levels = {}

    for cat_id1, (collar1, pos1) in cats_cam1.items():
        best_match = None
        min_distance = float('inf')

        for cat_id2, (collar2, pos2) in cats_cam2.items():
            distance = np.linalg.norm(np.array(pos1) - np.array(pos2))  # คำนวณระยะห่าง
            if distance < 150 and distance < min_distance:  # ขยายระยะให้แมวถือว่าเป็นตัวเดียวกัน
                best_match = cat_id2
                min_distance = distance

        if best_match:
            collar2 = cats_cam2[best_match][0]
            if collar1 == collar2 and collar1 != "Null":
                confidence_levels[cat_id1] = ("Very High", pos1)
            elif collar1 == "Null" or collar2 == "Null":
                confidence_levels[cat_id1] = ("High", pos1)
            else:
                confidence_levels[cat_id1] = ("Medium", pos1)
        else:
            confidence_levels[cat_id1] = ("Medium", pos1)

    for cat_id2 in cats_cam2:
        if cat_id2 not in confidence_levels:
            confidence_levels[cat_id2] = ("Medium", cats_cam2[cat_id2][1])

    return confidence_levels

# ฟังก์ชันแสดงผลและบันทึกวิดีโอ
def play_videos_side_by_side(video_path1, video_path2, output_path="output_combined.mp4"):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    target_width = 640
    target_height = 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (target_width * 2, target_height))

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame1 = cv2.resize(frame1, (target_width, target_height))
        frame2 = cv2.resize(frame2, (target_width, target_height))

        cats_cam1 = {}
        cats_cam2 = {}
        seen_collar_colors_cam1 = set()  # สีที่เจอแล้วในกล้อง 1
        seen_collar_colors_cam2 = set()  # สีที่เจอแล้วในกล้อง 2

        # ตรวจจับแมวในกล้อง 1
        results1 = model(frame1, conf=0.5)
        for r in results1:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 0 and conf >= 0.5:
                    collar_color = detect_collar_color(frame1, (x1, y1, x2, y2))

                    # ถ้าสีซ้ำกันในกล้อง 1 ให้ขึ้น "Null"
                    if collar_color in seen_collar_colors_cam1:
                        collar_color = "Null"
                    else:
                        seen_collar_colors_cam1.add(collar_color)

                    cats_cam1[f"cat_{x1}_{y1}"] = (collar_color, (x1, y1))

                    cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame1, f"Cat: {conf:.2f}, Collar: {collar_color}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ตรวจจับแมวในกล้อง 2
        results2 = model(frame2, conf=0.5)
        for r in results2:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 0 and conf >= 0.5:
                    collar_color = detect_collar_color(frame2, (x1, y1, x2, y2))

                    # ถ้าสีซ้ำกันในกล้อง 2 ให้ขึ้น "Null"
                    if collar_color in seen_collar_colors_cam2:
                        collar_color = "Null"
                    else:
                        seen_collar_colors_cam2.add(collar_color)

                    cats_cam2[f"cat_{x1}_{y1}"] = (collar_color, (x1, y1))

                    cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame2, f"Cat: {conf:.2f}, Collar: {collar_color}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # คำนวณระดับความมั่นใจ
        confidence_levels = determine_confidence(cats_cam1, cats_cam2)

        for cat_id, (confidence, pos) in confidence_levels.items():
            if pos is not None and isinstance(pos, tuple) and len(pos) == 2:
                x, y = pos
                color = (0, 255, 255) if confidence == "Very High" else (0, 165, 255) if confidence == "High" else (0, 0, 255)
                cv2.putText(frame1, confidence, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        combined_frame = np.hstack((frame1, frame2))
        cv2.imshow("Cat Detection", combined_frame)
        out.write(combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

play_videos_side_by_side("catfront.mp4", "catfront1.mp4", "output_combined.mp4")
