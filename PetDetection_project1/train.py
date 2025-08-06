from ultralytics import YOLO

# โหลดไฟล์ YAML ของข้อมูล และโมเดลที่ต้องการเทรน
model = YOLO('yolov5s.pt')  # หรือ path ไปยังโมเดลอื่นที่คุณมี

# เริ่มการเทรน
model.train(
    data='data.yaml',  # ไฟล์ YAML ที่กำหนดข้อมูล
    imgsz=640,             # ขนาดของภาพ (640x640)
    epochs=5    ,             # จำนวนรอบการเทรน
    batch=16,              # จำนวนภาพต่อ batch
    name='food_detect'  # ชื่อผลลัพธ์การเทรน
)
