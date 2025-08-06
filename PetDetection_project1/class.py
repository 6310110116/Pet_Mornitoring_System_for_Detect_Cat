from ultralytics import YOLO

# โหลดโมเดล
model = YOLO("yolov5s.pt")  # หรือชื่อโมเดลที่คุณฝึก

# ดูชื่อคลาสทั้งหมด
print(model.names)
results = model(frame, conf=0.5, classes=[0, 1])  # 0=Cat, 1=Litter Box
