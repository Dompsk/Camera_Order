from ultralytics import YOLO

# สร้างโมเดลใหม่ (จาก yolov8n.pt)
model = YOLO('yolov8n.pt')  # ใช้ขนาดเล็กสำหรับเริ่มต้น

# เทรนโมเดลกับ dataset ของเรา
model.train(
    data='dataset.yaml',  # path ไปยังไฟล์ dataset config
    epochs=50,            # จำนวน epoch ที่ต้องการเทรน
    imgsz=640,            # ขนาดภาพ
    batch=8               # batch size
)
