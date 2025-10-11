from ultralytics import YOLO

# โหลดโมเดลที่เคยเทรนไว้
model = YOLO("runs/detect/train2/weights/best.pt")

# เทรนต่อจากโมเดลเดิม
model.train(
    data="dataset.yaml",  # path ไปยังไฟล์ data.yaml
    epochs=100,                    # จำนวนรอบการเทรนเพิ่ม
    imgsz=640                       # ขนาดภาพ
)
