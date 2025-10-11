from ultralytics import YOLO
import os

# --- 1. การตั้งค่า ---
# โหลดโมเดล YOLOv8 ขนาดเริ่มต้น (n = nano, เป็นขนาดที่เล็กและเร็วที่สุด)
# โมเดลจะถูกดาวน์โหลดมาโดยอัตโนมัติหากยังไม่มี
model = YOLO('yolov8n.pt')

# --- 2. เริ่มต้นการเทรนโมเดล ---
if __name__ == '__main__':
    # สั่งให้โมเดลเริ่มเรียนรู้จากข้อมูลของคุณ
    # data='data.yaml' -> บอกโมเดลให้ไปอ่านการตั้งค่าจากไฟล์ data.yaml
    # epochs=150 -> จำนวนรอบที่จะให้โมเดลเรียนรู้ข้อมูลทั้งหมด (ปรับเพิ่มได้)
    # imgsz=640 -> ปรับขนาดรูปภาพเป็น 640x640 pixels ก่อนนำไปเทรน
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640
    )

    print("\n✅ Training complete for Object Detection!")
    
    # พิมพ์ที่อยู่ของไฟล์โมเดลที่ดีที่สุดที่ถูกบันทึกไว้
    # โดยทั่วไปจะอยู่ในโฟลเดอร์ runs/detect/train/weights/best.pt
    # ใช้ os.path.abspath เพื่อแสดงผลเป็น path แบบเต็ม
    final_model_path = os.path.abspath(results.save_dir)
    print(f"📁 Your trained model is saved in: {final_model_path}")