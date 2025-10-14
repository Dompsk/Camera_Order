# นำเข้าไลบรารีที่จำเป็น
from ultralytics import YOLO
import cv2

# --- 1. โหลดโมเดลที่เทรนเสร็จแล้ว ---
# ✅ สำคัญ: จากรูปที่คุณส่งมา ไฟล์ best.pt ควรจะอยู่ที่ path นี้
# หากไม่เจอ ให้ตรวจสอบในโฟลเดอร์ runs ของคุณอีกครั้ง
model = YOLO(r'D:\ปี3เทอม1\A.I\Camera_Order\best.pt')

# --- 2. เปิดใช้งานกล้อง Webcam ---
# cv2.VideoCapture(0) คือการใช้กล้องหลักของเครื่อง
# ถ้าใช้กล้องอื่นอาจเปลี่ยนเป็น 1, 2, ...
cap = cv2.VideoCapture(0)

# ตรวจสอบว่าเปิดกล้องได้หรือไม่
if not cap.isOpened():
    print("Error: ไม่สามารถเปิดกล้องได้")
    exit()

# --- 3. เริ่มลูปเพื่ออ่านภาพจากกล้องและตรวจจับ ---
while True:
    # อ่านภาพทีละเฟรมจากกล้อง
    # ret จะเป็น True หากอ่านสำเร็จ, frame คือภาพที่อ่านได้
    ret, frame = cap.read()
    if not ret:
        print("Error: ไม่สามารถรับภาพจากกล้องได้")
        break

    # --- 4. ส่งภาพไปให้โมเดลตรวจจับ ---
    # ปรับให้ตรวจจับไวขึ้น (อาจมีผลผิดพลาดบ้าง) แต่กรองกรอบซ้อนได้ดีขึ้น
    results = model(frame, conf=0.65, classes=[0 ,1, 2])

    print(f"Found {len(results[0].boxes)} objects in this frame.")

    # --- 5. แสดงผลลัพธ์บนภาพ ---
    # results[0].plot() เป็นคำสั่งพิเศษของ YOLOv8
    # ที่จะวาดกรอบ, ชื่อคลาส, และค่าความมั่นใจลงบนภาพให้โดยอัตโนมัติ
    annotated_frame = results[0].plot()

    # แสดงภาพที่วาดกรอบแล้วในหน้าต่างใหม่
    cv2.imshow("Real-time Object Detection", annotated_frame)

    # --- 6. รอรับการกดปุ่ม 'q' เพื่อปิดโปรแกรม ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("กำลังปิดโปรแกรม...")
        break

# --- 7. ปิดการใช้งานกล้องและหน้าต่างทั้งหมด ---
cap.release()
cv2.destroyAllWindows()