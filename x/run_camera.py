import cv2
from ultralytics import YOLO

# โหลดโมเดล
model = YOLO("runs/detect/train2/weights/best.pt")

# เปิดกล้อง (ลองปรับค่า index ถ้าไม่เห็นภาพ)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # ตรวจจับวัตถุ
    results = model(frame, stream=True)

    # วาดกรอบและแสดงผล
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("Custom Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
