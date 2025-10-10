from ultralytics import YOLO
import cv2

# โหลดโมเดลจาก path ที่ถูกต้อง
model = YOLO(r"D:\ปี3เทอม1\A.I\Camera_Order\runs\train\best\weights\best.pt")

# แม็ปราคา
product_prices = {
    "ขนม": 10,
    "ขวดน้ำเล็ก": 5,
    "ขวดน้ำใหญ่": 15,
    "ปากกา": 7
}

# สแกนหากล้องที่ใช้งานได้
cap = None
for i in range(5):
    temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"ใช้กล้องที่ index {i}")
        break

if cap is None:
    print("ไม่พบกล้องที่ใช้งานได้")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("อ่าน frame ไม่ได้")
        break

    # ตรวจจับวัตถุ
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            price = product_prices.get(label, "N/A")

            # วาดกรอบและข้อความ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {price} บาท ({conf:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Product Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
