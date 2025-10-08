from ultralytics import YOLO
import cv2

# โหลดโมเดลที่เทรนไว้
model = YOLO("best.pt")  # โมเดลที่คุณเทรน เช่น ขวด, ขนม, กล่องนม ฯลฯ

# แม็ปราคา
product_prices = {
    "ขวดน้ำ": 10,
    "ขนม": 5,
    "สบู่": 20
}

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับวัตถุในเฟรม
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]  # เช่น "ขวดน้ำ"
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # หาราคา
            price = product_prices.get(label, "N/A")

            # วาดกรอบและข้อความ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label}: {price} บาท",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Product Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
