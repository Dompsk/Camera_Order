import tensorflow as tf

# ============================
# 1️⃣ โหลดข้อมูล train / val
# ============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    r"D:\ปี3เทอม1\A.I\Camera_Order\dataset\train",
    image_size=(128, 128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    r"D:\ปี3เทอม1\A.I\Camera_Order\dataset\val",
    image_size=(128, 128),
    batch_size=32
)

# ดูชื่อ class ที่ TensorFlow ตรวจเจอ (ตรวจว่าตรงกับ 4 หมวดไหม)
class_names = train_ds.class_names
print("Classes detected:", class_names)

# ============================
# 2️⃣ สร้างโมเดล CNN แบบง่าย
# ============================
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # 4 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================
# 3️⃣ ฝึกโมเดล (train)
# ============================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15  # เพิ่มรอบเทรนได้ถ้า dataset ไม่เยอะ
)

# ============================
# 4️⃣ บันทึกโมเดลไว้ใช้งานภายหลัง
# ============================
model.save(r"D:\ปี3เทอม1\A.I\Camera_Order\model.h5")

print("✅ Training finished! Model saved successfully.")
print("Model saved to: D:\\ปี3เทอม1\\A.I\\Camera_Order\\model.h5")
