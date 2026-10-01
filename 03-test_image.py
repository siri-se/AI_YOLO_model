from ultralytics import YOLO

# โหลดโมเดลที่ผ่านการฝึก (Trained Model)
model = YOLO("Model/best.pt")

# นำโมเดลไปทดสอบกับรูปภาพ
results = model.predict(
    "test_photos",    # ชื่อไฟล์รูปภาพที่ต้องการทดสอบ เช่น "coffee.jpg"
    conf=0.35,       # กำหนดค่า Confidence ขั้นต่ำที่ 50%
    save=True,       # บันทึกภาพผลลัพธ์ที่ตรวจจับได้
    project="test",    # เก็บผลที่ test/
    name="photos_conf035",
)

# แสดงผลลัพธ์การตรวจจับของรูปภาพแรก
results[0].show()