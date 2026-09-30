from ultralytics import YOLO

# โหลดโมเดลที่ผ่านการฝึก (Trained Model)
model = YOLO("D:\learning\Mr.Ruji\AI_YOLO/best.pt")

# นำโมเดลไปทดสอบกับรูปภาพ
results = model.predict(
    "FILE_NAME",    # ชื่อไฟล์รูปภาพที่ต้องการทดสอบ เช่น "coffee.jpg"
    conf=0.5,       # กำหนดค่า Confidence ขั้นต่ำที่ 50%
    save=True       # บันทึกภาพผลลัพธ์ที่ตรวจจับได้
)

# แสดงผลลัพธ์การตรวจจับของรูปภาพแรก
results[0].show()