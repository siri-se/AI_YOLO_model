from ultralytics import YOLO

# โหลดโมเดลที่ผ่านการฝึก (Trained Model)
model = YOLO("best.pt")

# กำหนดชื่อไฟล์วิดีโอที่ต้องการนำมาทดสอบ
video_to_test = "FILE_NAME"

# นำโมเดลไปทดสอบกับวิดีโอ
results = model.predict(
    source=video_to_test,   # กำหนดไฟล์วิดีโอที่ต้องการทดสอบ
    save=True,              # บันทึกวิดีโอผลลัพธ์หลังจากตรวจจับวัตถุ
    show=True,              # แสดงผลการตรวจจับวัตถุแบบเรียลไทม์ขณะประมวลผล
    conf=0.5                # กำหนดค่า Confidence ขั้นต่ำที่ 50%
)

# แสดงข้อความเมื่อการทดสอบเสร็จสิ้น
# วิดีโอผลลัพธ์จะถูกบันทึกไว้ในโฟลเดอร์ runs/detect/predict
print("ทดสอบเสร็จสิ้น! สามารถเข้าดูวิดีโอผลลัพธ์ได้ที่โฟลเดอร์ runs/detect/predict")