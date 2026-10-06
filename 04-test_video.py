from ultralytics import YOLO

model = YOLO("Model/best.pt")

video_to_test = "test_video/video_1.mp4"   # แก้ชื่อไฟล์ให้ตรง

results = model.predict(
    source=video_to_test,
    conf=0.3,
    device=0,
    save=True,
    project="test",
    name="video_result",
)

print("เสร็จแล้ว ผลอยู่ที่ runs/detect/test/video_result")