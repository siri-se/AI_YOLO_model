from ultralytics import YOLO

model = YOLO("Model/best.pt")

results = model.predict(
    "test_single_photo/img_1.png",      
    conf=0.3,
    save=True,
    project="test",
    name="single",
)

# พิมพ์ผลให้ดู
for box in results[0].boxes:
    print(model.names[int(box.cls)], f"{float(box.conf):.2f}")