import cv2
from ultralytics import YOLO

def main():

    # โหลดโมเดลที่ผ่านการฝึก
    model = YOLO(r"D:\learning\Mr.Ruji\AI_YOLO\best.pt")

    # เปิดใช้งานกล้องเว็บแคม
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    print("กด 'q' เพื่อออกจากโปรแกรม")

    while True:

        success, frame = cap.read()

        if success:

            # ตรวจจับด้วย YOLO และใช้ GPU
            results = model.predict(
                source=frame,
                stream=True,
                device=0
            )

            for r in results:
                annotated_frame = r.plot()

            # แสดงภาพ
            cv2.imshow("YOLO26 Real-time Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()