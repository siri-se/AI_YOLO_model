# ตรวจจับและจำแนกกระป๋องน้ำอัดลม 3 ยี่ห้อด้วย YOLO26

โปรเจกต์ตรวจจับและจำแนกกระป๋องน้ำอัดลม 3 ยี่ห้อ (**ARABUS**, **DMALT**, **SPONSOR**) ด้วย **YOLO26** จากรูปภาพ วิดีโอ และกล้อง Real-time
ใช้ [Ultralytics YOLO](https://docs.ultralytics.com/) สำหรับเทรนและทดสอบ และ [Label Studio](https://labelstud.io/) สำหรับ label รูป

## ผลลัพธ์โดยย่อ

### ผลบน Validation set (89 กรอบ)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | 0.940 | 0.932 | 0.978 | 0.842 |
| ARABUS | 0.940 | 0.921 | 0.981 | 0.799 |
| DMALT | 0.911 | 0.909 | 0.963 | 0.860 |
| SPONSOR | 0.970 | 0.965 | 0.989 | 0.866 |

### ผลบน Test set (75 รูปที่กันไว้ ไม่เคยเห็นตอนเทรน)

นับระดับรูปจาก log (ปรับค่า `conf` ใน `03-test_image.py`)

| ค่า conf | ตรวจเจอ | ไม่เจอ |
|---|---|---|
| 0.5 | 69 / 75 | 6 |
| 0.25 | 72 / 75 | 3 |

ทายยี่ห้อถูกทุกรูปที่ตรวจเจอ (นับจาก log)

<!-- ใส่ภาพตัวอย่างผลทดสอบ 1-2 ภาพ เช่น ![ตัวอย่างผลทดสอบ](images/test_example_1.png) -->

---

## Project Structure

```text
AI_YOLO_model/
├── 01-export_dataset.py   # แปลงผลจาก Label Studio เป็น YOLO dataset
├── 02-train.py            # เทรนโมเดล
├── 03-test_image.py       # ทดสอบกับรูปภาพ
├── 04-test_video.py       # ทดสอบกับวิดีโอ
├── 05-test-camera.py      # ทดสอบกับกล้อง Real-time
├── check.py               # เช็กว่า PyTorch ใช้ GPU (CUDA) ได้
├── prepare_photos.py      # ย่อรูป ตั้งชื่อ และแยกรูปทดสอบ
├── data.yml               # config dataset สำหรับเทรน
├── requirements.txt
├── dataset/
│   ├── classes.txt
│   ├── data.yaml
│   └── labels/            # label แบบ YOLO (train / val)
└── README.md
```

โฟลเดอร์ที่สร้างตอนใช้งาน (ไม่ได้อยู่ใน repo)

```text
raw_photos/<ยี่ห้อ>/       # รูปต้นฉบับที่ถ่ายมา
frame/images/              # รูปที่ย่อแล้ว สำหรับ label (675 รูป)
test_photos/               # รูปทดสอบที่กันไว้ (75 รูป)
dataset/images/            # รูป train / val ที่ export แล้ว
Model/best.pt              # โมเดลที่เทรนเสร็จ (ดูลิงก์ท้ายไฟล์)
```

**Classes** (เรียงตามตัวอักษร ตรงกับ `classes.txt`)

```text
0: ARABUS
1: DMALT
2: SPONSOR
```

---

## Installation

### 1) สร้าง Virtual Environment

```bash
python -m venv env
# cmd
env\Scripts\activate
# PowerShell
.\env\Scripts\Activate.ps1
```

### 2) ติดตั้ง PyTorch แบบ CUDA

เลือกคำสั่งให้ตรงกับเวอร์ชัน CUDA ของเครื่องจาก https://pytorch.org/get-started/locally/

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

(แทน `cuXXX` ด้วยเวอร์ชันที่ใช้จริง)

### 3) ติดตั้งไลบรารี

```bash
pip install -r requirements.txt
```

`requirements.txt` รวม `ultralytics`, `label-studio`, `opencv-python`, `pillow` ไว้แล้ว
ถ้าต้องการรองรับรูป HEIC จาก iPhone ให้ติดตั้งเพิ่ม: `pip install pillow-heif`

### 4) เช็กว่าใช้ GPU ได้

```bash
python check.py
```

ถ้าพิมพ์ `True` และชื่อการ์ดจอ แสดงว่าพร้อมใช้งาน

---

## ขั้นตอนทำงาน (Pipeline)

เตรียมรูป → Label → Export → Train → Test

### 1) เตรียมรูป

1. ถ่ายรูปยี่ห้อละ **250 รูป** เก็บใน `raw_photos/<ยี่ห้อ>/` (ชื่อโฟลเดอร์ใช้เป็นชื่อไฟล์ เช่น `arabus`, `dmalt`, `sponsor`)
2. รัน

   ```bash
   python prepare_photos.py
   ```

   สคริปต์จะ
   - ย่อรูปให้ด้านยาวไม่เกิน 1280 px (และหมุนรูปตาม EXIF)
   - ตั้งชื่อไฟล์สำหรับ label เป็น `arabus_001.jpg`, `arabus_002.jpg`, ... ไว้ที่ `frame/images/`
   - สุ่มแยกรูปทดสอบ **25 รูปต่อยี่ห้อ** (75 รูป) เป็น `arabus_test_001.jpg` ... ไว้ที่ `test_photos/`
3. ผลลัพธ์: `frame/images/` มี **675 รูป** สำหรับ label

> **ข้อควรระวัง:** สคริปต์จะหยุดถ้า `frame/images` หรือ `test_photos` ไม่ว่าง อย่ารันซ้ำหลังเริ่ม label เพราะชื่อไฟล์จะไม่ตรงกับที่ label ไว้

### 2) Label ด้วย Label Studio

ตั้งค่า environment และเปิด Label Studio (ใช้ตามเครื่องที่ใช้)

PowerShell:

```powershell
$env:NLTK_DISABLE_IMPORT_SECURITY = "1"
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "YOUR_PATH\AI_YOLO"
label-studio start
```

cmd:

```bat
set NLTK_DISABLE_IMPORT_SECURITY=1
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=YOUR_PATH\AI_YOLO
label-studio start
```

ขั้นตอน:

1. สร้าง Project ใหม่ ตั้ง 3 class ด้วย Labeling Interface ด้านล่าง
2. เชื่อม **Local Files** ไปที่ `frame/images`
3. label ประมาณ **150 รูปต่อยี่ห้อ** (รวม 450 รูป) ให้สมดุลทุก class
4. Export เป็น **JSON** (ไม่ใช่ JSON-MIN) วางไว้ในโฟลเดอร์เดียวกับ `01-export_dataset.py` ให้เหลือไฟล์ `.json` เพียงไฟล์เดียว

**Labeling Interface:**

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="ARABUS" background="#FF0000"/>
    <Label value="DMALT" background="#00AA00"/>
    <Label value="SPONSOR" background="#0000FF"/>
  </RectangleLabels>
</View>
```

**กติกาการวาดกรอบ:**

- กรอบชิดตัวกระป๋อง
- วาดให้ครบทุกใบในภาพ
- กด **Submit** ทุกรูป

### 3) Export เป็น YOLO Dataset

แก้ path ด้านบนของ `01-export_dataset.py` ให้ตรงกับเครื่อง

```python
IMAGES_DIR = Path(r"YOUR_PATH\AI_YOLO\frame\images")  # โฟลเดอร์รูปที่ label
OUTPUT_DIR = Path(r"YOUR_PATH\AI_YOLO\dataset")       # ที่เก็บ dataset ที่ได้
TRAIN_SPLIT = 0.8
```

แล้วรัน

```bash
python 01-export_dataset.py
```

ข้อความที่ควรเห็น เช่น

```text
Found 3 classes: ['ARABUS', 'DMALT', 'SPONSOR']
Train: 359 images
Val:   91 images
```

แบ่ง Train/Val แบบสุ่มสัดส่วน 80/20 (`SEED = 42`) รูป 3 รูปถูกข้ามเพราะกรอบเกินขอบภาพ

### 4) Training

แก้ `path:` ใน `data.yml` ให้ชี้ไปที่โฟลเดอร์ `dataset` ของเครื่องคุณ แล้วรัน

```bash
python 02-train.py
```

ค่าที่ใช้จริงใน `02-train.py`:

| ค่า | ที่ใช้ |
|---|---|
| โมเดล | `yolo26n.pt` |
| epochs / patience | 200 / 30 |
| imgsz | 640 |
| batch / workers | 8 / 2 (ลดเพราะ GTX 1650 4GB) |
| optimizer | MuSGD |
| device | 0 (GPU) |
| degrees / shear / perspective | 15.0 / 5.0 / 0.001 |
| fliplr / flipud | 0.5 / 0.0 |
| mosaic / mixup | 1.0 / 0.1 |
| close_mosaic | 10 |
| ชื่อผลลัพธ์ (`name`) | `can_v` |

เวลาเทรนราว **0.92 ชั่วโมง** บน GTX 1650 (4GB)
ผลอยู่ที่ `runs/detect/can_v/weights/best.pt` แล้ว copy ไปที่ `Model/best.pt`


### 5) ทดสอบ

ทุกสคริปต์โหลดโมเดลจาก `Model/best.pt` และควรรันจากโฟลเดอร์โปรเจกต์

**รูปภาพ**

```bash
python 03-test_image.py
```

ตรวจรูปทั้งโฟลเดอร์ `test_photos` ด้วย `conf=0.35` บันทึกผลที่ `test/photos_conf035/` (แก้ `conf` และ `name` ได้ตามต้องการ)

**วิดีโอ**

```bash
python 04-test_video.py
```

<!-- ใส่ชื่อไฟล์วิดีโอที่ใช้และที่เก็บผลลัพธ์จาก 04-test_video.py -->

**กล้อง Real-time**

```bash
python 05-test-camera.py
```

ใช้กล้อง `VideoCapture(0)`, `device=0` (GPU), `conf=0.35` กด `q` เพื่อออก

**ผลของค่า conf:** สูง = กรอบขยะน้อยแต่พลาดง่าย, ต่ำ = เจอเพิ่มแต่เสี่ยงกรอบเกิน

<!-- ใส่ตัวอย่างผลทดสอบรูป 3-5 รูป และแคปหน้าจอกล้องสดจาก images/ -->

---

## Troubleshooting

| ปัญหา | วิธีแก้ |
|---|---|
| `01-export_dataset.py` ขึ้น `JSONDecodeError` | มี `.json` เกิน 1 ไฟล์ หรือไฟล์ว่าง ให้เหลือไฟล์ Export จริงไฟล์เดียว |
| Label Studio ขึ้น `Blocked import of regex` | ตั้ง `NLTK_DISABLE_IMPORT_SECURITY=1` (PowerShell ใช้ `$env:`) |
| เทรนขึ้น CUDA OOM / MemoryError | ลด `batch` (เช่น 4), `workers`, ปิด `mixup`, ปิดโปรแกรมอื่น |
| Label ถูกข้ามเพราะ out of bounds coordinates | กรอบเกินขอบภาพ ลากกรอบใหม่ใน Label Studio |
| `FileNotFoundError` ตอนทดสอบ | เช็กว่ามี `Model/best.pt` และโฟลเดอร์ `test_photos` และรันจากโฟลเดอร์โปรเจกต์ |
| `prepare_photos.py` หยุดบอกว่าโฟลเดอร์ไม่ว่าง | ลบหรือย้ายของเก่าใน `frame/images` และ `test_photos` ก่อนรัน |
| กล้องไม่เปิด | ปิดโปรแกรมที่ใช้กล้อง หรือเปลี่ยน `VideoCapture(0)` เป็น `1` |
| กรอบขึ้นบนของที่ไม่ใช่กระป๋อง (เช่น เสื้อลาย) | เพิ่ม conf เป็น 0.4–0.5 หรือเพิ่มรูปพื้นหลังเข้าเทรน |

---

## Notes / ข้อจำกัด

- ตัวเลข Val มาจากการสุ่มแบ่งจากชุดเดียวกับรูปเทรน อาจสูงกว่าการใช้งานจริง จึงทดสอบเพิ่มด้วยรูปที่กันไว้
- ผลบน test 75 รูปนับระดับรูปจาก log ไม่ใช่ mAP เพราะไม่ได้ label ชุดนี้
- ยังพลาดบางกรณี <!-- เติมหลังเปิดดู 3 รูปที่ตรวจไม่เจอ เช่น มุมแปลก แสงสะท้อน -->
- ตอนเทรนมี 3 รูปที่ถูกข้ามเพราะกรอบเกินขอบภาพ
- ใน repo มีเฉพาะ label ของ dataset ไม่มีรูปและโมเดล (ดูลิงก์ด้านล่าง)

---

## ลิงก์ Dataset Model

- Dataset / รูปทดสอบ: <https://github.com/siri-se/AI_YOLO_model/tree/main/dataset/images>
- โมเดล `best.pt`: <https://github.com/siri-se/AI_YOLO_model/blob/main/Model/best.pt>

