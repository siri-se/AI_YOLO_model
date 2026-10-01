import random
import shutil
from pathlib import Path
from PIL import Image, ImageOps

# ---------- ตั้งค่า ----------
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw_photos"          # รูปดิบ แยกโฟลเดอร์ตามยี่ห้อ
OUT_DIR = SCRIPT_DIR / "frame" / "images"    # รูปที่ย่อแล้ว (ส่งเข้า Label Studio)
TEST_DIR = SCRIPT_DIR / "test_photos"        # รูปกันไว้ทดสอบ
TEST_PER_CLASS = 25
MAX_SIZE = 1280
SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
# -----------------------------

# รองรับ HEIC (iPhone) ถ้าติดตั้ง pillow-heif ไว้: pip install pillow-heif
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def save_resized(src: Path, dst: Path):
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    img.save(dst, quality=90)


def main():
    random.seed(SEED)

    if not RAW_DIR.exists():
        raise SystemExit(f"ไม่พบโฟลเดอร์ {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # กันการรันซ้ำแล้วรูปปนกัน
    if any(OUT_DIR.iterdir()) or any(TEST_DIR.iterdir()):
        raise SystemExit(
            "frame/images หรือ test_photos ไม่ว่าง ลบหรือย้ายของเก่าออกก่อนรัน "
            "(กันไม่ให้ชื่อและรูปปนกัน)"
        )

    total_train = total_test = 0

    for brand_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        brand = brand_dir.name.lower()
        files = sorted(f for f in brand_dir.iterdir() if f.suffix.lower() in EXTS)

        if not files:
            print(f"[{brand}] ไม่พบรูป ข้าม")
            continue

        random.shuffle(files)
        n_test = min(TEST_PER_CLASS, len(files))
        test_files, train_files = files[:n_test], files[n_test:]

        for i, f in enumerate(train_files, 1):
            save_resized(f, OUT_DIR / f"{brand}_{i:03d}.jpg")
        for i, f in enumerate(test_files, 1):
            save_resized(f, TEST_DIR / f"{brand}_test_{i:03d}.jpg")

        total_train += len(train_files)
        total_test += len(test_files)
        print(f"[{brand}] ทั้งหมด {len(files)} | label {len(train_files)} | ทดสอบ {len(test_files)}")

    print(f"\nเสร็จแล้ว: frame/images = {total_train} รูป, test_photos = {total_test} รูป")


if __name__ == "__main__":
    main()