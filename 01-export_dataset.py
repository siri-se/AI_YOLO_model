import json
import random
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = Path(r"D:\learning\Mr.Ruji\AI_YOLO\frame\images")
OUTPUT_DIR = Path(r"D:\learning\Mr.Ruji\AI_YOLO\dataset")
TRAIN_SPLIT = 0.8
SEED = 42

def find_json_file(folder: Path) -> Path:
    """Find the first .json file in the folder where this script is located."""
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        sys.exit(
            f"ERROR: No .json file found in folder: {folder}\n"
            f"Place the exported JSON file from Label Studio in the same folder as this script first."
        )
    if len(json_files) > 1:
        print(f"WARNING: Found more than 1 .json file in this folder, using the first one: {json_files[0].name}")
        for f in json_files:
            print(f"    - {f.name}")
    return json_files[0]


def get_image_filename(task):
    """
    Extract the actual image filename (basename) from task['data'] in Label Studio.
    Supports multiple formats: a direct 'image' key, local-files URL (?d=...), or a generic URL.
    """
    data = task.get("data", {})

    image_value = None
    for key in ("image", "img", "picture", "photo"):
        if key in data:
            image_value = data[key]
            break

    if image_value is None:
        for key, value in data.items():
            if isinstance(value, str) and (
                "image" in key.lower() or "/data/local-files" in value
            ):
                image_value = value
                break

    if image_value is None:
        return None

    parsed = urlparse(image_value)
    if "d=" in parsed.query:
        rel_path = unquote(parsed.query.split("d=", 1)[1])
        # normalize both Windows (\) and POSIX (/) separators before taking the basename
        rel_path = rel_path.replace("\\", "/")
        return Path(rel_path).name

    return unquote(Path(parsed.path).name)


def collect_classes(tasks):
    """Scan all tasks for labels actually used, to build the class list."""
    classes = set()
    for task in tasks:
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                labels = result.get("value", {}).get("rectanglelabels", [])
                classes.update(labels)
    return sorted(classes)


def convert_task_to_yolo_lines(task, class_to_id):
    """Convert the annotations of one task into a list of YOLO format lines (<class_id> <cx> <cy> <w> <h>)."""
    lines = []
    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue
        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue
            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue

            # Label Studio stores values as % of image (0-100), top-left corner + width/height
            x_pct, y_pct = value["x"], value["y"]
            w_pct, h_pct = value["width"], value["height"]

            # YOLO expects normalized (0-1), center-based coordinates
            x_center = (x_pct + w_pct / 2) / 100.0
            y_center = (y_pct + h_pct / 2) / 100.0
            width = w_pct / 100.0
            height = h_pct / 100.0

            for label in labels:
                class_id = class_to_id[label]
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def main():
    random.seed(SEED)

    json_path = find_json_file(SCRIPT_DIR)
    print(f"Using JSON file: {json_path.name}")

    if not IMAGES_DIR.exists():
        sys.exit(f"ERROR: Source images folder not found at path: {IMAGES_DIR}\nUpdate IMAGES_DIR at the top of this script.")

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        sys.exit(
            "ERROR: The JSON file is not a list of tasks as expected — "
            "check that the export format selected was 'JSON' (not JSON-MIN or another format)."
        )

    classes = collect_classes(tasks)
    if not classes:
        sys.exit(
            "ERROR: No bounding box labels (rectanglelabels) found in this file — "
            "check that the labeling setup is Object Detection with Bounding Boxes "
            "and that at least one task has been labeled."
        )
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    print(f"Found {len(classes)} classes: {classes}")

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    valid_tasks = []
    skipped_no_file = 0
    skipped_no_annotation = 0

    for task in tasks:
        filename = get_image_filename(task)
        if filename is None:
            skipped_no_file += 1
            continue
        src_image_path = IMAGES_DIR / filename
        if not src_image_path.exists():
            skipped_no_file += 1
            continue
        if not task.get("annotations"):
            skipped_no_annotation += 1
            continue
        valid_tasks.append((task, filename, src_image_path))

    if skipped_no_file:
        print(f"WARNING: Skipped {skipped_no_file} tasks: source image file not found in IMAGES_DIR")
    if skipped_no_annotation:
        print(f"WARNING: Skipped {skipped_no_annotation} tasks: no annotation present")

    if not valid_tasks:
        sys.exit(
            "ERROR: No usable tasks found — check that IMAGES_DIR points to the correct folder "
            "and that filenames match what is referenced in Label Studio."
        )

    random.shuffle(valid_tasks)
    split_idx = int(len(valid_tasks) * TRAIN_SPLIT)
    train_tasks = valid_tasks[:split_idx]
    val_tasks = valid_tasks[split_idx:]

    def process_split(split_tasks, split_name):
        count = 0
        for task, filename, src_image_path in split_tasks:
            lines = convert_task_to_yolo_lines(task, class_to_id)
            if not lines:
                continue

            dst_image_path = OUTPUT_DIR / "images" / split_name / filename
            shutil.copy2(src_image_path, dst_image_path)

            label_path = (OUTPUT_DIR / "labels" / split_name / filename).with_suffix(".txt")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            count += 1
        return count

    n_train = process_split(train_tasks, "train")
    n_val = process_split(val_tasks, "val")

    if n_train == 0:
        sys.exit("ERROR: No images were successfully converted — check paths and annotations again.")

    print(f"Train: {n_train} images")
    print(f"Val:   {n_val} images")

    (OUTPUT_DIR / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    data_yaml_content = (
        f"path: {OUTPUT_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    (OUTPUT_DIR / "data.yaml").write_text(data_yaml_content, encoding="utf-8")

    print(f"\ndata.yaml and classes.txt written to: {OUTPUT_DIR}")
    print("Ready to train with Ultralytics, e.g.:")
    print(f'  yolo detect train data="{OUTPUT_DIR / "data.yaml"}" model=yolov8n.pt epochs=100')


if __name__ == "__main__":
    main()
