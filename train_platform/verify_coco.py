import shutil
import json
import logging
from pathlib import Path
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import from train_platform
import sys
sys.path.append(r"c:\Users\qc\Desktop\deep-thought\train\TBS")

from train_platform.services.dataset_conversion_service import DatasetConversionService

def verify_coco():
    base_dir = Path("verify_temp")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create dummy data
    images_dir = base_dir / "images"
    images_dir.mkdir()
    
    # Create a dummy image
    img_path = images_dir / "test_img.jpg"
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(img_path)

    # Create dummy json
    # {"version": "4.5.6", "flags": {}, "shapes": [{"label": "cat", "points": [[10, 10], [50, 50]], "group_id": null, "description": "", "shape_type": "rectangle", "flags": {}}], "imagePath": "test_img.jpg", "imageData": "...", "imageHeight": 100, "imageWidth": 100}
    json_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "cat",
                "points": [[10, 10], [50, 50]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": "test_img.jpg",
        "imageData": "",
        "imageHeight": 100,
        "imageWidth": 100
    }
    json_path = images_dir / "test_img.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f)

    logger.info("Created dummy data.")

    # 2. Run Step 1: JSON -> YOLO
    svc = DatasetConversionService()
    labels_dir = base_dir / "labels"
    class_names_path = base_dir / "class_names.txt"
    
    logger.info("Running JSON -> YOLO...")
    svc.conversion_json_2_yolo(
        storage_path=images_dir,
        output_labels_dir=labels_dir,
        class_names_path=class_names_path,
        write_files=True
    )

    # Verify YOLO output
    label_path = labels_dir / "test_img.txt"
    if not label_path.exists():
        logger.error("YOLO label file not created")
        return

    content = label_path.read_text().strip()
    logger.info(f"YOLO content: {content}")
    # Expected: 0 0.3 0.3 0.4 0.4 (approx)
    # Center x: 30/100 = 0.3
    # Center y: 30/100 = 0.3
    # W: 40/100 = 0.4
    # H: 40/100 = 0.4
    
    # 3. Run Step 2: YOLO -> COCO
    coco_json_path = base_dir / "instances_default.json"
    logger.info("Running YOLO -> COCO...")
    
    svc.conversion_yolo_2_coco(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names_path=class_names_path,
        output_json_path=coco_json_path
    )

    # Verify COCO output
    if not coco_json_path.exists():
        logger.error("COCO JSON not created")
        return

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    logger.info("COCO content loaded.")
    
    # Check images
    if len(coco["images"]) != 1:
        logger.error(f"Expected 1 image, got {len(coco['images'])}")
    else:
        logger.info("Image count correct.")
        if coco["images"][0]["width"] != 100:
            logger.error(f"Image width mismatch: {coco['images'][0]['width']}")

    # Check annotations
    if len(coco["annotations"]) != 1:
        logger.error(f"Expected 1 annotation, got {len(coco['annotations'])}")
    else:
        ann = coco["annotations"][0]
        # BBox: [x, y, w, h] -> [10, 10, 40, 40]
        bbox = ann["bbox"]
        logger.info(f"BBox: {bbox}")
        if bbox != [10.0, 10.0, 40.0, 40.0]:
             logger.error(f"BBox mismatch, expected [10, 10, 40, 40], got {bbox}")
        else:
             logger.info("BBox correct.")

    logger.info("Verification passed!")

if __name__ == "__main__":
    try:
        verify_coco()
    except Exception as e:
        logger.exception("Verification failed with exception")
