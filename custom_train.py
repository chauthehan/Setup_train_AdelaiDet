from adet.config import get_cfg # use the default config from adet instead of the detectron2 one
from detectron2.engine import DefaultTrainer
import os
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

# def get_char_dicts(img_dir):
#     json_file = os.path.join(img_dir, "via_region_data.json")
#     with open(json_file) as f:
#         imgs_anns = json.load(f)

#     dataset_dicts = []
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}
        
#         filename = os.path.join(img_dir, v["filename"])
#         height, width = cv2.imread(filename).shape[:2]
        
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width
      
#         annos = v["regions"]
#         objs = []
#         for _, anno in annos.items():
#             assert not anno["region_attributes"]
#             anno = anno["shape_attributes"]
#             px = anno["all_points_x"]
#             py = anno["all_points_y"]
#             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#             poly = [p for x in poly for p in x]

#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": 0,
#             }
#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#     return dataset_dicts

def get_char_dicts(img_dir):
    json_file = os.path.join(img_dir, "annotation_coco.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    idx= 0
    for v in imgs_anns["images"]:
        record = {}
        record["file_name"] = os.path.join(img_dir, v["file_name"])
        record["image_id"] = v["id"]
        record["height"] = v["height"]
        record["width"] = v["width"]
      
        annos = imgs_anns["annotations"]
        objs = []
        for anno in annos:
            if anno["image_id"] == record["image_id"]:
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": anno["segmentation"],
                    "category_id": anno["category_id"],
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
classs = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
for d in ["train", "val"]:
    DatasetCatalog.register("char_" + d, lambda d=d: get_char_dicts("char/" + d))
    MetadataCatalog.get("char_" + d).set(thing_classes=classs)

char_metadata = MetadataCatalog.get("char_train")

cfg = get_cfg()
cfg.merge_from_file("configs/SOLOv2/R101_3x.yaml") # path to config file (here, it's specified relative to the AdelaiDet root folder but can also be an absolute path)
cfg.MODEL.WEIGHTS = "SOLOv2_R101_3x.pth" # path to the corresponding pre-trained weights, assuming these have been downloaded as described here https://github.com/aim-uofa/AdelaiDet/tree/master/configs/SOLOv2

cfg.DATASETS.TRAIN = ("char_train",) # name should match the one used when registering the dataset
cfg.DATASETS.TEST = ("char_val",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
cfg.SOLVER.MAX_ITER = 50000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate

# Override SOLOv2 config settings
print(cfg)
cfg.MODEL.SOLOV2.NUM_CLASSES = 94 # only one class for the char dataset

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
