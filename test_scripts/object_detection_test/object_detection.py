# For this file to run please clone the following repo
# git clone https://github.com/ultralytics/yolov5.git
# This repo should be saved in the same folder as this file 

import torch
from pathlib import Path
import os
import sys
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

model_path = ROOT / 'best.pt'
repo_path = './yolov5'

model = torch.hub.load(repo_path,'custom', path=model_path, source='local')

image = Image.open(ROOT / 'Mambo2.jpg')

results = model(image)
for det in results.xyxy[0]:
    x_min, y_min, x_max, y_max, confidence, class_label = det
results.print()
results.save()

results.pandas().xyxy[0]

print("This worked")