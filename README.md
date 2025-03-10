# Coca-Cola Product Detection with YOLOv8

## Overview
This repository demonstrates a full pipeline for detecting Coca‑Cola products  using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). 
It includes:
1. Image preprocessing .
2. Model creation and configuration.
3. Training and validation.
4. Evaluation on validation and test sets.
5. Running inference on new images.

**Dataset**:  
- The dataset used for this project can be found here:  
  [https://universe.roboflow.com/test01-fr735/brands-qv6fs/dataset/1](https://universe.roboflow.com/test01-fr735/brands-qv6fs/dataset/1)
## Note on Dataset Format
This project uses a dataset exported in **YOLOv8** format.


## Repository Structure

├── runs/
│   └── detect/                # Folder created by YOLOv8 during training 
├── to_test/                   # Folder containing sample images for testing
├── .gitignore
├── data.yaml                  # YOLOv8 dataset configuration file
├── evaluate.py                # evaluate the trained model
├── model.py                   # create_yolo_model function
├── predict.py                 # test the training model on new data
├── preprocessing.py           # pre-processing the images
├── requirements.txt           # Dependencies for the project
├── train.py                   # train the YOLOv8 model
├── yolov8n-seg.pt             # YOLOv8 segmentation weights
└── yolov8n.pt                 # YOLOv8 detection weights


## Installation & Setup

1. **Clone This Repository**  
   ```bash
   git clone https://github.com/MarRazane/coca-recognition.git
   cd coca-recognition


## 1. Preprocessing 
If you need to resize and/or normalize your images before training, run the following command:
```bash
python preprocessing.py \
  --input data \
  --output data_preprocessed \
  --width 640 \
  --height 640 \
  --normalize
