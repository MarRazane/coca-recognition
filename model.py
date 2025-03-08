from ultralytics import YOLO
import os

def create_yolo_model(
    weights="yolov8n.pt", 
    task="detect"
):
    
    model = YOLO(weights)
    
    model.task = task
    
    return model


def main():
    
    
    # 1) Create a detection model with YOLOv8n
    detection_model = create_yolo_model(
        weights="yolov8n.pt",
        task="detect"
    )
    print("Created detection model with weights:", detection_model.ckpt_path)
    
    # 2) create a segmentation model with YOLOv8n-seg
    seg_model = create_yolo_model(
        weights="yolov8n-seg.pt",
        task="segment"
    )
    print("Created segmentation model with weights:", seg_model.ckpt_path)
    
if __name__ == "__main__":
    main()
