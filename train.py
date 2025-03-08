from model import create_yolo_model

def main():
    # 1) create a YOLO detection model
    model = create_yolo_model(weights="yolov8n.pt", task="detect")
    
    # 2) train on your data
    results = model.train(
        data="data.yaml",  
        epochs=50,
        imgsz=640,
        batch=8,
        name="cocacola_run"
    )
    
    print("Training finished. Check runs/detect/cocacola_run/ for logs.")

if __name__ == "__main__":
    main()

    
