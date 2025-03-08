from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/cocacola_run3/weights/best.pt")
    results = model.predict(
        source="to_test",  
        conf=0.25,             
        save=True              
    )
    print("Predictions saved to runs/detect/predict/")

if __name__ == "__main__":
    main()
