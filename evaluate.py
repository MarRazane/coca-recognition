
from ultralytics import YOLO

def main():
    # 1) Load your best weights
    model = YOLO("runs/detect/cocacola_run3/weights/best.pt")

    # 2) Evaluate on validation
    val_metrics = model.val(
        data="data.yaml", 
        split="val"        
    )
    print("Validation metrics:", val_metrics)

    # 3) Evaluate on test set 
    test_metrics = model.val(
        data="data.yaml",
        split="test"
    )
    print("Test metrics:", test_metrics)

if __name__ == "__main__":
    main()
