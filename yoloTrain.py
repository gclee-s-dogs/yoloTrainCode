from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    # Train the model
    results = model.train(data="datasets/data.yaml", epochs=100, imgsz=640, batch=16, device='0')  # 改成 'cpu' 或 0

