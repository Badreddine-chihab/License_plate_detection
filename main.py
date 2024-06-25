from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch  ²

# Use the model
model.train(data="config.yaml", epochs=50,batch = 8)  # train the model