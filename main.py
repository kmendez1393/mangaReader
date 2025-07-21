from yolov8Model import YoloV8Model

model = YoloV8Model("all")
model.orderSequenceFrames("./images/test1.jpg")
print(model.extractFrames())
