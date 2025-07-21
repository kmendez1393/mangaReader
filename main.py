from yolov8Model import YoloV8Model

model = YoloV8Model("frame")
model.orderSequenceFrames("./images/test4.jpg")
print(model.extractFrames())
