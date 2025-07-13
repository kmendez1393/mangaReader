import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

class YoloV8Model:
    def __init__(self, model_type="all", conf=0.25):
        # Define model path based on the type selected
        path_to_model = "./models/yolo8l_50epochs/best.pt" #default
        if(model_type == "frame"):
            path_to_model = "./models/yolo8l_50epochs/best.pt" #will be updated when new models are available
        elif(model_type == "text-frame"):
            path_to_model = "./models/yolo8l_50epochs/best.pt" #will be updated when new models are available
        self.model = YOLO(path_to_model)  # Assumes best.pt is in the same directory
        
        # Define which classes to consider based on input type
        self.model_type = model_type
        # Confidence level
        self.conf = conf
        # Class names
        list_of_classes = ["frame", "face", "text", "body"]
        self.allowed_classes = []
        for i, cls in enumerate(list_of_classes):
            if (i == 1 and self.model_type == "frame"):
                break
            elif(i == 1 and self.model_type == "text-frame"):
                continue
            else:
                self.allowed_classes.append(cls)


    def visualize(self, image_path):
        """Run inference, print detected classes, and save image with boxes"""
        results = self.model(image_path, save=True, conf=self.conf)
        # Display the results
        for r in results:
            im_array = r.plot()  # BGR image with boxes
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            plt.imshow(im_rgb)
            plt.title("YOLOv8 Result")
            plt.axis("off")
            plt.show()

    def getSegmentsDetected(self, image_path):
        """Return a list of detected segments as (class_name, [x1, y1, x2, y2])"""
        print(self.allowed_classes)
        print(self.model.names)
        results = self.model(image_path, save=True,conf=self.conf)
        r = results[0]

        segments = []
        class_ids = r.boxes.cls.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for i, cls_id in enumerate(class_ids):
            class_name = self.model.names[int(cls_id)]
            if class_name in self.allowed_classes:
                box = boxes[i].tolist()
                segments.append((class_name, box))

        return segments
