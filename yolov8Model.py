import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

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

        self.frames = []
        self.image = ""


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
    
    def searchHorizontalFrames(self,rank,tmpArray,y_min_tol,target):
        
        # Check for frames at the same horizontal level
        anyFrameAtTheSameHorizontalLevel = "no"
        candidates = []
        for frame in tmpArray:
                if frame["analyzed"] == "yes":
                    continue
                elif frame["ymax_norm"] < y_min_tol:
                    # Calculate the distance from previous frame identified = target
                    top_right_corner = np.array([frame["xmax_norm"],frame["ymin_norm"]])
                    frame["distance"] = abs(np.linalg.norm(top_right_corner - target)) #Euclidean distance)
                    frame["candidate"] = "yes"
                    anyFrameAtTheSameHorizontalLevel = "yes"
                    candidates.append(frame)
        if anyFrameAtTheSameHorizontalLevel == "yes":
            # Find the minimum distance
            min_distance = min(obj["distance"] for obj in candidates)
            # Assign ascending rank
            for frame in tmpArray:
                if (min_distance == frame["distance"]):
                    frame["analyzed"] = "yes"
                    frame["rank"] = rank
                    rank = rank + 1

        return rank

    def getSegmentsDetected(self, image_path):
        """Return a list of detected segments as (class_name, [x1, y1, x2, y2])"""
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
    
    def orderSequenceFrames(self,image_path):
        #The idea is to order the frames and elements in order to be processed sequentially 
        """Return a list of detected segments as (class_name, [xmin, ymin, xmax, ymax])"""
        results = self.model(image_path, save=True,conf=self.conf)
        r = results[0]
        height, width = r.orig_shape

        segments = []
        class_ids = r.boxes.cls.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for i, cls_id in enumerate(class_ids):
            class_name = self.model.names[int(cls_id)]
            if class_name in self.allowed_classes:
                box = boxes[i].tolist()
                segments.append((class_name, box))
        """
        The process to organize the frames follows the next steps:
        - Get the first frame. This can be done by getting the top right corner that is closer to
        the coordenates (x=1,y=0). The corresponding coordinates of the frame to be analyzed are xmax and ymin
        - Then, identify if there are other frames in the horizontal level of the first frame or above
        - Find the next row with the frame on the top right and repeat the process. 
        """
        
        # Find the frame with the top right corner closest to (1,0) using distance = (1-xmax)+(ymin)
        tolerance = 0.03 
        frames_list = []
        global_rank = 0
        for frame in segments:
            algorithm_object = {"distance":0,"analyzed":"no","rank":1000,"xmin":0, "ymin":0,"xmax":0, "ymax":0, 
                                "xmin_norm":0, "ymin_norm":0,"xmax_norm":0, "ymax_norm":0, "candidate":"no"}
            algorithm_object["xmin_norm"]= (frame[1][0])/width
            algorithm_object["ymin_norm"] = (frame[1][1])/height
            algorithm_object["xmax_norm"] = (frame[1][2])/width
            algorithm_object["ymax_norm"] = (frame[1][3])/height
            algorithm_object["xmin"]= (frame[1][0])
            algorithm_object["ymin"] = (frame[1][1])
            algorithm_object["xmax"] = (frame[1][2])
            algorithm_object["ymax"] = (frame[1][3])
            frames_list.append(algorithm_object)

        target = np.array([1.0, 0.0]) #initial target
        while(global_rank<len(frames_list)):
            global_rank, y_min_tol,index_updated_element = self.getFrameTopRightCorner(frames_list, global_rank, tolerance,target)
            target = np.array([frames_list[index_updated_element]["xmin_norm"],frames_list[index_updated_element]["ymin_norm"]])
            global_rank = self.searchHorizontalFrames(global_rank,frames_list,y_min_tol,target)
            target = np.array([frames_list[index_updated_element]["xmax_norm"], frames_list[index_updated_element]["ymax_norm"]])

        self.frames = frames_list
        self.image = image_path
                
    def getFrameTopRightCorner(self, tmpArray, rank, tolerance,target):
        candidates = []
        for frame in tmpArray:
            if frame["analyzed"] == "yes":
                continue
            else:
                top_right_corner = np.array([frame["xmax_norm"],frame["ymin_norm"]])
                frame["distance"] = np.linalg.norm(top_right_corner - target) #Euclidean distance
                frame["candidate"] = "yes"
                candidates.append(frame)

        # Find the minimum distance
        min_distance = min(obj["distance"] for obj in candidates)

        for i,frame in enumerate(tmpArray):

            if frame["distance"] == min_distance:
                frame["analyzed"] = "yes"
                frame["rank"] = rank
                rank = rank + 1
                y_min_tol = frame["ymax_norm"]*(1+tolerance)
                min_index = i
            frame["candidate"] = "no"
        
        return rank, y_min_tol,min_index 

    def extractFrames(self): 
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = "img-"+timestamp_str
        #Create folder to save imgs
        os.makedirs(folder, exist_ok=True)
        # Load the image
        image = cv2.imread(self.image)

        # Iterate and crop
        for frame in self.frames:
            cropped = image[int(frame["ymin"]):int(frame["ymax"]), int(frame["xmin"]):int(frame["xmax"])]
            frame_number = int(frame["rank"])
            save_path = os.path.join(folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(save_path, cropped)
        return folder


    
    
    



            
            


