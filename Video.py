import torch
import cv2
import numpy as np
import time
import os

model = torch.hub.load('ultralytics/yolov5', model="custom", path='best.pt')
# device = torch.device("cuda")
# model.to(device)
video = cv2.VideoCapture("Data Test/TestFaceMask2.mp4")
# video = cv2.VideoCapture(0)
classes = ["No Facemask", "Wear Facemask"]
colors = [(0,0,255), (0,255,0)]
font = cv2.FONT_HERSHEY_SIMPLEX
path_storage = "Warning/"
prev_frame_time = 0
new_frame_time = 0
count = 0

if not os.path.exists(path_storage): 
    os.mkdir("Warning")

while video.isOpened():
    record = False
    ret, image = video.read()
    if ret == False:
        break
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    height, width, _ = image.shape
    results = model(image)
    labels, bbox = results.xyxyn[0][:,-1], results.xyxyn[0][:,:-1]
    cv2.rectangle(image, (0,0), (width, 50), (0,255,0), -1)
    if len(labels) > 0:
        for i in range(len(labels)):
            x1, y1 = bbox[i][0].item()*width, bbox[i][1].item()*height
            x2, y2 = bbox[i][2].item()*width, bbox[i][3].item()*height
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = classes[int(labels[i])]
            probability = int(bbox[i][4].item()*100)
            if probability > 50:
                cv2.rectangle(image, (x1,y1), (x2,y2), colors[int(labels[i])], 2)
                cv2.putText(image,  label +" "+str(probability)+"%", (x1,y1), font, 1, colors[int(labels[i])],2)
            if count%100==0: 
                if label=="No Facemask":
                    print("Warning")
                    record = True
                else:
                    print("Safety")
    current_time = time.ctime(time.time())
    
    cv2.putText(image, "FPS: "+str(int(fps)), (10,20), font, 0.8, (255,0,0),2)
    cv2.putText(image, current_time, (10,40), font, 0.8, (255,0,0),2)
    cv2.imshow("Image", image)
    count+=1
    if record:
        cv2.imwrite(path_storage+current_time+".jpg", image)
    if cv2.waitKey(1) == ord("q"):
        break
video.release()
cv2.destroyAllWindows()