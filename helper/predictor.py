import io
from PIL import Image
import json
import numpy as np
import cv2

confidenceThreshold = 0.7
NMSThreshold = 0.3

class Predictor:
    def __init__(self, ):
        pass

    def read_image_to_rgb(self, imgData):
        image = Image.open(imgData)
        image = image.rotate(90)

        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def prediction(self, modelConfiguration, modelWeights, labels, imageDataStream):
        np.random.seed(10)
        COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

        layerName = net.getLayerNames()
        layerName = [layerName[i - 1] for i in net.getUnconnectedOutLayers()]
        frame = io.BytesIO(imageDataStream)
        frame = np.array(self.read_image_to_rgb(frame)) 
        # print(frame)
        # Convert RGB to BGR 
        # video = video[:, :, ::-1].copy() 
        (W, H) = (None, None)
        if W is None or H is None:
            (H,W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
        net.setInput(blob)
        layersOutputs = net.forward(layerName)

        boxes = []
        confidences = []
        classIDs = []

        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        #Apply Non Maxima Suppression
        detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

        if(len(detectionNMS) > 0):
            for i in detectionNMS.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                mx = int((width+x)/2)
                my = int((y+height)/2)
                center_coordinates = (mx, my)
                print(center_coordinates)
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # cv2.imshow('output', frame)
        arrayResult = np.array(labels)[classIDs].tolist()
        finalResp = {"classes": arrayResult, "confidences": confidences,'boxes': boxes, "error": ""}   
        return finalResp     
