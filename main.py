from random import random
from fastapi import FastAPI, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# from helper import detectFunction
from helper.predictor import Predictor

# create apps:
app = FastAPI(
    title="YOLO REST API",
    description="""Obtain object value out of image
    and return image and json result""",
    version="0.0.1",
)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8001",
    "*"
]
app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "wellcome to hololens server websocket"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        respFromClient = await websocket.receive_text()
        print("respFromClient ==>" + respFromClient)
        # videoUrl = 'D:/AR/Thesis/backend/ObjectDetection/fastapi/assets/video/video.mp4'
        # detect_result = detectFunction.detectWithYoloV3(videoUrl)
        # print("detect_result ==> " + detect_result)
        # await websocket.send_json(detect_result, "text")
        return {"message": "success get ws data"}

@app.post("/predict")
async def predict(file: bytes = File(...)):
    if not file:
        return {"classes": [], "confidences": [],'boxes': [], "error": "there is no file"}
    else:
        modelConfiguration = "./assets/model/cfg/yolov3.cfg"
        modelWeights = "./assets/model/yolov3.weights"
        labelsPath = "./assets/model/coco.names"
        labels = open(labelsPath).read().strip().split('\n')

        imageStream = file
        pred = Predictor()
        detect_result = pred.prediction(modelConfiguration, modelWeights, labels, imageStream)
        return detect_result

@app.post("/predict-tiny")
async def predict(file: bytes = File(...)):
    if not file:
        return {"classes": [], "confidences": [],'boxes': [], "error": "there is no file"}
    else:
        modelConfiguration = "./assets/model/cfg/yolov3-tiny.cfg"
        modelWeights = "./assets/model/yolov3-tiny.weights"
        labelsPath = "./assets/model/coco.names"
        labels = open(labelsPath).read().strip().split('\n')

        imageStream = file
        pred = Predictor()
        detect_result = pred.prediction(modelConfiguration, modelWeights, labels, imageStream)
        return detect_result

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0")

# respFromClient = await websocket.receive_text()
# print("respFromClient ==> " + respFromClient)
# wsEvent = json.loads(respFromClient)["eventMsg"]
# videoUrl = 'D:/AR/Thesis/backend/ObjectDetection/fastapi/assets/video/video.mp4'
# detect_result = detectFunction.detectWithYoloV3(videoUrl)
# await websocket.send_json(detect_result, "text")

# if(wsEvent == "img event"):
#     img = useConvertByteToImage(respFromClient);
#     print("img ==> " + img)