import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue

import cv2
import numpy as np
import websockets
import random
from PIL import Image

from ultralytics import YOLO

from traffic_sign_detection import detect_traffic_signs
from car_controller import car_control

lane_detection_model = YOLO("./lane_segment_model_v15.pt")
traffic_sign_detection_model = YOLO("./traffic_sign_detection.pt")

traffic_sign_model = cv2.dnn.readNetFromONNX("./traffic_sign_classifier_lenet_v3.onnx")

traffic_image_queue = Queue(maxsize=5)
lane_image_queue = Queue(maxsize=5)

traffic_sign_queue = Queue(maxsize=5)
lane_segment_queue = Queue(maxsize=5)

driving_info_queue = Queue(maxsize=5)

def process_traffic_sign_loop(traffic_image_queue, traffic_sign_queue):
    while True:
        if traffic_image_queue.empty():
            time.sleep(0.1)
            continue
        image = traffic_image_queue.get()

        # traffic_sign_detection_model.predict(show=True, show_labels= True, show_boxes=True, source=image, verbose=False)
        # Prepare visualization image
        draw = image.copy()
        signal = detect_traffic_signs(image, traffic_sign_model, draw=draw)
        
        if not traffic_sign_queue.full() and signal != None:
            traffic_sign_queue.put(signal)

        # cv2.imshow("Traffic signs", draw)
        # cv2.waitKey(1)

def process_lane_line_loop(lane_image_queue, lane_segment_queue):
    while True:
        if lane_image_queue.empty():
            time.sleep(0.1)
            continue
        image = lane_image_queue.get()

        results = lane_detection_model.predict(show=False, show_labels= False, show_boxes=False, source=image, verbose=False)
        
        if results[0].masks == None: continue
        masks = results[0].masks.data.cpu().numpy()[0]

        if not lane_segment_queue.full():
            lane_segment_queue.put([image, masks])

def drive_processing(traffic_sign_queue, lane_segment_queue, driving_info_queue):
    while True:
        signal = None
        lane_segment = None

        if not traffic_sign_queue.empty():
            signal = traffic_sign_queue.get()
            # print(signal)

        if not lane_segment_queue.empty():
            lane_segment = lane_segment_queue.get()
            # cv2.imshow('YOLO V8 Detection1', lane_segment)
            # cv2.waitKey(1)
        
        message = car_control(signal, lane_segment, isShow= False)

        if (not driving_info_queue.full()) and (message != None):
            driving_info_queue.put(message)

async def socket_conversation(websocket, path):
    async for message in websocket:
        # Get image from simulation
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (640, 480))

        cv2.imwrite(f'./imgs/{random.random()}.jpg',image)

        if not traffic_image_queue.full():
            traffic_image_queue.put(image)

        if not lane_image_queue.full():
            lane_image_queue.put(image)

        if not driving_info_queue.empty():
            message = driving_info_queue.get()
            # print(message)
            await websocket.send(message)

async def main():
    async with websockets.serve(socket_conversation, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    p1 = Process(target=process_traffic_sign_loop, args=(traffic_image_queue, traffic_sign_queue))
    p2 = Process(target=process_lane_line_loop, args=(lane_image_queue, lane_segment_queue))
    p3 = Process(target=drive_processing, args=(traffic_sign_queue, lane_segment_queue, driving_info_queue))
    p1.start()
    p2.start()
    p3.start()
    asyncio.run(main())