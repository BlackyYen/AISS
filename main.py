#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

from fuzzy_system import fuzzy_system, grade

from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

from mobilenet_resize import resize_keep_aspectratio

g1 = tf.Graph() # 加载到Session 1的graph
g2 = tf.Graph() # 加载到Session 2的graph
 
sess1 = tf.Session(graph=g1) # Session1
sess2 = tf.Session(graph=g2) # Session2

with sess1.as_default(): 
    with g1.as_default():
        #load the model
        #load json and create model
        json_file = open("../weights/mobilenetv2_model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_mobilenet = model_from_json(loaded_model_json)
        # load weights into new model
        model_mobilenet.load_weights("../weights/mobilenetv2_final_weights.h5")

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = InteractiveSession(config=config)

name_of_class = 'sperm'
video_path = '../video/test_video/1sec.mp4'
output_path = '../video/test_video_out/'
output_name = 'test.avi'
my_maxlen = 40

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video",
                default=video_path)
ap.add_argument("-c", "--class", help="name of class", default=name_of_class)
args = vars(ap.parse_args())

pts = [deque(maxlen=my_maxlen) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")
#list = [[] for _ in range(100)]


def main(yolo):

    start = time.time()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    counter = []
    # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    find_objects = ['person']
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(
            output_path + args["input"][43:57] + args["class"] + '_' + output_name, fourcc, 15, (w, h))
        list_file = open('./firebase/detection_rslt.txt', 'w')
        frame_index = -1

    fps = 0.0
    real_id = 0
    data = {}
    data_id = {}
    ct = []

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        try:
            org = frame.copy()
        except AttributeError:
            break
        t1 = time.time()

        #image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, confidence, class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature)
                      for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []

        for det in detections:
            bbox = det.to_tlbr()
            # 畫出 yolo 偵測框
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
            #     bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            # print(class_names)
            # print(class_names[p])

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            # print(frame_index)
            # list_file.write(str(frame_index)+',')
            # list_file.write(str(track.track_id)+',')

            # 畫出 deepsort 預測框
            # cv2.rectangle(frame, (int(bbox[0]), int(
            #     bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)

            # .split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b0 = str(bbox[0])
            # .split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b1 = str(bbox[1])
            # .split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b2 = str(bbox[2]-bbox[0])
            b3 = str(bbox[3]-bbox[1])

            # list_file.write(str(b0) + ','+str(b1) + ','+str(b2) + ','+str(b3))
            # print(str(track.track_id))
            # list_file.write('\n')
            # list_file.write(str(track.track_id)+',')

            # 原始的物件編號
            # cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            # 新的物件編號(不會跳號)
            # try:
            #     cv2.putText(frame, str(data_id[track.track_id]), (int(
            #         bbox[0]), int(bbox[1] - 10)), 0, 5e-3 * 120, (color), 2)
            # except KeyError:
            #     real_id += 1
            #     data_id.setdefault(track.track_id, real_id)
            #     cv2.putText(frame, str(data_id[track.track_id]), (int(
            #         bbox[0]), int(bbox[1] - 10)), 0, 5e-3 * 120, (color), 2)
            #     print(data_id[track.track_id])

            if len(class_names) > 0:
                class_name = class_names[0]
            # 畫出物件名稱
            #    cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),
                      int(((bbox[1])+(bbox[3]))/2))
            # track_id[center]

            pts[track.track_id].append(center)

            thickness = 5
            # center point
            # 劃出重心
            # cv2.circle(frame,  (center), 1, color, thickness)

            # draw motion path
            # 劃出路徑
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[track.track_id][j-1]),
                         (pts[track.track_id][j]), (255,255,255), 2)
            # 劃出 class 名稱       
            # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

            fr = my_maxlen
            # pts[track.track_id][(len(pts[track.track_id])-1)] 當前偵之座標
            if len(pts[track.track_id]) == fr:

                # 計算向量
                coordinate_cur = (pts[track.track_id]
                                  [(len(pts[track.track_id]) - 1)])
                coordinate_5 = (pts[track.track_id]
                                [(len(pts[track.track_id])) - 5])
                coordinate_10 = (pts[track.track_id]
                                 [(len(pts[track.track_id])) - 10])
                coordinate_15 = (pts[track.track_id]
                                 [(len(pts[track.track_id])) - 15])
                coordinate_20 = (pts[track.track_id]
                                 [(len(pts[track.track_id])) - 20])
                coordinate_40 = (pts[track.track_id]
                                 [(len(pts[track.track_id])) - 40])
                # 一個計算距離的向量
                x1 = coordinate_cur[0]-coordinate_40[0]
                y1 = coordinate_cur[1]-coordinate_40[1]
                xy1 = np.array([x1, y1])
                # 兩個計算角度差的向量
                x2 = coordinate_cur[0]-coordinate_20[0]
                y2 = coordinate_cur[1]-coordinate_20[1]
                xy2 = np.array([x2, y2])
                x3 = coordinate_20[0]-coordinate_40[0]
                y3 = coordinate_20[1]-coordinate_40[1]
                xy3 = np.array([x3, y3])
                # 計算向量長度
                d1 = np.sqrt(xy1.dot(xy1))
                d2 = np.sqrt(xy2.dot(xy2))
                d3 = np.sqrt(xy3.dot(xy3))
                # 計算xy2與xy3兩個向量之間的角度差
                cos_angle = xy2.dot(xy3)/(d2*d3)
                angle = np.arccos(cos_angle)
                angle2 = angle*360/2/np.pi
                # 距離
                d = round(d1, 1)
                # 角度
                a = round(angle2, 1)
                if str(a) == 'nan':
                    a = 0.0
# =============================================================================
#                     try:
#                         cv2.putText(frame,str(data_id[track.track_id]),(int(bbox[0]), int(bbox[1] - 10)),0, 5e-3 * 120, (color),2)
#                     except KeyError:
#                         real_id += 1
#                         data_id.setdefault(track.track_id, real_id)
#                         cv2.putText(frame,str(data_id[track.track_id]),(int(bbox[0]), int(bbox[1] - 10)),0, 5e-3 * 120, (color),2)
#                         print(data_id[track.track_id])
# =============================================================================
                # data[track.track_id][0] ID
                # data[track.track_id][1] distance
                # data[track.track_id][2] angle
                # data[track.track_id][3] fuzzy output
                # 建立字典
                try:
                    data[track.track_id][0][track.track_id]
                except KeyError:
                    real_id += 1
                    data_id.setdefault(track.track_id, real_id)
                    data_list = [data_id, [], [], [], []]
                    data.setdefault(track.track_id, data_list)
                data[track.track_id][1].append(d)
                data[track.track_id][2].append(a)

                # 把值平均並丟入模糊系統
                if len(data[track.track_id][1]) == 20 and len(data[track.track_id][2]) == 20:
                    d_total = 0
                    a_total = 0
                    for j in range(20):
                        d_total += data[track.track_id][1][j]
                        a_total += data[track.track_id][2][j]
                    d_ave = round(d_total/20, 1)
                    a_ave = round(a_total/20, 1)
                    # fuzzy_system
                    g = grade(grade_system=grade_system,
                              input1=d_ave, input2=a_ave)
                    # 刪除最舊的n筆資料
                    while(True):
                        del data[track.track_id][1][0], data[track.track_id][2][0]
                        if len(data[track.track_id][1]) == 19 and len(data[track.track_id][2]) == 19:
                            break
                    data[track.track_id][3].clear()
                    data[track.track_id][3].append(g)
                try:
                    # color(bgr)
                    if data[track.track_id][3][0] >= 0 and data[track.track_id][3][0] <= 35:
                        cv2.rectangle(frame, (int(
                            bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [255, 0, 0], 2)
                    elif data[track.track_id][3][0] <= 55:
                        cv2.rectangle(frame, (int(
                            bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 255, 242], 2)
                    elif data[track.track_id][3][0] <= 75:
                        cv2.rectangle(frame, (int(
                            bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 255, 0], 2)
                    elif data[track.track_id][3][0] <= 100:
                        cv2.rectangle(frame, (int(
                            bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 0, 255], 2)
                except IndexError:
                    break
                try:
                    ct[int(data[track.track_id][0][track.track_id]-1)]
                except:
                    ct.append(data[track.track_id][0][track.track_id])
                    # 裁切圖片
                    x1 = int(bbox[0])
                    x2 = int(bbox[2])
                    y1 = int(bbox[1])
                    y2 = int(bbox[3])
                    ww = int(x2-x1)
                    hh = int(y2-y1)
                    crop_img = org[y1: y2, x1: x2]
                try:
                    # 寫出類別
                    cv2.putText(frame,str(data[track.track_id][4][0]),(int(bbox[0]-6), int(bbox[1] - 12)),0, 5e-3 * 120, (0,0,0), 2)
                except:
                    # mobilenet 分類
                    with sess1.as_default():
                        with sess1.graph.as_default():
                            # cv2.imwrite('classImg.jpg',crop_img)
                            # img_input = cv2.imread('classImg.jpg')
                            img = resize_keep_aspectratio(crop_img)
                            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = np.expand_dims(img, axis=0)
                            img = preprocess_input(img)
                            preds = model_mobilenet.predict(img)
                            predictions = np.argmax(preds, axis = 1)[0]
                            # classes = ['amorphous', 'normal', 'pyriform', 'tapered']
                            classes = ['amorphous', 'normal', 'pyriform', 'tapered']
                            print('Predicted:', classes[predictions])
                            data[track.track_id][4].append(classes[predictions])

                # 將資料寫成文字檔案
                list_file.write(str(frame_index)+',')
                list_file.write(str(data[track.track_id][0][track.track_id])+',')
                list_file.write(str(round(((bbox[0])+(bbox[2]))/2,1))+','+str(round(((bbox[1])+(bbox[3]))/2,1))+',')
                list_file.write(str(round(d,1))+',')
                list_file.write(str(round(a,1))+',')
                list_file.write(str(round(g,1)))
                list_file.write('\n')

        count = len(set(counter))
        # cv2.putText(frame, "Total Pedestrian Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        # cv2.putText(frame, "Current Pedestrian Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        # cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.putText(frame, "Total " + args["class"].capitalize() + " Counter: "+str(
            count), (int(10), int(60)), 0, 5e-3 * 100, (0, 255, 0), 1)
        cv2.putText(frame, "Current " + args["class"].capitalize() + " Counter: "+str(
            i), (int(10), int(40)), 0, 5e-3 * 100, (0, 255, 0), 1)
        cv2.putText(frame, "FPS: %f" % (fps), (10, 20),
                    0, 5e-3 * 100, (0, 255, 0), 1)
        cv2.namedWindow("YOLO4_Deep_SORT", 0)
        cv2.resizeWindow('YOLO4_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO4_Deep_SORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1./(time.time()-t1))) / 2
        # out.write(frame)
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
        # print(args["input"][43:57]+": " + str(count) +
        #       " " + str(class_name) + ' Found')
        print(str(count) + " " + str(class_name) + ' Found')

    else:
        print("[No Found]")
        #print("[INFO]: model_image_size = (960, 960)")
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gd, grade_system = fuzzy_system()
    main(YOLO())
