#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
from cv2 import cv2
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
from math import sqrt 

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from fuzzy_system import fuzzy_system, grade

#%%
mode = 2
# m = d or a
m = 'd' # d and a  
video_path  = 'test_video/sperm3.mp4'
output_path = 'test_video_out/'
output_name = 'test.avi'
my_maxlen   = 40
#%%
backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = video_path)
ap.add_argument("-c", "--class",help="name of class", default = 'sperm')
args = vars(ap.parse_args())
# maxlen至少是2，兩點連成一直線
pts = [deque(maxlen = my_maxlen) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(29)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

def main(yolo):
    start = time.time()
    #Definition of the parameters
    max_cosine_distance = 0.5 #0.9 余弦距离的控制阈值
    max_euclidean_distance = 2
    nn_budget = None #将每个类别的样本最多固定为该数目。 达到预算后，删除最旧的样本。
    nms_max_overlap = 0.5#非极大抑制的阈值

    counter = []
    #deep_sort
    #model_filename = 'model_data/market1501.pb'
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=50)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# =============================================================================
#     metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_euclidean_distance, nn_budget)
# =============================================================================
    tracker = Tracker(metric)

    writeVideo_flag = True
# =============================================================================
#     video_capture = cv2.VideoCapture(0)
# =============================================================================
    video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path + args["input"][43:57] + args["class"] + '_' + output_name, fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    real_id = 0
    n = 0
    nw = 0
    nh1 = 0
    nh2 = 0
    nh3 = 0
    data = {}
    data_id = {}
    ct = []
    
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        try:
            org = frame.copy()
        except AttributeError:
            break
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs,class_names = yolo.detect_image(image)
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
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
            # 畫出yolo辨識框
            #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            
            if mode == 0 or mode == 1 :
                #劃出框
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 2)
                # 劃出物件ID
                try:
                    cv2.putText(frame,str(data_id[track.track_id]),(int(bbox[0]), int(bbox[1] - 10)),0, 5e-3 * 120, (color),2)
                except KeyError:
                    real_id += 1
                    data_id.setdefault(track.track_id, real_id)
                    cv2.putText(frame,str(data_id[track.track_id]),(int(bbox[0]), int(bbox[1] - 10)),0, 5e-3 * 120, (color),2)
                    print(data_id[track.track_id])
                #cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] - 10)),0, 5e-3 * 150, (color),2)
            #%%
            if len(class_names) > 0:
               class_name = class_names[0]
               obj_name_split = str(class_name).split("'")
               obj_name = obj_name_split[1]
               # 劃出物件名稱
               if mode == 0 or mode == 1 :
                   cv2.putText(frame, str(obj_name),(int(bbox[0]+30), int(bbox[1] - 10)),0, 5e-3 * 120, (color),2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            
            if mode == 0 or mode == 1 :
                #center point
                #劃出重心
                cv2.circle(frame,  (center), 1, color, thickness)
        	    #draw motion path
                #劃出路徑
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                       continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),2)
                    #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)
            #%%
            #將每偵中心點儲存為list
# =============================================================================
#             for j in range(1, len(pts[track.track_id])):
#                 if j == 1:
#                     x.append(pts[track.track_id][j-1][0])
#                     y.append(pts[track.track_id][j-1][1])
#                 if x[(len(x)-1)] != pts[track.track_id][j][0] or y[(len(y)-1)] != pts[track.track_id][j][1]:
#                     x.append(pts[track.track_id][j][0])
#                     y.append(pts[track.track_id][j][1])
# =============================================================================
            
            fr = my_maxlen
            # pts[track.track_id][(len(pts[track.track_id])-1)] 當前偵之座標
            if len(pts[track.track_id]) == fr :
                # 劃出起點與終點
                if mode == 1:
                    cv2.circle(frame, (pts[track.track_id][(len(pts[track.track_id])-1)]),
                                       1,[0, 0, 0], 2)
                    cv2.circle(frame, (pts[track.track_id][(len(pts[track.track_id])-1) - my_maxlen//2]),
                                       1,[0, 0, 0], 2)
                    cv2.circle(frame, (pts[track.track_id][(len(pts[track.track_id])-1) - (my_maxlen-1)]),
                                       1,[0, 0, 0], 2)
                    coordinate_cur = (pts[track.track_id][(len(pts[track.track_id]) - 1)])
                    coordinate_5  = (pts[track.track_id][(len(pts[track.track_id])) - 5])
                    coordinate_10 = (pts[track.track_id][(len(pts[track.track_id])) - 10])
                    coordinate_15 = (pts[track.track_id][(len(pts[track.track_id])) - 15])
                    coordinate_20 = (pts[track.track_id][(len(pts[track.track_id])) - 20])
                    coordinate_40 = (pts[track.track_id][(len(pts[track.track_id])) - 40])
                    # 計算總移動距離
# =============================================================================
#                     for l in range(1,len(pts[track.track_id])+1):
#                         pts[track.track_id][(len(pts[track.track_id]) - l)]-
# =============================================================================
                    # 一個計算距離的向量
                    x1  = coordinate_cur[0]-coordinate_40[0]
                    y1  = coordinate_cur[1]-coordinate_40[1]
                    xy1 = np.array([x1,y1])
                    # 兩個計算角度差的向量
                    x2  = coordinate_cur[0]-coordinate_20[0]
                    y2  = coordinate_cur[1]-coordinate_20[1]
                    xy2 = np.array([x2,y2])
                    x3  = coordinate_20[0]-coordinate_40[0]
                    y3  = coordinate_20[1]-coordinate_40[1]
                    xy3 = np.array([x3,y3])
                    # 計算向量長度
                    d1=np.sqrt(xy1.dot(xy1))
                    d2=np.sqrt(xy2.dot(xy2))
                    d3=np.sqrt(xy3.dot(xy3))
                    # 計算xy2與xy3兩個向量之間的角度差
                    cos_angle=xy2.dot(xy3)/(d2*d3)
                    angle=np.arccos(cos_angle)
                    angle2=angle*360/2/np.pi
                    # 距離
                    d = round(d1,1)
                    # 角度
                    a = round(angle2, 1)
                    if str(a) == 'nan':
                        a = 0.0
                    if m == 'd':
                        cv2.putText(frame, str(d),(int(center[0] + 6), int(center[1] + 6)),0, 5e-3 * 120, (0,0,0),2)
                    if m == 'a':
                        cv2.putText(frame, str(a),(int(center[0] + 6), int(center[1] + 6)),0, 5e-3 * 120, (0,0,0),2)
                #%%
                # data[track.track_id][0] ID
                # data[track.track_id][1] distance
                # data[track.track_id][2] angle
                # data[track.track_id][3] fuzzy output
                if mode == 2:
                    # 計算向量
                    coordinate_cur = (pts[track.track_id][(len(pts[track.track_id]) - 1)])
                    coordinate_5  = (pts[track.track_id][(len(pts[track.track_id])) - 5])
                    coordinate_10 = (pts[track.track_id][(len(pts[track.track_id])) - 10])
                    coordinate_15 = (pts[track.track_id][(len(pts[track.track_id])) - 15])
                    coordinate_20 = (pts[track.track_id][(len(pts[track.track_id])) - 20])
                    coordinate_40 = (pts[track.track_id][(len(pts[track.track_id])) - 40])
                    #一個計算距離的向量
                    x1  = coordinate_cur[0]-coordinate_40[0]
                    y1  = coordinate_cur[1]-coordinate_40[1]
                    xy1 = np.array([x1,y1])
                    #兩個計算角度差的向量
                    x2  = coordinate_cur[0]-coordinate_20[0]
                    y2  = coordinate_cur[1]-coordinate_20[1]
                    xy2 = np.array([x2,y2])
                    x3  = coordinate_20[0]-coordinate_40[0]
                    y3  = coordinate_20[1]-coordinate_40[1]
                    xy3 = np.array([x3,y3])
                    # 計算向量長度
                    d1=np.sqrt(xy1.dot(xy1))
                    d2=np.sqrt(xy2.dot(xy2))
                    d3=np.sqrt(xy3.dot(xy3))
                    # 計算xy2與xy3兩個向量之間的角度差
                    cos_angle=xy2.dot(xy3)/(d2*d3)
                    angle=np.arccos(cos_angle)
                    angle2=angle*360/2/np.pi
                    # 距離
                    d = round(d1,1)
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
                    # 建立字典
                    try:
                        data[track.track_id][0][track.track_id]
                    except KeyError:
                        real_id += 1
                        data_id.setdefault(track.track_id, real_id)
                        data_list = [data_id, [], [], []]
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
                        g = grade(grade_system = grade_system, input1 = d_ave, input2 = a_ave) 
                        # 刪除最舊的n筆資料
                        while(True):
                            del data[track.track_id][1][0], data[track.track_id][2][0]
                            if len(data[track.track_id][1]) == 15 and len(data[track.track_id][2]) == 15:
                                break
                        data[track.track_id][3].clear()
                        data[track.track_id][3].append(g)
                    try:
                        # color(bgr)
                        if data[track.track_id][3][0] >= 0 and data[track.track_id][3][0] <= 35:
                            cv2.rectangle(frame, (int(bbox[0]-5), int(bbox[1])-5), (int(bbox[2]+5), int(bbox[3]+5)),[255, 0, 0], 2)
                        elif data[track.track_id][3][0] <= 55:
                            cv2.rectangle(frame, (int(bbox[0]-5), int(bbox[1])-5), (int(bbox[2]+5), int(bbox[3]+5)),[0, 255, 242], 2)
                        elif data[track.track_id][3][0] <= 75:
                            cv2.rectangle(frame, (int(bbox[0]-5), int(bbox[1])-5), (int(bbox[2]+5), int(bbox[3]+5)),[0, 255, 0], 2)
                        elif data[track.track_id][3][0] <= 100:
                            cv2.rectangle(frame, (int(bbox[0]-5), int(bbox[1]-5)), (int(bbox[2]+5), int(bbox[3]+5)),[0, 0, 255], 2) 
                    except IndexError:
                        break
                    try:
                        ct[int(data[track.track_id][0][track.track_id]-1)]
                    except:
                        ct.append(data[track.track_id][0][track.track_id])
                        # 裁切圖片
                        x1 = int(bbox[0]-5)
                        x2 = int(bbox[2]+5)
                        y1 = int(bbox[1]-5)
                        y2 = int(bbox[3]+5)
                        ww = int(x2-x1)
                        hh = int(y2-y1)
                        crop_img = org[ y1 : y2, x1 : x2 ]
# =============================================================================
#                         gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#                         ret,b_img = cv2.threshold(gray_img, 160,255, cv2.THRESH_BINARY)
# =============================================================================
                        # cropped                   
# =============================================================================
#                         n += 1
#                         cv2.namedWindow("cropped%s"%n, 0)
#                         cv2.resizeWindow("cropped%s"%n, ww, hh)
#                         if n == 1:
#                             cv2.moveWindow("cropped%s"%n,int(110 + w), 100)
#                         elif n > 1:
#                             nh1 = nh1 + 120
#                             cv2.moveWindow("cropped%s"%n,int(110 + w), int(100 + nh1))
#                         cv2.imshow("cropped%s"%n, crop_img)
# =============================================================================
                        # gray
# =============================================================================
#                         cv2.namedWindow("gray%s"%n, 0)
#                         cv2.resizeWindow("gray%s"%n, ww, hh)
#                         if n == 1:
#                             cv2.moveWindow("gray%s"%n,int(240 + w), 100)
#                         elif n > 1:
#                             nh2 = nh2 + 120
#                             cv2.moveWindow("gray%s"%n,int(240 + w), int(100 + nh2))
#                         cv2.imshow("gray%s"%n, gray_img)
# =============================================================================
                        # binary
# =============================================================================
#                         cv2.namedWindow("binary%s"%n, 0)
#                         cv2.resizeWindow("binary%s"%n, ww, hh)
#                         if n == 1:
#                             cv2.moveWindow("binary%s"%n,int(370 + w), 100)
#                         elif n > 1:
#                             nh3 = nh3 + 120
#                             cv2.moveWindow("binary%s"%n,int(370 + w), int(100 + nh3))
#                         cv2.imshow("binary%s"%n, b_img)
# =============================================================================
        #%%
        count = len(set(counter))
        cv2.putText(frame, "Total " + args["class"].capitalize() + " Counter: "+str(count),(int(10), int(60)),0, 5e-3 * 100, (0,255,0),1)
        cv2.putText(frame, "Current " + args["class"].capitalize() + " Counter: "+str(i),(int(10), int(40)),0, 5e-3 * 100, (0,255,0),1)
        cv2.putText(frame, "FPS: %f"%(fps),(10, 20),0, 5e-3 * 100, (0,255,0),1)
        cv2.namedWindow("YOLO4_Deep_SORT", 0)
        cv2.resizeWindow("YOLO4_Deep_SORT",int(w), int(h))
        cv2.moveWindow("YOLO4_Deep_SORT",100,100)
        cv2.imshow('YOLO4_Deep_SORT', frame)

        #%%
        if writeVideo_flag:
            #save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

#%%
# fuzyy system
if __name__ == '__main__':
    gd, grade_system = fuzzy_system()
    main(YOLO())
