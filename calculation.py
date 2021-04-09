from __future__ import division, print_function, absolute_import
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

video_name = r'sperm_01_14.mp4'
video_path = r'./video/sperm'
results_path = r'./video/results'
maxlen = 30

ap = argparse.ArgumentParser()
ap.add_argument('-i',
                '--input',
                help='path to input video',
                default=os.path.join(video_path, video_name))
ap.add_argument('-c', '--class', help='name of class', default='sperm')
args = vars(ap.parse_args())

pts = [deque(maxlen=maxlen) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(2021)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
#list = [[] for _ in range(100)]


def main(yolo):

    start = time.time()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    counter = []
    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    find_objects = ['person']
    metric = nn_matching.NearestNeighborDistanceMetric('cosine',
                                                       max_cosine_distance,
                                                       nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(args['input'])

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(results_path, video_name), fourcc,
                              30, (w, h))
        list_file = open(r'./detection_information/detection_rslt.txt', 'w')
        frame_index = -1

    fps = 0.0
    real_id = 0
    data = {}
    data_id = {}

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        #image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  #bgr to rgb
        boxs, confidence, class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [
            Detection(bbox, 1.0, feature)
            for bbox, feature in zip(boxs, features)
        ]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap,
                                                    scores)
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
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            #print(class_names)
            #print(class_names[p])

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            #print(frame_index)
            list_file.write(str(frame_index) + ',')
            list_file.write(str(track.track_id) + ',')
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (color), 2)
            b0 = str(bbox[0]
                     )  #.split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1]
                     )  #.split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2] - bbox[0]
                     )  #.split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3] - bbox[1])

            list_file.write(
                str(b0) + ',' + str(b1) + ',' + str(b2) + ',' + str(b3))
            #print(str(track.track_id))
            list_file.write('\n')
            #list_file.write(str(track.track_id)+',')

            # 原始的物件編號
            # cv2.putText(frame, str(track.track_id),
            #             (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150,
            #             (color), 2)
            # 新的物件編號（不會跳號）
            try:
                cv2.putText(frame, str(data_id[track.track_id]),
                            (int(bbox[0]), int(bbox[1] - 10)), 0, 5e-3 * 120,
                            (color), 2)
            except KeyError:
                real_id += 1
                data_id.setdefault(track.track_id, real_id)
                cv2.putText(frame, str(data_id[track.track_id]),
                            (int(bbox[0]), int(bbox[1] - 10)), 0, 5e-3 * 120,
                            (color), 2)

            if len(class_names) > 0:
                class_name = class_names[0]
            # 畫出物件名稱
            #     cv2.putText(frame, str(class_names[0]),
            #                 (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150,
            #                 (color), 2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(
                ((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            #track_id[center]

            pts[track.track_id].append(center)

            thickness = 5
            #center point
            cv2.circle(frame, (center), 1, color, thickness)

            # draw motion path
            # 劃出路徑
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[
                        track.track_id][j] is None:
                    continue
                # thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                thickness = 2
                cv2.line(frame, (pts[track.track_id][j - 1]),
                         (pts[track.track_id][j]), (color), thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

            # 計算移動距離與轉向角
            # pts[track.track_id][(len(pts[track.track_id])-1)] 當前偵之座標
            if len(pts[track.track_id]) == maxlen:
                # 劃出起點與終點
                cv2.circle(
                    frame,
                    (pts[track.track_id][(len(pts[track.track_id]) - 1)]), 1,
                    [0, 0, 0], 2)
                cv2.circle(
                    frame,
                    (pts[track.track_id][(len(pts[track.track_id]) - 1) -
                                         maxlen // 2]), 1, [0, 0, 0], 2)
                cv2.circle(
                    frame,
                    (pts[track.track_id][(len(pts[track.track_id]) - 1) -
                                         (maxlen - 1)]), 1, [0, 0, 0], 2)
                coordinate_cur = (pts[track.track_id][(
                    len(pts[track.track_id]) - 1)])
                coordinate_15 = (
                    pts[track.track_id][(len(pts[track.track_id])) - 15])
                coordinate_30 = (
                    pts[track.track_id][(len(pts[track.track_id])) - 30])
                # 計算總移動距離
                # for l in range(1,len(pts[track.track_id])+1):
                #     pts[track.track_id][(len(pts[track.track_id]) - l)]
                # 一個計算距離的向量
                x1 = coordinate_cur[0] - coordinate_30[0]
                y1 = coordinate_cur[1] - coordinate_30[1]
                xy1 = np.array([x1, y1])
                # 兩個計算角度差的向量
                x2 = coordinate_cur[0] - coordinate_15[0]
                y2 = coordinate_cur[1] - coordinate_15[1]
                xy2 = np.array([x2, y2])
                x3 = coordinate_15[0] - coordinate_30[0]
                y3 = coordinate_15[1] - coordinate_30[1]
                xy3 = np.array([x3, y3])
                # 計算向量長度
                d1 = np.sqrt(xy1.dot(xy1))
                d2 = np.sqrt(xy2.dot(xy2))
                d3 = np.sqrt(xy3.dot(xy3))
                # 計算xy2與xy3兩個向量之間的角度差
                cos_angle = xy2.dot(xy3) / (d2 * d3)
                angle = np.arccos(cos_angle)
                angle2 = angle * 360 / 2 / np.pi
                # 距離
                d = round(d1, 1)
                # 角度
                a = round(angle2, 1)
                if str(a) == 'nan':
                    a = 0.0
                cv2.putText(frame, str(d),
                            (int(center[0] + 15), int(center[1] - 5)), 0,
                            5e-3 * 100, (0, 0, 0), 2)
                cv2.putText(frame, str(a),
                            (int(center[0] + 15), int(center[1] + 15)), 0,
                            5e-3 * 100, (0, 0, 0), 2)

        # 顯示偵測相關資訊
        count = len(set(counter))
        # cv2.putText(frame, 'Total Pedestrian Counter: ' + str(count),
        #             (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0), 2)
        # cv2.putText(frame, 'Current Pedestrian Counter: ' + str(i),
        #             (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        # cv2.putText(frame, 'FPS: %f' % (fps), (int(20), int(40)), 0,
        #             5e-3 * 200, (0, 255, 0), 3)
        cv2.putText(frame, 'FPS: %f' % (fps), (10, 25), 0, 5e-3 * 125,
                    (0, 255, 0), 2)
        cv2.putText(
            frame,
            'Current ' + args['class'].capitalize() + ' Counter: ' + str(i),
            (10, 50), 0, 5e-3 * 125, (0, 255, 0), 2)
        cv2.putText(
            frame,
            'Total ' + args['class'].capitalize() + ' Counter: ' + str(count),
            (10, 75), 0, 5e-3 * 125, (0, 255, 0), 2)
        cv2.namedWindow('YOLOv4_DeepSORT', 0)
        cv2.resizeWindow('YOLOv4_DeepSORT', 1024, 768)
        cv2.imshow('YOLOv4_DeepSORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1. / (time.time() - t1))) / 2
        out.write(frame)
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(' ')
    print('[Finish]')
    end = time.time()

    if len(pts[track.track_id]) != None:
        print(args['input'][43:57] + ': ' + str(count) + ' ' +
              str(class_name) + ' Found')

    else:
        print('[No Found]')


#print('[INFO]: model_image_size = (960, 960)')
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
