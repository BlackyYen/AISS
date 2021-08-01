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
from tensorflow import InteractiveSession
from keras.models import model_from_json
from PyQt5 import QtGui

from utils.distance import moving_distance
from utils.curvature import path_curvature
from utils.fuzzy import fuzzy_system, motility_system

from keras.applications.mobilenet import preprocess_input
from utils.images_resize import resize_keep_aspectratio

# visem_01_14
# visem_02_14
# visem_03_23
# visem_04_38
# visem_05_47
# visem_06_33
# visem_07_39

video_name = r'visem_05_47.mp4'
output_name = video_name
video_dir = r'./video/sperm'
output_dir = r'./video/results'

video_path = os.path.join(video_dir, video_name)
output_path = os.path.join(output_dir, output_name)

video_capture = cv2.VideoCapture(video_path)


def main(yolo,
         video_capture=video_capture,
         output_path=output_path,
         yolo_box=False,
         label=False,
         deepsort_box=True,
         center_point=True,
         item_number_old=False,
         item_number_new=False,
         trajectory=True,
         show_moving_dist=False,
         show_path_curv=False,
         point_visualization=False,
         fuzzy=True,
         mobilenetv2=True,
         label_show_image=None,
         lcdNumber_1=None,
         lcdNumber_2=None,
         lcdNumber_3=None,
         information=False):
    '''
    yolo_box: 劃出yolo辨識框
    label: 劃出物件名稱
    deepsort_box: 劃出deepsort預測框
    item_number_old: 劃出舊的ID
    item_number_new: 劃出新的ID
    trajectory: 劃出軌跡
    show_moving_dist: 在每隻精子旁顯示移動距離
    show_path_curv: 在每支精子旁顯示路徑軌跡
    '''

    g1 = tf.Graph()  # 加载到Session 1的graph
    g2 = tf.Graph()  # 加载到Session 2的graph

    sess1 = tf.Session(graph=g1)  # Session1
    sess2 = tf.Session(graph=g2)  # Session2

    with sess1.as_default():
        with g1.as_default():
            # load the model
            # load json and create model
            json_file = open(
                r'./model_weights/model-mobilenetv2-cbam-hushem-gray-ratio4-alpha1e-4-dropout0.5-ep100-lr1e-3.json',
                'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model_mobilenet = model_from_json(loaded_model_json)
            # load weights into new model
            model_mobilenet.load_weights(
                r'./model_weights/mobilenetv2-ep0100-acc1.0000-loss0.0017-val_acc0.8931-val_loss0.4444.h5'
            )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # 設定軌跡長度
    maxlen = 61
    pts = [deque(maxlen=maxlen) for _ in range(9999)]
    warnings.filterwarnings('ignore')

    # initialize a list of colors to represent each possible class label
    np.random.seed(2021)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
    #list = [[] for _ in range(100)]

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

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
        list_file = open(r'./detection_information/detection_rslt.txt', 'w')
        frame_index = -1

    fps = 0.0
    real_id = 0
    data = {}
    data_id = {}

    data_dist = []
    data_curv = []
    fps_list = []

    # 定義歸屬函數與模糊規則庫
    motility_base, motility_ = fuzzy_system()

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        try:
            org_img = frame.copy()
        except AttributeError:
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
            if yolo_box:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (0, 0, 0), 2)
            # print(class_names)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            # 第幾幀
            # print(frame_index)

            # 將辨識資訊寫入記事本裡
            list_file.write(str(frame_index) + ',')
            list_file.write(str(track.track_id) + ',')
            b0 = str(bbox[0]
                     )  #.split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1]
                     )  #.split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2] - bbox[0]
                     )  #.split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3] - bbox[1])
            list_file.write(
                str(b0) + ',' + str(b1) + ',' + str(b2) + ',' + str(b3))
            # 物件編號
            # print(str(track.track_id))
            # 換行
            list_file.write('\n')
            #list_file.write(str(track.track_id)+',')

            # 劃出deepsort預測框
            if deepsort_box:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (color), 2)

            # 舊的物件編號
            if item_number_old:
                cv2.putText(frame, str(track.track_id),
                            (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150,
                            (color), 2)
            # 新的物件編號（不會跳號）
            elif item_number_new:
                try:
                    cv2.putText(frame, str(data_id[track.track_id]),
                                (int(bbox[0]), int(bbox[1] - 8)), 0,
                                5e-3 * 125, (color), 2)
                except KeyError:
                    real_id += 1
                    data_id.setdefault(track.track_id, real_id)
                    cv2.putText(frame, str(data_id[track.track_id]),
                                (int(bbox[0]), int(bbox[1] - 8)), 0,
                                5e-3 * 125, (color), 2)

            if len(class_names) > 0:
                class_name = class_names[0]
                # 畫出物件名稱
                if label:
                    cv2.putText(frame, str(class_names[0]),
                                (int(bbox[0]), int(bbox[1] - 8)), 0,
                                5e-3 * 125, (0, 0, 0), 2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(
                ((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            #track_id[center]

            # 將每一幀之中心位置儲存起來
            pts[track.track_id].append(center)

            # 框中心點之大小
            thickness = 5
            # 劃出框中心點
            if center_point:
                cv2.circle(frame, (center), 1, color, thickness)

            # 劃出移動路徑
            if trajectory:
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[
                            track.track_id][j] is None:
                        continue
                    # 移動路徑由細至粗
                    # thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    # 固定路徑粗細
                    thickness = 2
                    cv2.line(frame, (pts[track.track_id][j - 1]),
                             (pts[track.track_id][j]), (color), thickness)

            # 計算移動距離與轉向角
            # pts[track.track_id][(len(pts[track.track_id])-1)] 為當前偵之座標
            if len(pts[track.track_id]) == maxlen:
                # 計算移動距離
                frame, moving_dist = moving_distance(frame, pts, track)
                # 計算路徑曲率
                frame, path_curv = path_curvature(
                    frame,
                    pts,
                    track,
                    maxlen,
                    point_visualization=point_visualization)
                # 將距離四捨五入至小數點第二位
                moving_dist = round(moving_dist, 2)
                data_dist.append(moving_dist)
                # 將曲率四捨五入至小數點第二位
                path_curv = round(path_curv, 2)
                data_curv.append(path_curv)

                # 將計算資訊顯示至精子旁
                if show_moving_dist:
                    cv2.putText(frame, '{:.2f}'.format(moving_dist),
                                (int(center[0] + 20), int(center[1])), 0,
                                5e-3 * 100, (0, 0, 139), 2)  # Red4
                if show_path_curv:
                    cv2.putText(frame, '{:.2f}'.format(path_curv),
                                (int(center[0] + 20), int(center[1] + 20)), 0,
                                5e-3 * 100, (139, 0, 0), 2)  # Blue4

                # 模糊推論系統
                if fuzzy:
                    # fuzzy_system
                    score = motility_system(
                        motility_base=motility_base,
                        motility=motility_,
                        moving_distance=moving_dist,
                        path_curvature=path_curv,
                    )
                    score = round(score, 2)

                    try:
                        # 將分數顯示至精子旁
                        cv2.putText(frame, '{:.2f}'.format(score),
                                    (int(center[0] + 20), int(center[1] + 5)),
                                    0, 5e-3 * 100, (0, 0, 0),
                                    2)  # purple4 (139, 26, 85)
                        # 劃出框
                        # if data[track.track_id][3][0] >= 0 and data[
                        #         track.track_id][3][0] <= 20:
                        #     cv2.rectangle(frame,
                        #                   (int(bbox[0] - 5), int(bbox[1]) - 5),
                        #                   (int(bbox[2] + 5), int(bbox[3] + 5)),
                        #                   [255, 0, 0], 2)
                        # elif data[track.track_id][3][0] <= 40:
                        #     cv2.rectangle(frame,
                        #                   (int(bbox[0] - 5), int(bbox[1]) - 5),
                        #                   (int(bbox[2] + 5), int(bbox[3] + 5)),
                        #                   [0, 255, 242], 2)
                        # elif data[track.track_id][3][0] <= 60:
                        #     cv2.rectangle(frame,
                        #                   (int(bbox[0] - 5), int(bbox[1]) - 5),
                        #                   (int(bbox[2] + 5), int(bbox[3] + 5)),
                        #                   [0, 255, 0], 2)
                        # elif data[track.track_id][3][0] <= 80:
                        #     cv2.rectangle(frame,
                        #                   (int(bbox[0] - 5), int(bbox[1] - 5)),
                        #                   (int(bbox[2] + 5), int(bbox[3] + 5)),
                        #                   [0, 0, 255], 2)
                    except:
                        continue
                if mobilenetv2:
                    try:
                        # 裁切圖片
                        x1 = int(bbox[0])
                        x2 = int(bbox[2])
                        y1 = int(bbox[1])
                        y2 = int(bbox[3])
                        # if x1 < 0:
                        #     x1 = 0
                        crop_img = org_img[y1:y2, x1:x2]
                        # mobilenet 分類
                        with sess1.as_default():
                            with sess1.graph.as_default():
                                # cv2.imwrite('classImg.jpg',crop_img)
                                # img_input = cv2.imread('classImg.jpg')
                                head_img = resize_keep_aspectratio(crop_img)
                                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                head_img = np.expand_dims(head_img, axis=0)
                                head_img = preprocess_input(head_img)
                                preds = model_mobilenet.predict(head_img)
                                predictions = np.argmax(preds, axis=1)[0]
                                classes = [
                                    'normal', 'tapered', 'pyriform',
                                    'amorphous'
                                ]
                        cv2.putText(frame, str(classes[predictions]),
                                    (x1, y1 - 8), 0, 5e-3 * 120, (0, 0, 0), 2)
                    except:
                        pass
        # 顯示偵測相關資訊
        count = len(set(counter))
        try:
            new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            new_frame = cv2.resize(new_frame, (512, 384))
            showImage = QtGui.QImage(new_frame.data, new_frame.shape[1],
                                     new_frame.shape[0],
                                     QtGui.QImage.Format_RGB888)
            label_show_image.setPixmap(QtGui.QPixmap.fromImage(showImage))
            lcdNumber_1.display(round(fps))
            lcdNumber_2.display(i)
            lcdNumber_3.display(count)
        except:
            if information:
                cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 25), 0,
                            5e-3 * 125, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    'Current ' + 'sperm'.capitalize() + ' Counter: ' + str(i),
                    (10, 50), 0, 5e-3 * 125, (0, 255, 0), 2)
                cv2.putText(
                    frame, 'Total ' + 'sperm'.capitalize() + ' Counter: ' +
                    str(count), (10, 75), 0, 5e-3 * 125, (0, 255, 0), 2)
            cv2.namedWindow('YOLOv4_DeepSORT', 0)
            cv2.resizeWindow('YOLOv4_DeepSORT', 1024, 768)
            cv2.imshow('YOLOv4_DeepSORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1. / (time.time() - t1))) / 2
        fps_list.append(fps)
        out.write(frame)
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(' ')
    print('[Finish]')
    end = time.time()

    print('distance max:', max(data_dist))
    print('curvature max:', max(data_curv))
    print('FPS:', round(np.mean(fps_list), 2))

    if len(pts[track.track_id]) != None:
        print(str(count) + ' ' + str(class_name) + ' Found')

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
