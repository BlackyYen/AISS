import numpy as np
from cv2 import cv2
from scipy.spatial.distance import pdist


def curvature(x2, y2):
    x = (x2[0] - x2[0 - 1], y2[0] - y2[0 - 1])
    y = (x2[0 + 1] - x2[0], y2[0 + 1] - y2[0])
    d = 1 - pdist([x, y], 'cosine')
    sin = np.sqrt(1 - d**2)
    dis = np.sqrt((x2[0 - 1] - x2[0 + 1])**2 + (y2[0 - 1] - y2[0 + 1])**2)
    k = 2 * sin / dis
    return k


def path_curvature(frame, pts, track, maxlen, point_visualization=True):
    '''
    frame：輸入之圖像
    pts：追蹤器之所有資訊
    track：單一追蹤對象
    maxlen：追蹤路徑之最大長度
    point_visualization：是否劃出座標點
    '''

    # 計算曲率
    curv_total = 0
    x_coor_1, y_coor_1 = [], []
    x_coor_2, y_coor_2 = [], []
    x_coor_3, y_coor_3 = [], []

    # 等比例從軌跡中提取7個座標點
    for k in range(7):
        x_coor_1.append(pts[track.track_id][maxlen // 6 * k][0])
        y_coor_1.append(pts[track.track_id][maxlen // 6 * k][1])
    # 等比例從軌跡中提取5個座標點
    for k in range(5):
        x_coor_2.append(pts[track.track_id][maxlen // 4 * k][0])
        y_coor_2.append(pts[track.track_id][maxlen // 4 * k][1])
    # 等比例從軌跡中提取3個座標點
    for k in range(3):
        x_coor_3.append(pts[track.track_id][maxlen // 2 * k][0])
        y_coor_3.append(pts[track.track_id][maxlen // 2 * k][1])
    if point_visualization:
        for index in range(len(x_coor_1)):
            cv2.circle(frame, (x_coor_1[index], y_coor_1[index]), 1,
                       [255, 0, 0], 8)
        for index in range(len(x_coor_2)):
            cv2.circle(frame, (x_coor_2[index], y_coor_2[index]), 1,
                       [0, 255, 0], 5)
        for index in range(len(x_coor_3)):
            cv2.circle(frame, (x_coor_3[index], y_coor_3[index]), 1,
                       [0, 0, 255], 2)
    # 從7個座標點中計算5個曲率
    freq = 0
    curv1 = 0
    for k in range(5):
        coor_cur = [x_coor_1[k + 2], y_coor_1[k + 2]]
        coor_mid = [x_coor_1[k + 1], y_coor_1[k + 1]]
        coor_pre = [x_coor_1[k], y_coor_1[k]]
        if coor_cur[0] == coor_mid[0] and coor_cur[1] == coor_mid[1]:
            continue
        elif coor_mid[0] == coor_pre[0] and coor_mid[1] == coor_pre[1]:
            continue
        elif coor_pre[0] == coor_cur[0] and coor_pre[1] == coor_cur[1]:
            continue
        cur = curvature(x2=(coor_cur[0], coor_mid[0], coor_pre[0]),
                        y2=(coor_cur[1], coor_mid[1], coor_pre[1]))[0]
        if cur >= 0.1:
            continue
        curv1 += cur
        freq += 1
    if freq != 0:
        curv_total += curv1 / freq * 2

    # 從5個座標點中計算3個曲率
    freq = 0
    curv2 = 0
    for k in range(3):
        coor_cur = [x_coor_2[k + 2], y_coor_2[k + 2]]
        coor_mid = [x_coor_2[k + 1], y_coor_2[k + 1]]
        coor_pre = [x_coor_2[k], y_coor_2[k]]
        if coor_cur[0] == coor_mid[0] and coor_cur[1] == coor_mid[1]:
            continue
        elif coor_mid[0] == coor_pre[0] and coor_mid[1] == coor_pre[1]:
            continue
        elif coor_pre[0] == coor_cur[0] and coor_pre[1] == coor_cur[1]:
            continue
        cur = curvature(x2=(coor_cur[0], coor_mid[0], coor_pre[0]),
                        y2=(coor_cur[1], coor_mid[1], coor_pre[1]))[0]
        if cur >= 0.1:
            continue
        curv2 += cur
        freq += 1
    if freq != 0:
        curv_total += curv2 / freq * 3

    # 從3個座標點中計算1個曲率
    freq = 0
    curv3 = 0
    for k in range(1):
        coor_cur = [x_coor_3[k + 2], y_coor_3[k + 2]]
        coor_mid = [x_coor_3[k + 1], y_coor_3[k + 1]]
        coor_pre = [x_coor_3[k], y_coor_3[k]]
        if coor_cur[0] == coor_mid[0] and coor_cur[1] == coor_mid[1]:
            continue
        elif coor_mid[0] == coor_pre[0] and coor_mid[1] == coor_pre[1]:
            continue
        elif coor_pre[0] == coor_cur[0] and coor_pre[1] == coor_cur[1]:
            continue
        cur = curvature(x2=(coor_cur[0], coor_mid[0], coor_pre[0]),
                        y2=(coor_cur[1], coor_mid[1], coor_pre[1]))[0]
        if cur >= 0.1:
            continue
        curv3 += cur
        freq += 1
    if freq != 0:
        curv_total += curv3 / freq * 5

    return frame, curv_total
