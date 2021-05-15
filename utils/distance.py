import numpy as np


def distance(coor_cur, coor_pre):
    vec_x = coor_cur[0] - coor_pre[0]
    vec_y = coor_cur[1] - coor_pre[1]
    vec = np.array([vec_x, vec_y])
    dist = np.linalg.norm(vec)
    return dist


def moving_distance(frame, pts, track):
    dist_total = 0
    for k in range(len(pts[track.track_id]) - 1):
        # 提取路徑起點與終點
        coor_cur = pts[track.track_id][k + 1]
        coor_pre = pts[track.track_id][k]
        # 計算距離的向量
        dist = distance(coor_cur, coor_pre)
        # 計算距離的向量長度
        dist_total += dist
    return frame, dist_total
