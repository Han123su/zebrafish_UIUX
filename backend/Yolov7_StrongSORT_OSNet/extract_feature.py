import numpy as np
import scipy
from math import dist
from scipy.optimize import linear_sum_assignment

def generate_feature(line):
    ori_x, ori_y = np.array(line[:, 1]), np.array(line[:, 0])

    total_length = len(ori_x)
    body_idx = total_length / 3
    center_idx = int(total_length / 2)
    if total_length % 3 > 0:
        body_x = [ori_x[0], ori_x[int(body_idx)], ori_x[int(body_idx * 2)], ori_x[int(body_idx * 3) - 1]]
        body_y = [ori_y[0], ori_y[int(body_idx)], ori_y[int(body_idx) * 2], ori_y[int(body_idx * 3) - 1]]
    else:
        body_idx = int(body_idx)
        body_x = [ori_x[0], ori_x[(body_idx) - 1], ori_x[(body_idx * 2) - 1], ori_x[body_idx * 3 - 1]]
        body_y = [ori_y[0], ori_y[(body_idx) - 1], ori_y[(body_idx * 2) - 1], ori_y[body_idx * 3 - 1]]
    if total_length % 2 == 0:
        center_idx = center_idx - 1
    body_cx = [ori_x[center_idx]]
    body_cy = [ori_y[center_idx]]

    return np.c_[list(map(int, body_x)), list(map(int, body_y))], center_idx, np.c_[
        list(map(int, body_cx)), list(map(int, body_cy))]


def get_body_to_tail(h_idx, skeleton_idx, l):
    if len(skeleton_idx) % 2 == 0:
        if h_idx == 0:
            body_to_tail = skeleton_idx[l - 1:]
        else:
            body_to_tail = skeleton_idx[0:l]
    else:
        if h_idx == 0:
            body_to_tail = skeleton_idx[l:]
        else:
            body_to_tail = skeleton_idx[0:l + 1]

    return body_to_tail


def azimuthOrient(curr, img):
    h, w = np.shape(img)
    w_size = 32
    # 當前位置- 先前位置判斷方向  and 頭尾兩點-中心點去判斷對不對 累積個4次
    angle = []
    area = []
    for i in range(0, len(curr), 3):
        # 面積判斷
        right, bottom = curr[i] + w_size
        left, top = curr[i] - w_size
        if right > w:
            rr = right - w
            left -= rr
            right = w
        if bottom > h:
            br = bottom - h
            top -= br
            bottom = h
        if left < 0:
            lr = left + 1
            right += (-lr)
            left = 0
        if top < 0:
            tr = top + 1
            bottom += (-tr)
            top = 0
        crop = img[top:bottom, left:right]
        c_area = crop[crop > 0].size
        area.append(c_area)

    if area[0] > area[1]:
        return 0, curr
    else:
        curr = curr[::-1]
        return 2, curr


def get_endpoint(skeleton):
    # Kernel to sum the neighbours
    kernel = [[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]]
    # 2D convolution (cast image to int32 to avoid overflow)
    img_conv = scipy.signal.convolve2d(skeleton.astype(np.int32), kernel, mode='same')
    # Pick points where pixel is 255 and neighbours sum 255
    endpoints = np.stack(np.where((skeleton == 255) & (img_conv == 255)), axis=1)

    return endpoints


def assignment_point(track, detect):
    tracks = track
    detections = detect
    if len(tracks) == 0:
        tracks = detections

    N = len(tracks)
    _position = [[] for i in range(N)]
    cost = [[] for i in range(N)]
    for i in range(N):
        for j in range(N):
            cost[i].append(dist(tracks[i], detections[j]))
    cost = np.array(cost)
    row_indices, col_indices = linear_sum_assignment(cost)  # row tracking ,,, col detection

    for i in range(len(_position)):
        track_id = row_indices[i]
        detect_id = col_indices[i]
        _position[track_id] = detections[detect_id]
    return np.array(_position)
    # return np.transpose(np.array(_position), [1, 0])


def curve_rate(curve):
    # BC 弧曲率
    x = np.sqrt(np.diff(curve[:, 1]) ** 2 + np.diff(curve[:, 0]) ** 2)
    real_len = np.sum(x)
    straight_len = dist(curve[0], curve[-1])
    bc_length = real_len / straight_len

    return np.round(bc_length, 5), straight_len


def get_AUC(A, length):
    # formula
    # (length**2) / 4 / length
    return A + (length ** 2) / 4 / length
