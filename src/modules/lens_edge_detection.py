import torch
from .color_space_conversion import rgb2yuv
from ..utils import generate_gaussian_kernel
from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import math


def edge_detection(y_input, gauss_kernel, threshold, max_value, device):
    _, _, h, w = y_input.shape
    gx_w = torch.tensor([[-2, -1, 0, 1, 2],
                         [-4, -2, 0, 2, 4],
                         [-2, -1, 0, 1, 2]]).float().to(device)
    gy_w = torch.tensor([[-1, -2, -4, -2, -1],
                         [0, 0, 0, 0, 0],
                         [1, 2, 4, 2, 1]]).float().to(device)
    edges = []
    interval_h = 200
    for sh in range(interval_h, h, interval_h):
        slice = y_input[..., sh - 5: sh + 6, :]
        gauss_result = torch.conv2d(slice, gauss_kernel)
        gx = torch.conv2d(gauss_result, gx_w[None, None, ...])
        gy = torch.conv2d(gauss_result, gy_w[None, None, ...])
        gradient = (gx ** 2 + gy ** 2) / 2

        max_value_tensor = torch.tensor(max_value).float().to(device)
        tan = torch.where(gx == 0, max_value_tensor, torch.abs(gy / gx))
        tensor_pi = torch.tensor((torch.pi)).float().to(device)
        tan225 = torch.tan(tensor_pi / 8)
        tan675 = torch.tan(tensor_pi / 2 - tensor_pi / 8)

        tensor0 = torch.tensor(0).float().to(device)
        x = torch.where(tan < tan225, 1, 0)
        y = torch.where(tan >= tan675, 2, 0)

        l_r = torch.where(tan >= tan225, tan, torch.tensor(1000).float().to(device))
        l_r = torch.where(l_r < tan675, 3, 0)

        l = torch.where(gx * gy > 0, l_r, 0)
        r = torch.where(gx * gy < 0, l_r * 4 / 3, tensor0)
        direction = x + y + l + r

        dx0 = torch.where(direction == 1, gradient - torch.roll(gradient, 1, -1), tensor0)
        dx0 = torch.where(dx0 > 0, 1, 0)
        dx1 = torch.where(direction == 1, gradient - torch.roll(gradient, -1, -1), tensor0)
        dx1 = torch.where(dx1 > 0, 1, 0)
        dx = dx0 * dx1

        dy0 = torch.where(direction == 2, gradient - torch.roll(gradient, 1, -2), tensor0)
        dy0 = torch.where(dy0 > 0, 1, 0)
        dy1 = torch.where(direction == 2, gradient - torch.roll(gradient, -1, -2), tensor0)
        dy1 = torch.where(dy1 > 0, 1, 0)
        dy = dy0 * dy1

        dl0 = torch.where(direction == 3, gradient - torch.roll(gradient, [-1, -1], [-2, -1]), tensor0)
        dl0 = torch.where(dl0 > 0, 1, 0)
        dl1 = torch.where(direction == 3, gradient - torch.roll(gradient, [1, 1], [-2, -1]), tensor0)
        dl1 = torch.where(dl1 > 0, 1, 0)
        dl = dl0 * dl1

        dr0 = torch.where(direction == 4, gradient - torch.roll(gradient, [-1, 1], [-2, -1]), tensor0)
        dr0 = torch.where(dr0 > 0, 1, 0)
        dr1 = torch.where(direction == 4, gradient - torch.roll(gradient, [1, -1], [-2, -1]), tensor0)
        dr1 = torch.where(dr1 > 0, 1, 0)
        dr = dr0 * dr1

        gradient = gradient * (dx + dy + dl + dr)
        e = torch.where(gradient > threshold, 255, 0)
        edges.append(e)

    edge_map = torch.zeros_like(y_input)
    for c, edge in zip(range(interval_h, h, interval_h), edges):
        edge_map[:, :, c - 1: c + 2, 5: w - 5] = edge

    edge_point = []
    for idx in range(interval_h, h, interval_h):
        line = edge_map[0, 0, idx, :]
        left = right = 0
        for i in range(0, line.shape[0]):
            if line[i] > 0:
                left = i
                break
        left = np.where(left == 0, 4095, left)
        edge_point.append([int(left), idx])
        for i in range(line.shape[0] - 1, -1, -1):
            if line[i] > 0:
                right = i
                break
        right = np.where(right == 0, 4095, right)
        edge_point.append([int(right), idx])

    return edge_point

def filter_valid_edges(points, distance_threshold=250):
    valid_points = []
    for i in range(0, len(points), 2):
        p1 = points[i]
        p2 = points[i+1]
        if p1[0] == 4095 and p2[0] == 4095:
            continue
        if p1[0] == 4095 and p2[0] != 4095:
            valid_points.append(p2)
            continue
        if p2[0] == 4095 and p1[0] != 4095:
            valid_points.append(p1)
            continue

        d_left = p1[0]
        d_right = 3840 - p2[0]
        if abs(d_right - d_left) > distance_threshold:
            valid_p = p1 if d_right > d_left else p2
            valid_points.append(valid_p)
        else:
            valid_points.extend([p1, p2])
    
    return valid_points

def find_circle_by_3pts(p1, p2, p3):
    v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    v2 = [p3[0] - p1[0], p3[1] - p1[1]]

    mp1 = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
    c1 = mp1[0] * v1[0] + mp1[1] * v1[1]
    mp2 = [(p1[0] + p3[0]) / 2.0, (p1[1] + p3[1]) / 2.0]
    c2 = mp2[0] * v2[0] + mp2[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    EPS = 1e-4
    if abs(det) <= EPS:
        d1 = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        d2 = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
        d3 = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
        radius = math.sqrt(max(d1, max(d2, d3))) * 0.5 + EPS
        if d1 >= d2 and d1 >= d3:
            center = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5]
        elif d2 >= d1 and d2 >= d3:
            center = [(p1[0] + p3[0]) * 0.5, (p1[1] + p3[1]) * 0.5]
        else:
            center = [(p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5]
        return center + [radius]
    cx = (c1 * v2[1] - c2 * v1[1]) / det
    cy = (v1[0] * c2 - v2[0] * c1) / det
    x = cx - p1[0]
    y = cy - p1[1]
    radius = math.sqrt(x * x + y * y) + EPS
    return [cx, cy, radius]

def find_circle(points):
    circle_candidates = {}
    total_num = len(points)
    for i in range(total_num):
        for j in range(i, total_num):
            for k in range(j, total_num):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cx, cy, r = find_circle_by_3pts(p1, p2, p3)
                if cx >= 1700 and cx <= 2100 and cy >= 900 and cy <= 1200:
                    circle_candidates[(cx, cy)] = r

    return circle_candidates

class lens_edge_detection:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._max_value = context.get("max_value")
        self._gradient_thresh = context.get("gradient_thresh")
        self._eps = context.get("eps")
        self._minpts = context.get("minpts")
        self._if_draw = context.get("if_draw")
        self._gauss_sigma = context.get("gauss_sigma")
        self._gauss_kernel = generate_gaussian_kernel(
            7, self._gauss_sigma).to(self._device)

    def run(self, x):
        yuv = rgb2yuv(x, self._max_value, self._device)
        y = yuv[:, 0:1]
        edges = edge_detection(y, self._gauss_kernel,
                               self._gradient_thresh, self._max_value, self._device)
        valid_edges = filter_valid_edges(edges, 250)
        if len(valid_edges) == 0:
            return x

        # clustering centers
        circles = find_circle(valid_edges)
        if len(circles) == 0:
            return x
        centers = np.array(list(circles.keys()))
        clustering_center = DBSCAN(eps=self._eps, min_samples=self._minpts).fit(centers)
        labels = clustering_center.labels_
        labels = np.delete(labels, np.where(labels==-1))
        if labels.shape[0] == 0:
            return x
        center_max_label = np.argmax(np.bincount(labels))
        close_centers = centers[clustering_center.labels_ == center_max_label]
        circles_with_similar_centers = {tuple(center): circles[tuple(center)] 
                                        for center in close_centers}
        # clustering radiuses
        radius = np.array(list(circles_with_similar_centers.values()))
        clustering_radius = DBSCAN(eps=20, min_samples=5).fit(radius.reshape(-1, 1))
        labels = clustering_radius.labels_
        labels = np.delete(labels, np.where(labels==-1))
        if labels.shape[0] == 0:
            return x
        radius_max_label = np.argmax(np.bincount(labels))
        close_radius = radius[clustering_radius.labels_ == radius_max_label]
        circles_with_similar_centers_and_radius = {k: v for k, v in circles_with_similar_centers.items() if v in close_radius}
        
        final_center = np.mean(np.array(list(circles_with_similar_centers_and_radius.keys())), axis=0)
        final_center = np.round(final_center)
        final_radius = np.mean(np.array(list(circles_with_similar_centers_and_radius.values())))
        final_radius = round(final_radius)

        # draw edge points and cicle
        if self._if_draw:
            dst = x.squeeze().cpu().numpy().transpose((1, 2, 0)).copy()
            # for edge in valid_edges:
            #     cv2.circle(dst, edge, radius=5, color=(0, self._max_value, 0), thickness=-1)
            # cv2.circle(dst, final_center.astype(np.uint16), radius=5, color=(self._max_value, 0, 0), thickness=-1)
            cv2.circle(dst, final_center.astype(np.uint16), radius=final_radius, color=(self._max_value, 0, 0), thickness=3)
            x = torch.tensor(dst.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
       
        return x
