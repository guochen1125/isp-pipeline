import torch
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_distance_weights(block_h, block_w, interp_num):
    w = []
    for i in range(block_h):
        for j in range(block_w):
            for k in range(interp_num):
                for l in range(interp_num):
                    distance = (abs(l - j / block_w - (interp_num - 1) // 2)) ** 2 + (
                        abs(k - i / block_h - (interp_num - 1) // 2)
                    ) ** 2
                    w.append(distance)
    w = torch.FloatTensor(w).reshape([-1, interp_num**2, 1])
    w = w / ((interp_num / 2) ** 2 * 2)
    ps = [0, 0.5, 0.8]
    vs = [1, 0, 0]
    w1 = torch.where(w < ps[1], (vs[1] - vs[0]) / (ps[1] - ps[0]) * (w - ps[0]) + vs[0],
                     (vs[2] - vs[1]) / (ps[2] - ps[1]) * (w - ps[1]) + vs[1])
    w = torch.where(w > ps[2], -vs[2] / (1 - ps[2]) * (w - 1), w1)

    return w


def downsample(y, block_h, block_w, interp_ks):
    map = torch.nn.functional.pad(
        y,
        [
            block_w // 2,
            block_w // 2,
            block_h // 2,
            block_h // 2,
        ],
        "reflect",
    )
    map = torch.nn.functional.avg_pool2d(
        map, [block_h, block_w], [block_h, block_w])
    pad_0 = math.ceil(interp_ks / 2) - 1
    pad_1 = interp_ks // 2 - 1
    map = torch.nn.functional.pad(
        map,
        [
            pad_0,
            pad_1,
            pad_0,
            pad_1,
        ],
        "reflect",
    )
    map = torch.nn.functional.unfold(map, kernel_size=interp_ks, stride=1)
    return map


class local_contrast_enhancement:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._block_h = context.get("block_h")
        self._block_w = context.get("block_w")
        self._interp_kernel_size = context.get("interp_kernel_size")
        self._thresh_pos = context.get("thresh_pos")
        self._thresh_neg = context.get("thresh_neg")
        self._thresh = context.get("thresh")
        self._amount0 = context.get("amount0")
        self._amount1 = context.get("amount1")
        self._max_value = context.get("max_value")
        self._distance_weights = generate_distance_weights(
            self._block_h, self._block_w, self._interp_kernel_size).to(self._device)

    def run(self, x):
        torch.set_printoptions(sci_mode=False)
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()
        h, w = x.shape[-2], x.shape[-1]

        y = x[:, 0:1]
        y_blur = y
        patches = torch.nn.functional.unfold(
            y_blur,
            kernel_size=[self._block_h, self._block_w],
            stride=[self._block_h, self._block_w],
        ).unsqueeze(-2)

        y_small = downsample(y, self._block_h, self._block_w,
                             self._interp_kernel_size)
        diff = torch.abs(patches - y_small) / self._max_value
        # pr = [0, 0.32, 0.4]
        # vr = [1, 0.1, 0.01]
        pr = [0, 0.2, 0.6]
        vr = [1, 0.6, 0.1]
        weights = torch.where(diff < pr[1], (vr[1] - vr[0]) / (pr[1] - pr[0]) * (diff - pr[0]) + vr[0],
                              (vr[2] - vr[1]) / (pr[2] - pr[1]) * (diff - pr[1]) + vr[1])
        weights = torch.where(
            diff > pr[2], -vr[2] / (1 - pr[2]) * (diff - 1), weights)
        weights = weights * self._distance_weights
        weights = weights / torch.sum(weights, -2, keepdim=True)

        p = torch.sum(y_small * weights, -2)
        p = torch.nn.functional.fold(
            p,
            [h, w],
            kernel_size=[self._block_h, self._block_w],
            stride=[self._block_h, self._block_w],
        )

        interp = p[..., ::2, ::2]
        p = torch.nn.functional.interpolate(
            interp, scale_factor=2, mode='bilinear')

        detail = y_blur - p
        detail = torch.where(detail > self._thresh_pos, self._thresh_pos +
                             (detail - self._thresh_pos) / 32, detail)
        detail = torch.where(detail < self._thresh_neg, self._thresh_neg +
                             (detail - self._thresh_neg) / 32, detail)

        mask = torch.where(detail < self._thresh)
        detail[mask] = detail[mask] * self._amount1
        mask = torch.where(detail >= self._thresh)
        detail[mask] = detail[mask] * self._amount0
        x[:, 0:1] = y + detail

        # return torch.clip(x, 0, self._max_value)
        return detail
