import torch
import math


def cal_gauss_coef(sigma):
    if sigma < 2.5:
        q = 3.97156 - 4.14554 * math.sqrt(1 - 0.26891 * sigma)
    else:
        q = 0.98711 * sigma - 0.96330

    b0 = 1.57825 + 2.44413 * q + 1.4281 * pow(q, 2) + 0.422205 * pow(q, 3)
    b1 = 2.44413 * q + 2.85619 * pow(q, 2) + 1.26661 * pow(q, 3)
    b2 = -1.4281 * pow(q, 2) - 1.26661 * pow(q, 3)
    b3 = 0.422205 * pow(q, 3)
    B = 1 - (b1 + b2 + b3) / b0

    b1 /= b0
    b2 /= b0
    b3 /= b0
    return [b1, b2, b3, B]


def iir_gaussian_blur(src, gauss_coef):
    b1, b2, b3, B = gauss_coef
    _, _, h, w = src.shape
    dst = torch.empty_like(src)

    dst[..., 0] = src[..., 0]
    dst[..., 1] = src[..., 0]
    dst[..., 2] = src[..., 0]
    for i in range(3, w-1):
        dst[..., i] = B * src[..., i] + b1 * dst[..., i-1] + \
            b2 * dst[..., i-2] + b3 * dst[..., i-3]

    dst[..., -1] = src[..., -1]
    dst[..., -2] = src[..., -1]
    dst[..., -3] = src[..., -1]
    for i in range(w-4, -1, -1):
        dst[..., i] = B * dst[..., i] + b1 * dst[..., i+1] + \
            b2 * dst[..., i+2] + b3 * dst[..., i+3]

    return dst


class vessel_enhancement:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._sigma = context.get("gauss_sigma")
        self.gauss_coef = cal_gauss_coef(self._sigma)
        self.rgb_coef = torch.FloatTensor(context.get(
            "rgb_coef")).reshape(3, 1, 1).to(self._device)
        self.detail_coef = context.get("detail_coef")
        self._max_value = context.get("max_value")

    def run(self, x):
        x_horizontal = iir_gaussian_blur(x, self.gauss_coef)
        x_vertical = iir_gaussian_blur(torch.transpose(
            torch.flip(x_horizontal, [2]), 2, 3), self.gauss_coef)
        blur = torch.flip(torch.transpose(x_vertical, 2, 3), [2])
        x = x * self.rgb_coef + (x - blur) * self.detail_coef

        return torch.clip(x, 0, self._max_value)
