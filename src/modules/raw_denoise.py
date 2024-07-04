import torch
import math


def vst(x, a, b):
    x = a * x + 3 / 8 * a * a + b
    x = 2 / a * torch.pow(x, 0.5)
    return x


def ivst(x, a, b):
    x = (
        1 / 4 * torch.pow(x, 2)
        + 1 / 4 * torch.pow(x, -1) * math.pow(3 / 2, 0.5)
        - 11 / 8 * torch.pow(x, -2)
        + 5 / 8 * math.pow(3 / 2, 0.5) * torch.pow(x, -3)
        - 1 / 8
        - b / a**2
    ) * a
    return x


def dct_1d(x):
    a0 = x[..., 0] + x[..., 7]
    a1 = x[..., 1] + x[..., 6]
    a2 = x[..., 2] + x[..., 5]
    a3 = x[..., 3] + x[..., 4]
    a4 = x[..., 0] - x[..., 7]
    a5 = x[..., 1] - x[..., 6]
    a6 = x[..., 2] - x[..., 5]
    a7 = x[..., 3] - x[..., 4]
    a8 = a0 + a3
    a9 = a1 + a2
    a10 = a0 - a3
    a11 = a1 - a2
    a12 = 1.38703984532215 * a4 + 0.275899379282943 * a7
    a13 = 1.17587560241936 * a5 + 0.785694958387102 * a6
    a14 = -0.785694958387102 * a5 + 1.17587560241936 * a6
    a15 = 0.275899379282943 * a4 - 1.38703984532215 * a7
    a16 = 0.353553390593274 * (a12 - a13)
    a17 = 0.353553390593274 * (a14 - a15)
    y0 = 0.353553390593274 * 1.4142135623731 * (a8 + a9)
    y1 = 0.353553390593274 * (a12 + a13)
    y2 = 0.461939766255643 * a10 + 0.191341716182545 * a11
    y3 = 0.707106781186547 * (a16 - a17)
    y4 = 0.353553390593274 * (a8 - a9)
    y5 = 0.707106781186547 * (a16 + a17)
    y6 = 0.191341716182545 * a10 - 0.461939766255643 * a11
    y7 = 0.353553390593274 * (a14 + a15)
    y = torch.concat(
        [torch.unsqueeze(y, -1) for y in [y0, y1, y2, y3, y4, y5, y6, y7]], -1
    ) / math.sqrt(2)
    return y


def idct_1d(x):
    x00 = 1.4142135623731 * x[..., 0]
    x01 = 1.38703984532215 * x[..., 1] + 0.275899379282943 * x[..., 7]
    x02 = 1.30656296487638 * x[..., 2] + 0.541196100146197 * x[..., 6]
    x03 = 1.17587560241936 * x[..., 3] + 0.785694958387102 * x[..., 5]
    x04 = 1.4142135623731 * x[..., 4]
    x05 = -0.785694958387102 * x[..., 3] + 1.17587560241936 * x[..., 5]
    x06 = 0.541196100146197 * x[..., 2] - 1.30656296487638 * x[..., 6]
    x07 = -0.275899379282943 * x[..., 1] + 1.38703984532215 * x[..., 7]
    x09 = x00 + x04
    x0a = x01 + x03
    x0b = 1.4142135623731 * x02
    x0c = x00 - x04
    x0d = x01 - x03
    x0e = 0.353553390593274 * (x09 - x0b)
    x0f = 0.353553390593274 * (x0c + x0d)
    x10 = 0.353553390593274 * (x0c - x0d)
    x11 = 1.4142135623731 * x06
    x12 = x05 + x07
    x13 = x05 - x07
    x14 = 0.353553390593274 * (x11 + x12)
    x15 = 0.353553390593274 * (x11 - x12)
    x16 = 0.5 * x13

    x0 = 0.25 * (x09 + x0b) + 0.353553390593274 * x0a
    x1 = 0.707106781186547 * (x0f + x15)
    x2 = 0.707106781186547 * (x0f - x15)
    x3 = 0.707106781186547 * (x0e + x16)
    x4 = 0.707106781186547 * (x0e - x16)
    x5 = 0.707106781186547 * (x10 - x14)
    x6 = 0.707106781186547 * (x10 + x14)
    x7 = 0.25 * (x09 + x0b) - 0.353553390593274 * x0a

    x = torch.concat(
        [torch.unsqueeze(x, -1) for x in [x0, x1, x2, x3, x4, x5, x6, x7]], -1
    )
    return x


def dct_2d(x):
    x = x.permute([0, 1, 3, 2])
    x = dct_1d(x)
    x = x.permute([0, 1, 3, 2])
    x = dct_1d(x)
    return x


def idct_2d(x):
    x = x.permute([0, 1, 3, 2])
    x = idct_1d(x)
    x = x.permute([0, 1, 3, 2])
    x = idct_1d(x)
    return x


class raw_denoise:
    def __init__(self, context):
        self._device = context.get("device")
        ak = torch.FloatTensor(context.get("ak")).reshape([4, 1, 1]).to(self._device)
        bk = torch.FloatTensor(context.get("bk")).reshape([4, 1, 1]).to(self._device)
        bb = torch.FloatTensor(context.get("bb")).reshape([4, 1, 1]).to(self._device)
        self._gain = context.get("gain")
        self._a = ak * self._gain
        self._b = bk * (self._gain) ** 2 + bb
        self._stride = context.get("stride")
        self._thresh = (
            torch.FloatTensor(context.get("thresh"))
            .reshape([4, 1, 1, 1])
            .to(self._device)
        )
        self._max_value = context.get("max_value")
        self._ZERO = torch.tensor([0.0]).to(self._device)


    def run(self, x):
        _, _, h, w = x.shape
        x = vst(x, self._a, self._b)
        x_padded = torch.nn.functional.pad(x, [7, 7, 7, 7], "reflect")
        patches = (
            torch.nn.functional.unfold(x_padded, kernel_size=8, stride=self._stride)
            .reshape([4, 8, 8, -1])
            .permute([0, 3, 1, 2])
        )
        dct = dct_2d(patches)
        dct = torch.where(torch.abs(dct) < self._thresh, self._ZERO, dct)
        idct = idct_2d(dct)
        idct = idct.permute([0, 2, 3, 1]).reshape([4, 64, -1])
        idct = torch.nn.functional.fold(
            idct, [h + 14, w + 14], kernel_size=8, stride=self._stride
        )[..., 7:-7, 7:-7]
        idct = idct.squeeze().unsqueeze(0)
        idct /= 64 / self._stride / self._stride
        x = ivst(idct, self._a, self._b)
        return torch.clip(x, 0, self._max_value)
