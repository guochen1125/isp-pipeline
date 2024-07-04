import torch
from ..utils import rtl_round

supported_pattern = {
    "bggr": [3, 2, 1, 0],
    "gbrg": [2, 3, 0, 1],
    "rggb": [0, 1, 2, 3],
    "grbg": [1, 0, 3, 2],
}


class vnr:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._height = context.get("img_height")
        self._width = context.get("img_width")
        self._pattern = context.get("pattern")
        self._smooth_thresh = context.get("smooth_thresh")
        self._white_level = context.get("white_level")
        self._max_value = context.get("max_value")
        self._stripe = torch.FloatTensor([0, 0, 0, 0]).to(self._device)

    def _calibrate(self, x):
        bayer0 = x
        bayer1 = torch.roll(x, -1, -1)
        bayer2 = torch.roll(x, -2, -1)
        bayer3 = torch.roll(x, -3, -1)
        bayer4 = torch.roll(x, -4, -1)
        bayer5 = torch.roll(x, -5, -1)
        shift = bayer0 + bayer1 - bayer2 - bayer3
        w1 = torch.where(
            torch.abs(bayer0 - bayer4) + torch.abs(bayer1 - bayer5)
            < self._smooth_thresh,
            1,
            0,
        )
        w2 = torch.where(bayer0 < self._white_level, 1, 0)
        w = w1 * w2

        shift = shift[:, : self._width - 6]
        w = w[:, : self._width - 6]

        mul = shift * w
        s0 = torch.sum(mul[:, ::4] / torch.sum(w[:, ::4])) / 4
        s1 = torch.sum(mul[:, 1::4] / torch.sum(w[:, 1::4])) / 4
        s2 = torch.sum(mul[:, 2::4] / torch.sum(w[:, 2::4])) / 4
        s3 = torch.sum(mul[:, 3::4] / torch.sum(w[:, 3::4])) / 4
        s = torch.FloatTensor([s0, s1, s2, s3]).to(self._device)
        diff = rtl_round(torch.max(s) - torch.min(s))
        c2 = torch.div(diff, 2, rounding_mode='trunc')
        c = [-c2, -c2, diff - c2, diff - c2]
        for i in range(4):
            index = i + torch.argmax(s)
            if index > 3:
                index -= 4
            self._stripe[index] = float(c[i])

    def run(self, x):
        _, _, h, w = x.shape
        order = supported_pattern[self._pattern.lower()]
        x = (
            x[:, order]
            .reshape([2, 2, h, w])
            .permute([2, 0, 3, 1])
            .reshape([h * 2, w * 2])
        )
        self._calibrate(x)
        x = x.reshape([-1, 4]) + self._stripe
        x = (
            x.reshape([h, 2, w, 2])
            .permute([1, 3, 0, 2])
            .reshape([1, 4, h, w])[:, order]
        )
        return torch.clip(x, 0, self._max_value)
