import torch

supported_pattern = {
    "bggr": [3, 2, 1, 0],
    "gbrg": [2, 3, 0, 1],
    "rggb": [0, 1, 2, 3],
    "grbg": [1, 0, 3, 2],
}


class demosaic:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._pattern = context.get("pattern")
        self._max_value = context.get("max_value")
        w0 = (
            torch.FloatTensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            .reshape([1, 1, 5, 5])
            .to(self._device)
        ) / 8
        w1 = (
            torch.FloatTensor(
                [
                    [0, 0, 0.5, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 0.5, 0, 0],
                ]
            )
            .reshape([1, 1, 5, 5])
            .to(self._device)
        ) / 8
        w2 = (
            torch.FloatTensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, 4, -1, 0],
                    [0.5, 0, 5, 0, 0.5],
                    [0, -1, 4, -1, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            .reshape([1, 1, 5, 5])
            .to(self._device)
        ) / 8
        w3 = (
            torch.FloatTensor(
                [
                    [0, 0, -1.5, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1.5, 0, 6, 0, -1.5],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1.5, 0, 0],
                ]
            )
            .reshape([1, 1, 5, 5])
            .to(self._device)
        ) / 8
        self._w = [
            [None, None, None],
            [None, None, None],
            [None, None, None],
            [None, None, None],
        ]
        self._w[0][1] = w0
        self._w[0][2] = w3
        self._w[1][0] = w1
        self._w[1][2] = w2
        self._w[2][0] = w2
        self._w[2][2] = w1
        self._w[3][0] = w3
        self._w[3][1] = w0
        self._width = context.get("img_width")
        self._height = context.get("img_height")
        self.y = torch.zeros([1, 3, self._height, self._width]).to(self._device)

    def run(self, x):
        _, _, h, w = x.shape
        order = supported_pattern[self._pattern.lower()]
        x = (
            x[:, order]
            .reshape([2, 2, h, w])
            .permute([2, 0, 3, 1])
            .reshape([1, 1, h * 2, w * 2])
        )
        x_padded = torch.nn.functional.pad(x, [1, 1, 1, 1], "reflect")
        for i in range(4):
            for j in range(3):
                w = self._w[order[i]][j]
                if w is None:
                    self.y[:, j, i // 2 :: 2, i % 2 :: 2] = x[
                        0, 0, i // 2 :: 2, i % 2 :: 2
                    ]
                    continue
                pad_l = (i % 2 + 1) % 2
                pad_r = 1 - pad_l
                pad_t = 1 - (i // 2)
                pad_b = 1 - pad_t
                ch = torch.nn.functional.conv2d(
                    torch.nn.functional.pad(
                        x_padded, [pad_l, pad_r, pad_t, pad_b], "reflect"
                    ),
                    w,
                    stride=2,
                )
                self.y[:, j, i // 2 :: 2, i % 2 :: 2] = ch
        return torch.clip(self.y, 0, self._max_value)
