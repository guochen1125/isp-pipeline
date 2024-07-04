import torch


def rgb2yuv(x, max_value, device):
    x = x.permute([0, 2, 3, 1]) / max_value
    matrix = (
        torch.FloatTensor(
            [0.299, 0.587, 0.114, -0.147, -0.289, 0.436, 0.615, -0.515, -0.1]
        )
        .reshape([3, 3])
        .T.to(device)
    )
    delta = torch.FloatTensor([0, 0.5, 0.5]).to(device)
    x = torch.matmul(x, matrix) + delta
    return x.permute([0, 3, 1, 2]) * max_value


def yuv2rgb(x, max_value, device):
    x = x.permute([0, 2, 3, 1]) / max_value
    matrix = (
        torch.FloatTensor([1, 0, 1.1398, 1, -0.395, -0.58, 1, 2.032, 0])
        .reshape([3, 3])
        .T.to(device)
    )
    delta = torch.FloatTensor([0, 0.5, 0.5]).to(device)
    x = torch.matmul(x - delta, matrix)
    return x.permute([0, 3, 1, 2]) * max_value


def rgb2hsv(rgb):
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb(hsv):
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (-torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


class color_space_conversion:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._func = None
        self._type = context.get("type")
        self._matrix = None
        self._delta = torch.FloatTensor([0, 0.5, 0.5]).to(self._device)
        if self._type == "rgb2yuv":
            self._func = rgb2yuv
        elif self._type == "yuv2rgb":
            self._func = yuv2rgb
        elif self._type == "rgb2hsv":
            self._func = rgb2hsv
        elif self._type == "hsv2rgb":
            self._func = hsv2rgb
        self._max_value = context.get("max_value")

    def run(self, x):
        if self._type == "rgb2yuv":
            x = self._func(x, self._max_value, self._device)
        elif self._type == "yuv2rgb":
            x = self._func(x, self._max_value, self._device)
        elif self._type == "rgb2hsv":
            x = self._func(x)
        elif self._type == "hsv2rgb":
            x = self._func(x)
        return torch.clip(x, 0, self._max_value)
