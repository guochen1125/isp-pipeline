import torch
from .color_space_conversion import rgb2hsv, hsv2rgb


def cal_coef(hue, adjust_value, device):
    hue_name = ["rr", "rg", "gg", "gb", "bb", "br"]
    t0 = torch.empty_like(hue)
    t1 = torch.empty_like(hue)
    coef = torch.empty_like(hue)
    interval = 1 / 6
    for i in range(6):
        start = i / 6
        roi = torch.where((hue >= start) & (hue <= start + interval))
        t0[roi] = (adjust_value[hue_name[i]] - 15) / 15
        t1[roi] = (adjust_value[hue_name[(i + 1) % 6]] - 15) / 15
        coef[roi] = torch.where(
            hue[roi] - start <= interval * 3 / 4,
            (start + interval * 3 / 4 - hue[roi]) / (interval / 2),
            torch.tensor(0).float().to(device),
        )
        coef[roi] = torch.where(
            hue[roi] - start <= interval / 4,
            torch.tensor(1).float().to(device),
            coef[roi],
        )

    return [t0, t1, coef]


class hsv_adjust:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._max_value = context.get("max_value")
        self._hue_adjust_value = context.get("hue_adjust_value")
        self._sat_adjust_value = context.get("saturation_adjust_value")
        self._val_adjust_value = context.get("value_adjust_value")
        self._hue_name = ["rr", "rg", "gg", "gb", "bb", "br"]

    def run(self, x):
        # hue
        hsv = rgb2hsv(x)
        hue = hsv[:, 0:1]
        t0, t1, coef = cal_coef(hue, self._hue_adjust_value, self._device)
        new_hue = hue + 1 / 6 / 4 * (t0 * coef + t1 * (1 - coef))
        new_hue = torch.where(new_hue > 1, new_hue - 1, new_hue)
        new_hue = torch.where(new_hue < 0, 1 + new_hue, new_hue)
        hsv[:, 0:1] = new_hue

        # value
        rgb = hsv2rgb(hsv)
        t0, t1, coef = cal_coef(new_hue, self._val_adjust_value, self._device)
        param = t0 * coef + t1 * (1 - coef)
        _, _, h, w = x.shape
        max, max_index = torch.max(rgb, 1, keepdim=True)
        min, min_index = torch.min(torch.flip(rgb, [1]), 1, keepdim=True)
        min_index = 2 - min_index
        diff = torch.where(param >= 0, max - rgb, rgb - min)
        new_rgb = torch.clip(rgb + diff * param, 0, self._max_value)

        # saturation
        t0, t1, coef = cal_coef(new_hue, self._sat_adjust_value, self._device)
        mid_index = 3 - max_index - min_index
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        mid = new_rgb[0, mid_index, grid_y, grid_x]
        max = new_rgb[0, max_index, grid_y, grid_x]
        min = new_rgb[0, min_index, grid_y, grid_x]

        diff1 = torch.where(self._max_value - max - min > 0, min, self._max_value - max)
        diff2 = max - (max + min) / 2
        diff_start = torch.where(t0 >= 0, diff1, diff2)
        diff_end = torch.where(t1 >= 0, diff1, diff2)
        sat_start = diff_start * t0 * coef
        sat_end = diff_end * t1 * (1 - coef)
        max_ = torch.where(max == min, max, max + sat_start + sat_end)
        max_ = torch.clip(max_, 0, self._max_value)
        min_ = torch.where(max == min, min, min - sat_start - sat_end)
        min_ = torch.clip(min_, 0, self._max_value)
        denom = torch.where(
            max != min, max - min, torch.tensor(1).float().to(self._device)
        )
        mid_ = (max_ * (mid - min) + min_ * (max - mid)) / denom
        mid_ = torch.where(max == min, mid, mid_)
        dst = torch.empty_like(new_rgb)
        dst[0, max_index, grid_y, grid_x] = max_
        dst[0, mid_index, grid_y, grid_x] = mid_
        dst[0, min_index, grid_y, grid_x] = min_
        return dst
