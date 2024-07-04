import torch


class local_color_enhancement:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._max_value = context.get("max_value")
        self._kernel = context.get("kernel")
        self._stride = context.get("stride")
        self._hue_min = context.get("hue_min")
        self._hue_max = context.get("hue_max")
        self._value_min = context.get("value_min")
        self._value_max = context.get("value_max")
        self._ds_min = context.get("ds_min")
        self._ds_max = context.get("ds_max")
        self._alpha = context.get("alpha")
        block_num = self._kernel * self._kernel
        self._smooth_min = context.get("smooth_ratio_min") * block_num
        self._smooth_max = context.get("smooth_ratio_max") * block_num

        distance = torch.arange(1, self._kernel + 1, 1).float()
        distance[self._kernel // 2 :] = torch.flip(distance[: self._kernel // 2], [0])
        weight_map = torch.unsqueeze(distance, -1) * distance
        self._weight_map = torch.unsqueeze(weight_map, -1).to(self._device)

        h, w = context.get("img_height"), context.get("img_width")
        h_block_num = (h - self._kernel) / self._stride + 1
        w_block_num = (w - self._kernel) / self._stride + 1
        block_num = int(h_block_num * w_block_num)

        weight_map_expand = weight_map.reshape([1, self._kernel**2, 1]).expand(
            1, -1, block_num
        )
        self._weight_map_denominator = torch.nn.functional.fold(
            weight_map_expand, [h, w], kernel_size=self._kernel, stride=self._stride
        ).to(self._device)

    def run(self, x):
        _, _, h, w = x.shape
        hue = x[:, 0:1, :, :]
        sat = x[:, 1:2, :, :]
        value = x[:, 2:, :, :]

        patches = torch.nn.functional.unfold(
            sat, kernel_size=self._kernel, stride=self._stride
        )

        mask = torch.where(
            ((hue < self._hue_min) | (hue > self._hue_max))
            & (value > self._value_min)
            & (value < self._value_max),
            torch.tensor(1).to(self._device).float(),
            torch.tensor(0).to(self._device).float(),
        )
        s_filtered = sat * mask
        patches_filtered = torch.nn.functional.unfold(
            s_filtered, kernel_size=self._kernel, stride=self._stride
        )
        count = torch.nn.functional.unfold(
            mask, kernel_size=self._kernel, stride=self._stride
        )
        count = torch.sum(count, -2)
        s_mean = torch.sum(patches_filtered, -2) / (count + 0.00001)
        numerator = 3 * torch.pow(s_mean, 2) - 2 * torch.pow(s_mean, 3)
        denominator = 1 - numerator

        ds = torch.clip(numerator / denominator, self._ds_min, self._ds_max)
        patches_new = 1 / (
            1 + ds.unsqueeze(-2) * torch.pow(1 / patches - 1, self._alpha)
        )

        s_weight = (count - self._smooth_min) / (self._smooth_max - self._smooth_min)
        s_weight = torch.clip(s_weight, 0, 1)

        patches_new = s_weight * patches_new + (1 - s_weight) * patches
        patches_new = patches_new.reshape([1, self._kernel, self._kernel, -1])
        patches_new = (patches_new * self._weight_map).reshape([1, self._kernel**2, -1])

        s_new = torch.nn.functional.fold(
            patches_new, [h, w], kernel_size=self._kernel, stride=self._stride
        )
        s_new = s_new / self._weight_map_denominator
        s_new = torch.clip(s_new, 0, 1)
        x[:, 1:2, :, :] = s_new
        x[:, 2:3, :, :] = value - value * (sat - s_new) / (2 - s_new)

        return torch.clip(x, 0, self._max_value)
