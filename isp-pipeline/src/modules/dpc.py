import torch


class dpc:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._max_value = context.get("max_value")
        self._param0 = context.get("dark_pixel_median_thresh")
        self._param1 = context.get("dark_pixel_median_ratio")
        self._param2 = context.get("white_pixel_subsmallest_thresh")
        pass

    def run(self, x) -> torch.Tensor:
        _, c, h, w = x.shape
        x_pad = torch.nn.functional.pad(x, [1, 1, 1, 1], mode="replicate")
        patches = torch.nn.functional.unfold(x_pad, [3, 3]).reshape([-1, c, 9, h, w])
        sort = torch.topk(patches, 9, 2).values

        dif = sort[..., 1, :, :] - sort[..., -2, :, :]
        avg = (torch.sum(patches, 2) - sort[..., 0, :, :] - sort[..., -1, :, :] - x) / 6

        dark_dead_pixel = torch.where(
            (x == sort[..., -1, :, :])
            & (x < avg - dif)
            & (sort[..., 4, :, :] > sort[..., -1, :, :] + self._param0)
            & (x < sort[..., 4, :, :] * self._param1)
        )

        white_dead_pixel = torch.where(
            (x == sort[..., 0, :, :])
            & (x > avg + dif)
            & (x > sort[..., 1, :, :] + dif)
            & (x > sort[..., 1, :, :] + self._param2)
        )

        x[dark_dead_pixel] = sort[..., 4, :, :][dark_dead_pixel]
        x[white_dead_pixel] = sort[..., 4, :, :][white_dead_pixel]

        return x
