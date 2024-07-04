import torch


class median_filter:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._max_value = context.get("max_value")
        self._kernel = context.get("kernel")
        self._axis = list(context.get("axis"))

    def run(self, x):
        _, _, h, w = x.shape
        for axis in self._axis:
            axis = int(axis)
            c = x[:, axis : axis + 1]
            pad = self._kernel // 2
            c_pad = torch.nn.functional.pad(
                c,
                [pad, pad, pad, pad],
                "reflect",
            )
            patches = torch.nn.functional.unfold(
                c_pad,
                kernel_size=[self._kernel, self._kernel],
                stride=[1, 1],
            ).reshape([-1, h, w])
            c = torch.median(patches, 0).values
            x[:, axis] = c
        return torch.clip(x, 0, self._max_value)
