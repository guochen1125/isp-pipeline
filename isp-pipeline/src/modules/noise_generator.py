import torch


class noise_generator:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._mean = context.get("mean")
        self._sigma = context.get("sigma")
        self._axis = context.get("channel_axis")
        self._max_value = context.get("max_value")

    def run(self, x):
        i = x[:, self._axis]
        x[:, self._axis] = (
            torch.normal(self._mean, self._sigma, i.shape).to(self._device) + i
        )
        return torch.clip(x, 0, self._max_value)
