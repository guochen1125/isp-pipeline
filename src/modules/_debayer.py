from .debayer import Debayer5x5, Layout
import torch


class debayer():
    def __init__(self, context) -> None:
        self._func = Debayer5x5(layout=Layout.RGGB).to(context.get('device'))
        self._max_value = context.get('max_value')

    def run(self, x):
        h, w = x.shape[2:]
        x = (x
             .reshape([2, 2, h, w])
             .permute([2, 0, 3, 1])
             .reshape([1, 1, 2*h, 2*w]))
        x = self._func(x / self._max_value) * self._max_value
        return torch.clip(x, 0, self._max_value)
