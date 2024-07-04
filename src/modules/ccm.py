import torch
from ..utils import to_string


class ccm():
    def __init__(self, context) -> None:
        self._ccm = torch.FloatTensor(
            context.get('color_correction_matrix')).to(context.get('device')).reshape([3, 3]).T
        self._max_value = context.get('max_value')

    def run(self, x):
        rgb = x.permute([0, 2, 3, 1])
        rgb = torch.clip(torch.matmul(rgb, self._ccm), 0, self._max_value)
        rgb = rgb.permute([0, 3, 1, 2])
        return rgb
