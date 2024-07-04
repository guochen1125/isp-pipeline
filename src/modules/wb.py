import torch


class wb:
    def __init__(self, context) -> None:
        self._wb = (
            torch.FloatTensor(context.get("white_balance"))
            .reshape([-1, 1, 1])
            .to(context.get("device"))
        )
        self._max_value = context.get("max_value")

    def run(self, x):
        return torch.clip(torch.mul(x, self._wb), 0, self._max_value)
