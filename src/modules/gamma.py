import torch


def _rec709(data, max_value):
    data = data / max_value
    data = torch.where(data < 0.018, data * 4.5, 1.099 * data ** 0.45 - 0.099)
    return data * max_value


_supported_func = {'rec709': _rec709}


class gamma():
    def __init__(self, context) -> None:
        self._func = _supported_func[context.get('type')]
        self._max_value = context.get('max_value')

    def run(self, x):
        return torch.clip(self._func(x, self._max_value), 0, self._max_value)
