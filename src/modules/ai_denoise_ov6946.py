import torch
from .model import pmrid, unet


class ai_denoise:
    def __init__(self, context) -> None:
        self._black_level = context.get("black_level")
        self._network_range = context.get("network_range")
        self._max_value = context.get("max_value")
        self._model_path = context.get("model_path")
        self._device = context.get("device")
        self._model = context.get("model")
        self._net = None
        if self._model == "pmrid":
            self._net = pmrid().to(self._device)
        elif self._model == "unet":
            self._net = unet().to(self._device)
        self._block_width = 128
        self._block_height = 96

        weights = torch.load(self._model_path)["net"]
        d = {}
        for key, value in weights.items():
            if key.startswith("module"):
                d[key[7:]] = value
            else:
                d[key] = value
        self._net.load_state_dict(d)

    def run(self, x):
        _, c, h, w = x.shape
        if h != 200 or w != 200:
            return x
        x = x - self._black_level
        x = x / self._max_value * self._network_range
        x_pad = torch.nn.functional.pad(x, [24, 24, 36, 36])
        patches = torch.nn.functional.unfold(
            x_pad,
            kernel_size=[self._block_height, self._block_width],
            stride=[
                self._block_height - 8,
                self._block_width - 8,
            ],
        )
        patches = patches.reshape(
            [c, self._block_height, self._block_width, -1]
        ).permute([3, 0, 1, 2])
        r = self._net(patches)
        x[..., :56, :100] = r[0, :, 36:-4, 24:-4]
        x[..., :56, 100:200] = r[1, :, 36:-4, 4:-24]

        x[..., 56:144, :100] = r[2, :, 4:-4, 24:-4]
        x[..., 56:144, 100:200] = r[3, :, 4:-4, 4:-24]

        x[..., 144:200, :100] = r[4, :, 4:-36, 24:-4]
        x[..., 144:200, 100:200] = r[5, :, 4:-36, 4:-24]

        x = x / self._network_range * self._max_value + self._black_level
        return torch.clip(x, 0, self._max_value)
