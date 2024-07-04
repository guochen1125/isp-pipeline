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
        self._pad = 4

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
        if h != 360 or w != 360:
            return x
        x = x - self._black_level
        x = x / self._max_value * self._network_range
        x_pad = torch.nn.functional.pad(x, [self._pad, self._pad])
        patches = torch.nn.functional.unfold(
            x_pad,
            kernel_size=[self._block_height, self._block_width],
            stride=[
                self._block_height - self._pad * 2,
                self._block_width - self._pad * 2,
            ],
        )
        patches = patches.reshape(
            [c, self._block_height, self._block_width, -1]
        ).permute([3, 0, 1, 2])
        r = self._net(patches)
        x[..., :92, :120] = r[0, :, : -self._pad, self._pad : -self._pad]
        x[..., :92, 120:240] = r[1, :, : -self._pad, self._pad : -self._pad]
        x[..., :92, 240:360] = r[2, :, : -self._pad, self._pad : -self._pad]

        x[..., 92:180, :120] = r[3, :, self._pad : -self._pad, self._pad : -self._pad]
        x[..., 92:180, 120:240] = r[
            4, :, self._pad : -self._pad, self._pad : -self._pad
        ]
        x[..., 92:180, 240:360] = r[
            5, :, self._pad : -self._pad, self._pad : -self._pad
        ]

        x[..., 180:268, :120] = r[6, :, self._pad : -self._pad, self._pad : -self._pad]
        x[..., 180:268, 120:240] = r[
            7, :, self._pad : -self._pad, self._pad : -self._pad
        ]
        x[..., 180:268, 240:360] = r[
            8, :, self._pad : -self._pad, self._pad : -self._pad
        ]

        x[..., 268:360, :120] = r[9, :, self._pad :, self._pad : -self._pad]
        x[..., 268:360, 120:240] = r[10, :, self._pad :, self._pad : -self._pad]
        x[..., 268:360, 240:360] = r[11, :, self._pad :, self._pad : -self._pad]

        x = x / self._network_range * self._max_value + self._black_level
        return torch.clip(x, 0, self._max_value)
