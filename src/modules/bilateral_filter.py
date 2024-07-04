import torch
import torch.nn.functional as F


class bilateral_filter():
    def __init__(self, context) -> None:
        self._sigmas = context.get('sigmas')
        self._sigmar = context.get('sigmar')
        self._kradius = context.get('kernel_size') // 2
        self._h = context.get('img_height')
        self._w = context.get('img_width')
        self._max_value = context.get('max_value')
        self._device = context.get('device')

        dist_space = torch.arange(-self._kradius,
                                  self._kradius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(dist_space, dist_space, indexing='ij')
        dist_square = grid_y ** 2 + grid_x ** 2
        weight_space = torch.exp(-0.5 * dist_square / self._sigmas ** 2)
        self._weight_space = torch.where(dist_square > self._kradius ** 2,
                                         torch.tensor(
                                             0, dtype=weight_space.dtype),
                                         weight_space).to(self._device)
    
    def _func(self, x):
        x_pad = F.pad(x, (self._kradius, self._kradius,
                      self._kradius, self._kradius), mode='constant', value=0)
        patches = F.unfold(x_pad, 2 * self._kradius + 1)
        patches = patches.permute(0, 2, 1).reshape(
            self._h * self._w, 2 * self._kradius + 1, 2 * self._kradius + 1)
        diff_range = torch.abs(
            patches - patches[:, self._kradius, self._kradius].unsqueeze(-1).unsqueeze(-1))
        wr = torch.exp(-0.5 * diff_range ** 2 / self._sigmar ** 2)
        ws = self._weight_space[None, ...]
        wr.mul_(ws)
        patches.mul_(wr)
        out = torch.sum(patches, (-1, -2)) / \
            torch.sum(wr, (-1, -2))
        x = out.reshape(self._h, self._w).unsqueeze(0).unsqueeze(0)
        return torch.clip(x, 0, self._max_value)

    def run(self, x):
        y = x[:, 0:1]
        x[:, 0:1] = self._func(y)
        return x


