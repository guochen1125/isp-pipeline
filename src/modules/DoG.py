import torch
import torch.nn.functional as F


class DoG():
    def __init__(self, context) -> None:
        self._sigma = context.get('sigma')
        self._kradius = context.get('gauss_kernel_size') // 2
        self._alpha = context.get('alpha')
        self._halo_amount = (100 - context.get('halo_amount')) * 0.01
        self._max_value = context.get('max_value')
        self._device = context.get('device')
        self._kernel_list = self.get_gauss_kernel()
        
        hc_weights = torch.tensor([2, 2, 2, 2, 11, 2, 2, 2, 2]).reshape([3, 3]) / 27
        self._hc_weights = hc_weights[None, None, ...].to(self._device)

    def run(self, x):
        y = x[:, 0:1]
        y_pad = F.pad(y, (self._kradius, self._kradius,
                      self._kradius, self._kradius), 'replicate')
        base = torch.clone(y)
        newy = torch.clone(y)
        for kernel, a in zip(self._kernel_list, self._alpha):
            base_ = F.conv2d(y_pad, kernel, padding='valid')
            newy += (base - base_) * a
            base = base_
            
        hc_pad = F.pad(y, (1, 1, 1, 1), 'replicate')
        npn = F.conv2d(hc_pad, self._hc_weights, padding='valid')
        max_ = F.max_pool2d(F.pad(npn, (1, 1, 1, 1), 'replicate'), 3, 1)
        min_ = -F.max_pool2d(F.pad(-npn, (1, 1, 1, 1), 'replicate'), 3, 1)
        max = torch.where(max_ < y, y, max_)
        min = torch.where(min_ > y, y, min_)
        newy = torch.where(newy > max, max + (newy - max) * self._halo_amount, newy)
        newy = torch.where(newy < min, min - (min - newy) * self._halo_amount, newy)
        x[:, 0:1] = torch.clip(newy, 0, self._max_value)

        return x

    def get_gauss_kernel(self):
        dist = torch.arange(-self._kradius, self._kradius +
                            1, dtype=torch.float32)
        kernel_list = []
        for sigma in self._sigma:
            kernel = torch.exp(-0.5 * dist ** 2 / sigma ** 2)
            kernel = kernel / torch.sum(kernel)
            kernel_list.append(
                (kernel[:, None] * kernel[None, :]).unsqueeze(0).unsqueeze(0).to(self._device))
        
        return kernel_list
