import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class LoG:
    def __init__(self, context) -> None:
        self._kradius = context.get("kernel_size") // 2
        self._sigma = context.get("sigma")
        self._alpha = context.get("sigma")
        self._max_value = context.get("max_value")
        self._device = context.get("device")
        self._kernel_list = self.get_laplacian_kernel()
        self._pad_func = transforms.Pad(self._kradius, padding_mode="symmetric")

    def run(self, x):
        y = x[:, 0:1]
        y_pad = self._pad_func(y)
        for kernel, a in zip(self._kernel_list, self._alpha):
            detail = F.conv2d(y_pad, kernel, padding="valid")
            y += detail * a
        x[:, 0:1] = torch.clip(y, 0, self._max_value)
        return x

    def get_laplacian_kernel(self):
        dist = torch.arange(-self._kradius, self._kradius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(dist, dist, indexing="ij")

        kernel_list = []
        for sigma in self._sigma:
            kernel = (
                1
                / (torch.pi * sigma**4)
                * (1 - 0.5 * (grid_y**2 + grid_x**2) / sigma**2)
                * torch.exp(-0.5 * (grid_y**2 + grid_x**2) / sigma**2)
            )
            kernel_sum = torch.sum(kernel)
            if kernel_sum > 0:
                roi = torch.where(kernel < 0)
                offset = kernel_sum / len(roi[0]) if len(roi[0]) > 0 else 0
                kernel = torch.where(kernel < 0, kernel - offset, kernel)
            kernel_list.append(kernel.unsqueeze(0).unsqueeze(0).to(self._device))

        return kernel_list
