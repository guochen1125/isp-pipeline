import torch
from ..utils import generate_gaussian_kernel


def gaussblur_1ch(x, filter, ksize):
    x = torch.nn.functional.pad(
        x,
        [
            ksize // 2,
            ksize // 2,
            ksize // 2,
            ksize // 2,
        ],
        "reflect",
    )
    return torch.nn.functional.conv2d(x, filter, stride=1)


def denoise_1ch(x, filter, ksize, th_24):
    _, _, h, w = x.shape
    lp = gaussblur_1ch(x, filter, ksize)
    x_pad = torch.nn.functional.pad(
        x,
        [2, 2, 2, 2],
        "reflect",
    )
    patches = torch.nn.functional.unfold(
        x_pad,
        kernel_size=[5, 5],
        stride=[1, 1],
    ).reshape([25, h, w])
    diff = torch.abs(x - lp)
    diff_seperate = torch.abs(patches - lp)
    diff_sum = torch.sum(diff_seperate, -3).reshape(diff.shape)
    mask = torch.where(diff > diff_sum * th_24)

    weights = 1 / torch.sqrt(diff_seperate + 0.00001)
    numerator = torch.sum(patches * weights, -3)
    denominator = torch.sum(weights, -3)
    x[mask] = (numerator / denominator).reshape(x.shape)[mask]
    return x


class chromatic_noise_reduction:
    def __init__(self, context) -> None:
        self._device = context.get("device")
        self._thresh = context.get("thresh")
        self._max_value = context.get("max_value")
        self._gauss_ksize = context.get("gauss_ksize")
        self._gauss_sigma = context.get("gauss_sigma")
        self._gauss = generate_gaussian_kernel(self._gauss_ksize, self._gauss_sigma).to(
            self._device
        )

    def run(self, x):
        u = x[:, 1:2]
        v = x[:, 2:]
        impthr = max(1, 5.5 - self._thresh)
        impthrDiv24 = impthr / 24.0
        x[:, 1:2] = denoise_1ch(u, self._gauss, self._gauss_ksize, impthrDiv24)
        x[:, 2:] = denoise_1ch(v, self._gauss, self._gauss_ksize, impthrDiv24)
        return torch.clip(x, 0, self._max_value)
