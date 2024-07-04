import torch
import math

def generate_gaussian_kernel(kernel, sigma):
    coef = [
        1 / sigma / math.sqrt(2 * torch.pi) * math.exp(-(i**2) / 2 / sigma**2)
        for i in range(kernel // 2, -1, -1)
    ]
    coef = torch.FloatTensor(coef).reshape([1, 1, 1, -1])
    coef = torch.nn.functional.pad(coef, [0, kernel // 2, 0, 0], "reflect")
    coef = coef / torch.sum(coef)
    return (coef.reshape([-1, 1]) * coef).reshape([1, 1, kernel, kernel])