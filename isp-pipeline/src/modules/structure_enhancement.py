import torch
from .DoG import DoG
from .local_contrast_enhancement import local_contrast_enhancement

class structure_enhancement:
    def __init__(self, context) -> None:
        self._max_value = context.get("max_value")
        self._run_lce = context.get("run_lce")
        self._run_DoG = context.get("run_DoG")
        self._dog = DoG(context)
        self._lce = local_contrast_enhancement(context)
    
    def run(self, x):
        x_clone = x.clone()
        if self._run_DoG:
            yuv = DoG.run(self._dog, x_clone)
        else:
            yuv = x_clone
        if self._run_lce:
            detail = local_contrast_enhancement.run(self._lce, x)
        else:
            detail = torch.zeros_like(x[:, 0:1])

        yuv[:, 0:1] += detail
        
        return torch.clip(yuv, 0, self._max_value)
