import numpy as np
import torch


def rtl_round(data):
    if type(data) == torch.Tensor:
        return torch.where(
            data - torch.floor(data) >= 0.5, torch.ceil(data), torch.floor(data)
        )
    elif type(data) == np.array:
        return np.where(data - np.floor(data) >= 0.5, np.ceil(data), np.floor(data))
    else:
        return np.ceil(data) if data - np.floor(data) >= 0.5 else np.floor(data)
