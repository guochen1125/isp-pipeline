import numpy as np
import cv2
import rawpy
import torch
from .color_space_conversion import yuyv2bggr


def imread(file, shape):
    img = None
    if file.endswith("png"):
        img = cv2.imread(file, -1)
    elif file.endswith("raw") or file.endswith("bin"):
        img = np.fromfile(file, np.uint16)
    elif file.endswith("dng"):
        img = rawpy.imread(file).raw_image
    return img.reshape(shape).astype(np.float32)


def imread_uvc(cap, shape):
    yuyv = cap.get_frame().gray.flatten().astype(np.uint16)
    bggr = yuyv2bggr(yuyv)
    return bggr.reshape(shape).astype(np.float32)


supported_pattern = {
    "rggb": [0, 1, 2, 3],
    "bggr": [3, 2, 1, 0],
    "gbrg": [2, 3, 0, 1],
    "grbg": [1, 0, 3, 2],
    "rgb": [0, 1, 2],
    "rbg": [0, 2, 1],
    "gbr": [2, 0, 1],
    "grb": [1, 0, 2],
    "bgr": [2, 1, 0],
    "brg": [1, 2, 0],
}


def to_torch(np_array, context):
    device = torch.device(context.runtime_attributes.get("device"))
    h, w = np_array.shape
    order = supported_pattern[context.pattern.lower()]

    if len(order) == 3:
        np_array = np_array[..., order]
    elif len(order) == 4:
        np_array = (
            np_array.reshape([h // 2, 2, w // 2, 2])
            .transpose([0, 2, 1, 3])
            .reshape([h // 2, w // 2, 4])[..., order]
        )
    return torch.FloatTensor(np_array).unsqueeze(0).permute([0, 3, 1, 2]).to(device)


def to_numpy(torch_tensor, context):
    n, c, h, w = torch_tensor.shape
    if c == 4:
        order = supported_pattern[context.pattern.lower()]
        torch_tensor = (
            torch_tensor[:, order, :]
            .reshape([2, 2, h, w])
            .permute([2, 0, 3, 1])
            .reshape([h * 2, w * 2])
        )
    elif c == 3:
        torch_tensor = torch.flip(torch_tensor, [1]).squeeze(0).permute([1, 2, 0])

    # c, h, w = torch_tensor[1:]
    return torch_tensor.cpu().detach().numpy()
