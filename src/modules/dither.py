import cv2
import numpy as np
import os

# module function: LSB 2 bit dithering
# input image type: 3color RGB 3channels, or YUV 3channels
import torch


class Dither():
    def __init__(self, context) -> None:
        self._device = context.get('device')
        self._max_value = context.get('max_value') // 4
        self._dither_space = context.get('dither_space')

        self._h, self._w = context.get('img_height'), context.get('img_width')
        
        tz = context.get('tamplate_z')
        t0, t1, t2, t3 = context.get('tamplate_0'), context.get('tamplate_1'), context.get('tamplate_2'), context.get('tamplate_3') 
        t4, t5 = context.get('tamplate_4'), context.get('tamplate_5')
        t6, t7, t8, t9 = context.get('tamplate_6'), context.get('tamplate_7'), context.get('tamplate_8'), context.get('tamplate_9')

        self._table = torch.FloatTensor(
                [
                    [tz, t0, t4, t6],
                    [tz, t1, t5, t7],
                    [tz, t2, t4, t8],
                    [tz, t3, t5, t9],
                    [tz, t1, t4, t7],
                    [tz, t0, t5, t6],
                    [tz, t3, t4, t9],
                    [tz, t2, t5, t8]
                ]
            ).reshape([8, 4, 4, 4]).to(self._device)

    # _func_rgb: rgb 3ch dither separate
    def _func_rgb(self, img, fcnt):
        for i in range(0, 3):
            fcnt_tensor = torch.tensor([fcnt % 8] * self._h * self._w).view(1, 1, self._h, self._w).to(self._device) 
            pixel_value_index = img[:, i, :, :].squeeze(0).squeeze(0) % 4
            h_index = torch.tensor([ih % 4 for ih in range(self._h)]).view(1, self._h, 1).expand(self._w, self._h, self._w).permute(1, 2, 0)[:,:,0]
            w_index = torch.tensor([iw % 4 for iw in range(self._w)]).view(1, 1, self._w).expand(self._h, self._w, self._w)[:,0,:]
            
            target_matrix = self._table[fcnt_tensor.squeeze().long(), pixel_value_index.long(), h_index.long(), w_index.long()]  
            target_matrix = target_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]  

            img[:, i, :, :] = img[:, i, :, :] // 4 + target_matrix[:, 0, :, :]
        
        return torch.clip(img, 0, self._max_value)

    # _func_yuv: yuv 3ch dither separate
    def _func_yuv(self, img, fcnt):
        for i in range(0, 3):
            fcnt_tensor = torch.tensor([fcnt % 8] * self._h * self._w).view(1, 1, self._h, self._w).to(self._device) 
            pixel_value_index = img[:, i, :, :].squeeze(0).squeeze(0) % 4
            h_index = torch.tensor([ih % 4 for ih in range(self._h)]).view(1, self._h, 1).expand(self._w, self._h, self._w).permute(1, 2, 0)[:,:,0]
            w_index = torch.tensor([iw % 4 for iw in range(self._w)]).view(1, 1, self._w).expand(self._h, self._w, self._w)[:,0,:]
            
            target_matrix = self._table[fcnt_tensor.squeeze().long(), pixel_value_index.long(), h_index.long(), w_index.long()]  
            target_matrix = target_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]  

            img[:, i, :, :] = img[:, i, :, :] // 4 + target_matrix[:, 0, :, :]

        return torch.clip(img, 0, self._max_value)

    def run(self, img, fcnt=0):
        if self._dither_space == 'rgb':
            img = self._func_rgb(img, fcnt) * 4

        if self._dither_space == 'yuv':
            img = self._func_yuv(img, fcnt) * 4
        
        return torch.clip(img, 0, self._max_value * 4)


def save_img(input, filename="", path=""):
        img8bit = input.astype(np.uint8)

        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        filename = filename + ".png"
        output_path = os.path.join(path, filename)
        cv2.imwrite(output_path, img8bit)

def rgbyuv(x, m):
    return torch.matmul(x, m)

def rgb2yuv(x, matrix, max_value, delta):
    x = x.permute([0, 2, 3, 1]) / max_value
    x = rgbyuv(x, matrix) + delta
    x = x.permute([0, 3, 1, 2]) * max_value

    return torch.clip(x, 0, max_value)

def yuv2rgb(x, matrix, max_value, delta):
    x = x.permute([0, 2, 3, 1]) / max_value
    x = rgbyuv(x - delta, matrix)
    x = x.permute([0, 3, 1, 2]) * max_value

    return torch.clip(x, 0, max_value)


# apply to varify module by one .py file
if __name__ == '__main__':
    context = { 
                "max_value": 1023,
                "device": 'cuda',
                "img_width": 1920,
                "img_height": 1080,
                "dither_space": 'yuv',  # 'rgb' or 'yuv'

                "tamplate_z": [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                "tamplate_0": [[0,0,0,1],[1,0,0,0],[0,0,0,1],[1,0,0,0]],
                "tamplate_1": [[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0]],
                "tamplate_2": [[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,1]],
                "tamplate_3": [[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]],
                "tamplate_4": [[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]],
                "tamplate_5": [[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]],
                "tamplate_6": [[1,1,1,0],[0,1,1,1],[1,1,1,0],[0,1,1,1]],
                "tamplate_7": [[1,0,1,1],[1,1,0,1],[1,0,1,1],[1,1,0,1]],
                "tamplate_8": [[0,1,1,1],[1,1,1,0],[0,1,1,1],[1,1,1,0]],
                "tamplate_9": [[1,1,0,1],[1,0,1,1],[1,1,0,1],[1,0,1,1]],
            }
    dither = Dither(context)

    h, w, device = context.get('img_height'), context.get('img_width'), context.get('device')
    gradient = torch.linspace(0, 1, w).repeat(h, 1).repeat(3, 1, 1).unsqueeze(0).to(device) 

    rgb2yuv_matrix = (
            torch.FloatTensor(
                [0.299, 0.587, 0.114, -0.147, -0.289, 0.436, 0.615, -0.515, -0.1]
            )
            .reshape([3, 3])
            .T.to(device)
        )
    yuv2rgb_matrix = (
            torch.FloatTensor([1, 0, 1.1398, 1, -0.395, -0.58, 1, 2.032, 0])
            .reshape([3, 3])
            .T.to(device)
        )
    
    delta = torch.FloatTensor([0, 0.5, 0.5]).to(device)
    
    for cnt in range(1):
        img = gradient * 1023

        # img = rgb2yuv(img, rgb2yuv_matrix, 1023, delta)
        img = dither.run(img, cnt) // 4  # input 0-1023, output 0-255
        # img = yuv2rgb(img, yuv2rgb_matrix, 255, delta)  # max = 255 after dither

        img_out = (torch.flip(img, [1]).squeeze(0).permute([1, 2, 0])).cpu().detach().numpy()
        # img_out = (img_out.astype(np.uint16) / 1023 * 255).astype(np.uint8)  # origin 0-1023 to 0-255
    
        # save_img(img_out, "dither_out_" + str(cnt), "./sim_output")
        
        # img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        # img_out = img_out.astype(np.uint8)
        # cv2.namedWindow('dither test', cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('dither test', img_out)
        # cv2.waitKey(1)
        
    # cv2.destroyAllWindows()
