import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import time
import torch
import os
import logging


def rtl_round(tensor, decimal_bit_width=0, torch_format=False):
    # tensor /= 2.0**decimal_bit_width
    tensor = tensor / 2.0**decimal_bit_width
    if torch_format:
        return torch.where(
            tensor - torch.floor(tensor) == 0.5, torch.ceil(tensor), torch.round(tensor)
        )
    else:
        return np.where(
            tensor - np.floor(tensor) == 0.5, np.ceil(tensor), np.round(tensor)
        )


def range_limit(value, sign_bw, integer_bw, decimal_bw, torch_format=False):
    value_max = 2 ** (integer_bw + decimal_bw) - 1
    value_min = 0 if sign_bw == 0 else (-1 * value_max - 1)
    if torch_format:
        data_range = torch.where(value > value_max, 1, 0) + torch.where(
            value < value_min, 1, 0
        )
        if torch.sum(data_range) > 0:
            print(torch.where(value > value_max))
            print(value[torch.where(value > value_max)])
            print(torch.where(value < value_min))
            print(value[torch.where(value < value_min)])
            assert False
    else:
        data_range = np.where(value > value_max, 1, 0) + np.where(
            value < value_min, 1, 0
        )
        if np.sum(data_range) > 0:
            print(np.where(value > value_max))
            print(value[np.where(value > value_max)])
            print(np.where(value < value_min))
            print(value[np.where(value < value_min)])
            assert False


def save_string_hex_unsigned(file_name, data, bit_width):
    """save the data to txt using hex foamat, which is limited bit width

    Args:
        file_name (string): txt file name
        data (int): data
        bit_width (int): limited data
    """
    print("Output: ", file_name)
    format = "%0" + str(math.ceil(bit_width / 4)) + "x"
    line = [format % int(i) for i in data.flatten()]
    f = open(file_name, "w")
    f.write("\n".join(line))


class lsc:
    # img size
    img_height = 0
    img_width = 0

    # preset paras for specified img size
    img_size = [[0, 0], [2160, 3840], [1080, 1920], [720, 720], [400, 400], [720, 1280]]
    img_padded_size = [
        [0, 0],
        [2176, 3840],
        [1088, 1920],
        [736, 736],
        [416, 416],
        [736, 1280],
    ]
    local_size = [[0, 0], [64, 64], [32, 32], [32, 32], [32, 32], [32, 32]]
    local_num = [[0, 0], [34, 60], [34, 60], [23, 23], [13, 13], [23, 40]]

    # init to decide for using which group of the preset paras above
    size_symbol = -1

    # refer to the refchannel for calcuating luma shading gain
    ref_channel = 0
    luma_shading_ratio = 1.0

    def __init__(self, context) -> None:
        """lsc init function"""
        self.device = context.get("device")
        self.luma_shading_ratio = context.get("luma_shading_ratio")
        # self.luma_shading_ratio = 1.0
        self.ref_channel = 1
        self.img_height = context.get("img_height")
        self.img_width = context.get("img_width")
        gain_path = context.get("gain_path")
        self.img_bit_width = 10
        self.gain_pattern = context.get("gain_pattern")
        if (self.img_height == self.img_size[1][0]) and self.img_width == self.img_size[
            1
        ][1]:
            self.size_symbol = 1
        elif (
            self.img_height == self.img_size[2][0]
        ) and self.img_width == self.img_size[2][1]:
            self.size_symbol = 2
        elif (
            self.img_height == self.img_size[3][0]
        ) and self.img_width == self.img_size[3][1]:
            self.size_symbol = 3
        elif (
            self.img_height == self.img_size[4][0]
        ) and self.img_width == self.img_size[4][1]:
            self.size_symbol = 4
        elif (
            self.img_height == self.img_size[5][0]
        ) and self.img_width == self.img_size[5][1]:
            self.size_symbol = 5
        else:
            self.img_size[0][0] = self.img_height
            self.img_size[0][1] = self.img_width
            if self.img_height > 1080 or self.img_width > 1920:
                self.local_size[0][0] = 64
                self.local_size[0][1] = 64
            else:
                self.local_size[0][0] = 32
                self.local_size[0][1] = 32
            self.local_num[0][0] = (
                (self.img_height // self.local_size[0][0])
                if (self.img_height % self.local_size[0][0] == 0)
                else ((self.img_height // self.local_size[0][0]) + 1)
            )
            self.local_num[0][1] = (
                (self.img_width // self.local_size[0][1])
                if (self.img_width % self.local_size[0][1] == 0)
                else ((self.img_width // self.local_size[0][1]) + 1)
            )
            self.img_padded_size[0][0] = int(
                self.local_num[0][0] * self.local_size[0][0]
            )
            self.img_padded_size[0][1] = int(
                self.local_num[0][1] * self.local_size[0][1]
            )
            self.size_symbol = 0
            logging.warning("[lsc]LSC init image size mismatch.")

        if gain_path.endswith(".txt") or gain_path.endswith(".bin"):
            if os.path.exists(gain_path):
                self.__get_gain_weight_map(gain_path)
            else:
                logging.error("LSC gain file does not exist")
                exit()

    def __get_gain_weight_map(self, gain_txt_path):
        # Parameter initialization
        img_local_height = self.local_size[self.size_symbol][0]
        img_local_width = self.local_size[self.size_symbol][1]
        local_num_height = self.local_num[self.size_symbol][0]
        local_num_width = self.local_num[self.size_symbol][1]

        # get gain map
        if gain_txt_path[-4:] == ".txt":
            self.local_vertex_gain_quantify = self.__get_gain_from_txt(
                gain_txt_path, local_num_height + 1, local_num_width + 1
            )
        elif gain_txt_path[-4:] == ".bin":
            self.local_vertex_gain_quantify = self.__get_gain_from_bin(
                gain_txt_path,
                local_num_height + 1,
                local_num_width + 1,
                self.size_symbol,
            )
        else:
            assert False, "gain format or path is not suitable"

        self.local_vertex_gain_quantify = self.local_vertex_gain_quantify.reshape(
            ((local_num_height + 1), (local_num_width + 1), 4)
        )

        self.local_vertex_gain_quantify = rtl_round(
            self.local_vertex_gain_quantify, 0, True
        )
        # print(self.local_vertex_gain_quantify.shape)
        # print(self.local_vertex_gain_quantify.dtype)
        # print(torch.max(self.local_vertex_gain_quantify))
        # exit()

        luma_shading = self.local_vertex_gain_quantify[
            :, :, self.ref_channel
        ].unsqueeze(-1)
        color_shading = self.local_vertex_gain_quantify / luma_shading
        new_luma_shading = (luma_shading - 64) * self.luma_shading_ratio + 64
        self.local_vertex_gain_quantify = color_shading * new_luma_shading

        weight_h_upper = torch.arange(1, img_local_height + 1, 1).to(self.device)
        weight_h_lower = img_local_height - weight_h_upper
        weight_w_upper = torch.arange(0, img_local_width, 1).to(self.device)
        weight_w_lower = img_local_width - weight_w_upper

        weight_map_00 = weight_h_lower.unsqueeze(1) * weight_w_lower.unsqueeze(0)
        weight_map_01 = weight_h_lower.unsqueeze(1) * weight_w_upper.unsqueeze(0)
        weight_map_10 = weight_h_upper.unsqueeze(1) * weight_w_lower.unsqueeze(0)
        weight_map_11 = weight_h_upper.unsqueeze(1) * weight_w_upper.unsqueeze(0)

        self.local_vertex_gain_quantify = self.local_vertex_gain_quantify.permute(
            [2, 0, 1]
        ).unsqueeze(0)

        gain_00 = self.local_vertex_gain_quantify
        gain_01 = torch.roll(gain_00.clone(), shifts=(0, -1), dims=(2, 3))[
            :, :, :-1, :-1
        ]
        gain_10 = torch.roll(gain_00.clone(), shifts=(-1, 0), dims=(2, 3))[
            :, :, :-1, :-1
        ]
        gain_11 = torch.roll(gain_00.clone(), shifts=(-1, -1), dims=(2, 3))[
            :, :, :-1, :-1
        ]
        gain_00 = gain_00[:, :, :-1, :-1]
        gain_00 = gain_00.unsqueeze(-1).unsqueeze(-1)
        gain_01 = gain_01.unsqueeze(-1).unsqueeze(-1)
        gain_10 = gain_10.unsqueeze(-1).unsqueeze(-1)
        gain_11 = gain_11.unsqueeze(-1).unsqueeze(-1)

        gain_weight_map = (
            gain_00 * weight_map_00
            + gain_01 * weight_map_01
            + gain_10 * weight_map_10
            + gain_11 * weight_map_11
        )

        if self.gain_pattern == "RGGB":
            pass
        elif self.gain_pattern == "GRBG":
            gain_weight_map = gain_weight_map[:, [1, 0, 3, 2]]
        elif self.gain_pattern == "BGGR":
            gain_weight_map = gain_weight_map[:, [3, 2, 1, 0]]
        elif self.gain_pattern == "GBRG":
            gain_weight_map = gain_weight_map[:, [2, 3, 0, 1]]
        else:
            logging.error("LSC gain_pattern wrong.")

        # 1 4 23 23 32 32
        self.gain_weight_map_final = torch.zeros(gain_weight_map.shape[2:]).to(
            self.device
        )

        self.gain_weight_map_final[:, :, 0::2, 0::2] = gain_weight_map[
            0, 0, :, :, 0::2, 0::2
        ]
        self.gain_weight_map_final[:, :, 0::2, 1::2] = gain_weight_map[
            0, 1, :, :, 0::2, 1::2
        ]
        self.gain_weight_map_final[:, :, 1::2, 0::2] = gain_weight_map[
            0, 2, :, :, 1::2, 0::2
        ]
        self.gain_weight_map_final[:, :, 1::2, 1::2] = gain_weight_map[
            0, 3, :, :, 1::2, 1::2
        ]

    def __img_border_replicate(self, img):
        width = 5
        img[:, :, :width, :] = img[:, :, width : width * 2, :]
        img[:, :, :, -width:] = img[:, :, :, -width * 2 : -width]
        img[:, :, -width:, :] = img[:, :, -width * 2 : -width, :]
        img[:, :, :, :width] = img[:, :, :, width : width * 2]
        return img

    def __fit_cos4th(self, g0, g1, g2, L0, L1, L2, Lx):
        """fit LSC gain using cos4th law

        Args:
            g0 (np.float): LSC gain in P0
            g1 (np.float): LSC gain in P1
            g2 (np.float): LSC gain in P2
            L0 (np.int): distance sq between P0 and img center
            L1 (np.int): distance sq between P1 and img center
            L2 (np.int): distance sq between P2 and img center
            Lx (np.int): distance sq between PX and img center

        Returns:
            np.float: LSC gain in PX
        """
        assert L0 != 0 and L1 != 0 and L2 != 0
        g0x = (1 + (Lx / L0) * (g0**0.5 - 1)) ** 2
        g1x = (1 + (Lx / L1) * (g1**0.5 - 1)) ** 2
        g2x = (1 + (Lx / L2) * (g2**0.5 - 1)) ** 2

        if L0 == Lx:
            return g0x
        elif L1 == Lx:
            return g1x
        elif L2 == Lx:
            return g2x
        else:
            w0 = 1 / abs(L0 - Lx)
            w1 = 1 / abs(L1 - Lx)
            w2 = 1 / abs(L2 - Lx)
            gx = (w0 * g0x + w1 * g1x + w2 * g2x) / (w0 + w1 + w2)
            return gx

    def __fit_gain(self, local_gain, plt_show):
        local_gain_expand = torch.zeros(
            (local_gain.shape[0] + 2, local_gain.shape[1] + 2, local_gain.shape[2])
        ).to(self.device)
        local_gain_ori = local_gain.clone()
        coor_yx = torch.tensor(
            [
                [i, j]
                for i in range(0, local_gain.shape[0] + 2)
                for j in range(0, local_gain.shape[1] + 2)
            ]
        ).to(self.device)
        coor_yx_local = torch.tensor(
            [
                [i + 1, j + 1]
                for i in range(0, local_gain.shape[0])
                for j in range(0, local_gain.shape[1])
            ]
        ).to(self.device)
        L_yx = [
            (
                (i + 0.5 - (local_gain.shape[0] + 2) / 2.0) ** 2
                + (j + 0.5 - (local_gain.shape[1] + 2) / 2.0) ** 2
            )
            for i in range(0, local_gain.shape[0] + 2)
            for j in range(0, local_gain.shape[1] + 2)
        ]
        L_yx = (
            torch.tensor(L_yx)
            .reshape((local_gain.shape[0] + 2, local_gain.shape[1] + 2))
            .to(self.device)
        )
        for k in range(0, local_gain.shape[2]):
            for i in range(0, local_gain.shape[0]):
                for j in range(0, local_gain.shape[1]):
                    local_gain_expand[i + 1, j + 1, k] = local_gain[i, j, k]
                    if i == 0:
                        g0 = local_gain[i + 2, j, k]
                        g1 = local_gain[i + 1, j, k]
                        g2 = local_gain[i, j, k]
                        L0 = L_yx[i + 3, j + 1]
                        L1 = L_yx[i + 2, j + 1]
                        L2 = L_yx[i + 1, j + 1]
                        Lx = L_yx[i, j + 1]
                        local_gain_expand[i, j + 1, k] = self.__fit_cos4th(
                            g0, g1, g2, L0, L1, L2, Lx
                        )
                    if j == 0:
                        g0 = local_gain[i, j + 2, k]
                        g1 = local_gain[i, j + 1, k]
                        g2 = local_gain[i, j, k]
                        L0 = L_yx[i + 1, j + 3]
                        L1 = L_yx[i + 1, j + 2]
                        L2 = L_yx[i + 1, j + 1]
                        Lx = L_yx[i + 1, j]
                        local_gain_expand[i + 1, j, k] = self.__fit_cos4th(
                            g0, g1, g2, L0, L1, L2, Lx
                        )
                    if i == local_gain.shape[0] - 1:
                        g0 = local_gain[i - 2, j, k]
                        g1 = local_gain[i - 1, j, k]
                        g2 = local_gain[i, j, k]
                        L0 = L_yx[i - 1, j + 1]
                        L1 = L_yx[i, j + 1]
                        L2 = L_yx[i + 1, j + 1]
                        Lx = L_yx[i + 2, j + 1]
                        local_gain_expand[i + 2, j + 1, k] = self.__fit_cos4th(
                            g0, g1, g2, L0, L1, L2, Lx
                        )
                    if j == local_gain.shape[1] - 1:
                        g0 = local_gain[i, j - 2, k]
                        g1 = local_gain[i, j - 1, k]
                        g2 = local_gain[i, j, k]
                        L0 = L_yx[i + 1, j - 1]
                        L1 = L_yx[i + 1, j]
                        L2 = L_yx[i + 1, j + 1]
                        Lx = L_yx[i + 1, j + 2]
                        local_gain_expand[i + 1, j + 2, k] = self.__fit_cos4th(
                            g0, g1, g2, L0, L1, L2, Lx
                        )
            H = local_gain_expand.shape[0]
            W = local_gain_expand.shape[1]
            gain_left_up = self.__fit_cos4th(
                local_gain_expand[0, 1, k],
                local_gain_expand[0, 2, k],
                local_gain_expand[0, 3, k],
                L_yx[0, 1],
                L_yx[0, 2],
                L_yx[0, 3],
                L_yx[0, 0],
            )
            gain_left_up += self.__fit_cos4th(
                local_gain_expand[1, 0, k],
                local_gain_expand[2, 0, k],
                local_gain_expand[3, 0, k],
                L_yx[1, 0],
                L_yx[2, 0],
                L_yx[3, 0],
                L_yx[0, 0],
            )
            local_gain_expand[0, 0, k] = gain_left_up / 2.0
            gain_left_bottom = self.__fit_cos4th(
                local_gain_expand[H - 1, 1, k],
                local_gain_expand[H - 1, 2, k],
                local_gain_expand[H - 1, 3, k],
                L_yx[H - 1, 1],
                L_yx[H - 1, 2],
                L_yx[H - 1, 3],
                L_yx[H - 1, 0],
            )
            gain_left_bottom += self.__fit_cos4th(
                local_gain_expand[H - 4, 0, k],
                local_gain_expand[H - 3, 0, k],
                local_gain_expand[H - 2, 0, k],
                L_yx[H - 4, 0],
                L_yx[H - 3, 0],
                L_yx[H - 2, 0],
                L_yx[H - 1, 0],
            )
            local_gain_expand[H - 1, 0, k] = gain_left_bottom / 2.0
            gain_right_up = self.__fit_cos4th(
                local_gain_expand[3, W - 1, k],
                local_gain_expand[2, W - 1, k],
                local_gain_expand[1, W - 1, k],
                L_yx[3, W - 1],
                L_yx[2, W - 1],
                L_yx[1, W - 1],
                L_yx[0, W - 1],
            )
            gain_right_up += self.__fit_cos4th(
                local_gain_expand[0, W - 4, k],
                local_gain_expand[0, W - 3, k],
                local_gain_expand[0, W - 2, k],
                L_yx[0, W - 4],
                L_yx[0, W - 3],
                L_yx[0, W - 2],
                L_yx[0, W - 1],
            )
            local_gain_expand[0, W - 1, k] = gain_right_up / 2.0
            gain_rb = self.__fit_cos4th(
                local_gain_expand[H - 4, W - 1, k],
                local_gain_expand[H - 3, W - 1, k],
                local_gain_expand[H - 2, W - 1, k],
                L_yx[H - 4, W - 1],
                L_yx[H - 3, W - 1],
                L_yx[H - 2, W - 1],
                L_yx[H - 1, W - 1],
            )
            gain_rb += self.__fit_cos4th(
                local_gain_expand[H - 1, W - 4, k],
                local_gain_expand[H - 1, W - 3, k],
                local_gain_expand[H - 1, W - 2, k],
                L_yx[H - 1, W - 4],
                L_yx[H - 1, W - 3],
                L_yx[H - 1, W - 2],
                L_yx[H - 1, W - 1],
            )
            local_gain_expand[H - 1, W - 1, k] = gain_rb / 2.0
            if plt_show:
                ax = plt.axes(projection="3d")
                ax.scatter3D(
                    coor_yx[:, 0],
                    coor_yx[:, 1],
                    local_gain_expand[:, :, k],
                    color="red",
                )
                ax.scatter3D(
                    coor_yx_local[:, 0],
                    coor_yx_local[:, 1],
                    local_gain[:, :, k],
                    color="green",
                )
                plt.show()
        local_gain_expand[1:-1, 1:-1, :] = local_gain_ori
        return local_gain_expand

    def __gain_quantify(self, gain_float):
        gain_bit_width = 16
        return rtl_round(
            torch.clip(gain_float * 64.0, 0, (1 << gain_bit_width) - 1),
            torch_format=True,
        )

    def __save_gain_to_txt(self, local_vertex_gain, height, width, file_path):
        data = []
        rtl_width = 32 if width < 32 else 64
        for i in range(0, height):
            for j in range(0, rtl_width):
                if j < width:
                    c = local_vertex_gain[i, j]
                    line0 = "{:08x}".format(int((c[0] * 2**16) + c[1]))
                    line1 = "{:08x}".format(int((c[2] * 2**16) + c[3]))
                    data.append(line0)
                    data.append(line1)
                else:
                    line0 = "{:08x}".format(0)
                    line1 = "{:08x}".format(0)
                    data.append(line0)
                    data.append(line1)
        f = open(file_path, "w")
        f.write("\n".join(data))

    def __save_gain_to_bin(
        self, local_vertex_gain, height, width, size_symbol, file_path
    ):
        data = []
        rtl_width = 32 if width < 32 else 64
        if size_symbol == 3:
            for i in range(0, height):
                for j in range(0, rtl_width):
                    if j < width:
                        gains = local_vertex_gain[i, j]
                        data.append(gains[1])
                        data.append(gains[0])
                        data.append(gains[3])
                        data.append(gains[2])
                    else:
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
            data = np.array(data, dtype=np.uint16)

            data.tofile(file_path)
        elif size_symbol == 4:
            for i in range(0, self.local_num[3][1] + 1):
                for j in range(0, rtl_width):
                    if j < width and i < height:
                        gains = local_vertex_gain[i, j]
                        data.append(gains[1])
                        data.append(gains[0])
                        data.append(gains[3])
                        data.append(gains[2])
                    else:
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
            data = np.array(data, dtype=np.uint16)

            data.tofile(file_path)
        elif (height == 24 and width == 41) or (height == 26 and width == 26):
            for i in range(0, height):
                for j in range(0, rtl_width):
                    if j < width:
                        gains = local_vertex_gain[i, j]
                        data.append(gains[1])
                        data.append(gains[0])
                        data.append(gains[3])
                        data.append(gains[2])
                    else:
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
            data = np.array(data, dtype=np.uint16)
            data.tofile(file_path)
        else:
            print("bin format gain is just support 400x400 and 720x720 image. skip.")

    def __get_gain_from_txt(self, gain_txt_path, height, width):
        f = open(gain_txt_path, "r")
        gain = torch.zeros((height * width, 4)).to(self.device)
        line_number = 0
        pixel_number = 0
        for line in f.readlines():
            value0 = int(line[0:4], 16)
            value1 = int(line[4:8], 16)
            if value0 == 0 and value1 == 0:
                continue
            pixel_number = line_number // 2
            if pixel_number * 2 == line_number:
                gain[pixel_number, 0] = value0
                gain[pixel_number, 1] = value1
            else:
                gain[pixel_number, 2] = value0
                gain[pixel_number, 3] = value1
            line_number += 1
        assert (pixel_number + 1) == (height * width), "gain txt size mismatch"
        return gain

    def __get_gain_from_bin(self, gain_bin_path, height, width, size_symbol):
        data = torch.from_numpy(
            np.fromfile(gain_bin_path, np.uint16).astype(np.float32)
        ).to(self.device)
        if size_symbol == 3:
            data = data.reshape((24, 32, 4))
            data = data[:, :, [1, 0, 3, 2]]
            return data[0:height, 0:width, :].reshape((-1, 4))
        elif size_symbol == 4:
            data = data.reshape((24, 32, 4))
            data = data[:, :, [1, 0, 3, 2]]
            return data[0:height, 0:width, :].reshape((-1, 4))
        elif height == 24 and width == 41:
            data = data.reshape((height, 64, 4))
            data = data[:, :, [1, 0, 3, 2]]
            return data[0:height, 0:width, :].reshape((-1, 4))
        else:
            print("bin format gain is just support 400x400 and 720x720 image. skip.")

    def lsc_pre(self, img_pre, golden_path):
        # img_pre replicate
        # for corners's error
        img_pre = self.__img_border_replicate(img_pre)

        # img_pre padding
        img_local_height = self.local_size[self.size_symbol][0]
        img_local_width = self.local_size[self.size_symbol][1]
        local_num_height = self.local_num[self.size_symbol][0]
        local_num_width = self.local_num[self.size_symbol][1]
        right_pad_length = (img_local_width * local_num_width - self.img_width) // 2
        bottom_pad_length = (img_local_height * local_num_height - self.img_height) // 2
        img_pre = torch.nn.functional.pad(
            img_pre, [0, right_pad_length, 0, bottom_pad_length], mode="reflect"
        )

        local_pixel_num_single_channel = (img_local_height // 2) * (
            img_local_width // 2
        )

        # calcuate local mean pixel value
        img_pre = (
            img_pre.reshape(
                [
                    1,
                    4,
                    local_num_height,
                    img_local_height // 2,
                    local_num_width,
                    img_local_width // 2,
                ]
            )
            .permute([0, 1, 2, 4, 3, 5])
            .reshape(
                [
                    1,
                    4,
                    local_num_height,
                    local_num_width,
                    local_pixel_num_single_channel,
                ]
            )
        )
        local_channel_mean = torch.mean(img_pre, dim=-1)

        # get the max value and gain
        local_max, max_indices = torch.max(
            local_channel_mean.reshape((1, 4, local_num_height * local_num_width)),
            dim=2,
        )
        local_gain = local_max.unsqueeze(-1).unsqueeze(-1) / local_channel_mean

        # gain expanded
        plt_show_s = False
        local_gain_expand0 = self.__fit_gain(
            local_gain.squeeze().permute([1, 2, 0]), plt_show_s
        )

        gain_axis = torch.concatenate(
            [
                local_gain_expand0.unsqueeze(0),
                torch.roll(local_gain_expand0, shifts=(0, -1), dims=(0, 1)).unsqueeze(
                    0
                ),
                torch.roll(local_gain_expand0, shifts=(-1, 0), dims=(0, 1)).unsqueeze(
                    0
                ),
                torch.roll(local_gain_expand0, shifts=(-1, -1), dims=(0, 1)).unsqueeze(
                    0
                ),
            ]
        )
        local_vertex_gain = torch.mean(gain_axis, dim=0)[:-1, :-1]

        print("before luma shading ratio")
        print(torch.max(local_vertex_gain[..., 0]))
        print(torch.max(local_vertex_gain[..., 1]))
        print(torch.max(local_vertex_gain[..., 2]))
        print(torch.max(local_vertex_gain[..., 3]))

        # luma shading and color shading
        # assume that the channel [self.ref_channel] is luma shading
        # and the other channels' diff from channel [self.ref_channel] is color chading
        luma_shading = local_vertex_gain[:, :, self.ref_channel].unsqueeze(-1)
        color_shading = local_vertex_gain / luma_shading
        new_luma_shading = (luma_shading - 1) * self.luma_shading_ratio + 1
        local_vertex_gain = color_shading * new_luma_shading
        print("after luma shading ratio")
        print(torch.max(local_vertex_gain[..., 0]))
        print(torch.max(local_vertex_gain[..., 1]))
        print(torch.max(local_vertex_gain[..., 2]))
        print(torch.max(local_vertex_gain[..., 3]))

        # gain fix
        local_vertex_gain_quantify = (
            self.__gain_quantify(local_vertex_gain).cpu().numpy()
        )

        file_name = (
            golden_path
            + "lsc_gain_"
            + str(self.img_height)
            + "x"
            + str(self.img_width)
            + "_RGGB"
        )

        self.__save_gain_to_txt(
            local_vertex_gain_quantify,
            local_num_height + 1,
            local_num_width + 1,
            file_name + ".txt",
        )
        self.__save_gain_to_bin(
            local_vertex_gain_quantify,
            local_num_height + 1,
            local_num_width + 1,
            self.size_symbol,
            file_name + ".bin",
        )

    def lsc_post(
        self,
        img_input: torch.Tensor,
        img_bit_width,
        golden_path,
        golden_gen_symbol,
    ):
        # Parameter initialization
        img_local_height = self.local_size[self.size_symbol][0]
        img_local_width = self.local_size[self.size_symbol][1]
        local_num_height = self.local_num[self.size_symbol][0]
        local_num_width = self.local_num[self.size_symbol][1]

        img_input = (
            img_input.reshape((2, 2, self.img_height // 2, self.img_width // 2))
            .permute([2, 0, 3, 1])
            .reshape((self.img_height, self.img_width))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if golden_gen_symbol:
            save_string_hex_unsigned(
                golden_path + "input.txt", img_input.squeeze(), img_bit_width
            )
        right_pad_length = img_local_width * local_num_width - self.img_width
        bottom_pad_length = img_local_height * local_num_height - self.img_height
        img_input = torch.nn.functional.pad(
            img_input, [0, right_pad_length, 0, bottom_pad_length], mode="reflect"
        )
        img_input = img_input.reshape(
            (local_num_height, img_local_height, local_num_width, img_local_width)
        ).permute([0, 2, 1, 3])

        img_input *= self.gain_weight_map_final

        # RGGB

        img_input = rtl_round(img_input, 16, True)

        img_input = img_input.permute([0, 2, 1, 3]).reshape(
            (
                self.img_padded_size[self.size_symbol][0],
                self.img_padded_size[self.size_symbol][1],
            )
        )
        img_input = torch.clip(img_input, 0, 2**img_bit_width - 1)[
            0 : self.img_height, 0 : self.img_width
        ]
        if golden_gen_symbol:
            save_string_hex_unsigned(
                golden_path + "output.txt", img_input, img_bit_width
            )

        return (
            img_input.reshape((self.img_height // 2, 2, self.img_width // 2, 2))
            .permute([1, 3, 0, 2])
            .reshape(1, 4, self.img_height // 2, self.img_width // 2)
        )

    def run(self, img_input: torch.Tensor):
        # Parameter initialization
        img_local_height = self.local_size[self.size_symbol][0]
        img_local_width = self.local_size[self.size_symbol][1]
        local_num_height = self.local_num[self.size_symbol][0]
        local_num_width = self.local_num[self.size_symbol][1]

        img_input = (
            img_input.reshape((2, 2, self.img_height // 2, self.img_width // 2))
            .permute([2, 0, 3, 1])
            .reshape((self.img_height, self.img_width))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        right_pad_length = img_local_width * local_num_width - self.img_width
        bottom_pad_length = img_local_height * local_num_height - self.img_height
        img_input = torch.nn.functional.pad(
            img_input, [0, bottom_pad_length, 0, right_pad_length], mode="reflect"
        )
        img_input = img_input.reshape(
            (local_num_height, img_local_height, local_num_width, img_local_width)
        ).permute([0, 2, 1, 3])

        img_input *= self.gain_weight_map_final

        img_input = rtl_round(img_input, 16, True)

        img_input = img_input.permute([0, 2, 1, 3]).reshape(
            (
                self.img_padded_size[self.size_symbol][0],
                self.img_padded_size[self.size_symbol][1],
            )
        )
        img_input = torch.clip(img_input, 0, 2**self.img_bit_width - 1)[
            0 : self.img_height, 0 : self.img_width
        ]

        return (
            img_input.reshape((self.img_height // 2, 2, self.img_width // 2, 2))
            .permute([1, 3, 0, 2])
            .reshape(1, 4, self.img_height // 2, self.img_width // 2)
        )


class context_lsc:
    def __init__(
        self, img_height, img_width, device, luma_shading_ratio, gain_path, gain_pattern
    ) -> None:
        self._modules = {
            "img_height": img_height,
            "img_width": img_width,
            "device": device,
            "luma_shading_ratio": luma_shading_ratio,
            "gain_path": gain_path,
            "gain_pattern": gain_pattern,
        }

    def get(self, module_name):
        return self._modules[module_name]


def test_pre():
    # bw = 12
    # root_path = (
    #     "/home/lizhan/workbench/1008_LSC_bitwidth/golden/800x800/" + str(bw) + "/"
    # )

    root_path = "/share/lizhan/workbench_datas/1010_imgprocess/1104/lizhan/img_mean/"
    img_pre_path = root_path + "lsc.png"

    img_pre = cv2.imread(img_pre_path, -1).astype(np.float32)
    # img_pre = np.flip(img_pre, axis=1)
    img_pre = torch.from_numpy(img_pre.copy()).cuda()
    img_height = img_pre.shape[0]
    img_width = img_pre.shape[1]

    context = context_lsc(img_height, img_width, "cuda", 0.25, "", "")

    # class init
    T1 = time.time()
    # obj_lsc_pre = lsc(img_height, img_width, 1.0, "BGGR")
    obj_lsc_pre = lsc(context)
    T2 = time.time()
    print("LSC pre init time cost", (T2 - T1) * 1000, "ms")
    # exit()

    # img (N C H W) bayer
    img_pre = (
        img_pre.reshape((img_height // 2, 2, img_width // 2, 2))
        .permute([1, 3, 0, 2])
        .reshape((1, 4, img_height // 2, img_width // 2))
    )

    # pre function, generate gain file (txt and bin)
    T1 = time.time()
    obj_lsc_pre.lsc_pre(
        img_pre=img_pre.clone(),
        golden_path=root_path,
    )
    T2 = time.time()
    print("LSC pre function time cost", (T2 - T1) * 1000, "ms")


def test_post():
    bw = 10
    # root_path = (
    #     "/home/lizhan/workbench/1008_LSC_bitwidth/golden/800x800/" + str(bw) + "/"
    # )
    root_path = "/home/lizhan/workbench/1002_ccm/ccm_lsc_0221/mean/"
    file_name = root_path + "lsc.png"
    img_pre = cv2.imread(file_name, -1).astype(np.float32)
    # img_pre = rtl_round(np.random.random((800, 800)) * (2**bw - 1))
    # img_pre = np.flip(img_pre, axis=1)
    img_pre = torch.from_numpy(img_pre.copy()).cuda()
    img_height = img_pre.shape[0]
    img_width = img_pre.shape[1]

    # class init
    T1 = time.time()
    context = context_lsc(
        img_height,
        img_width,
        "cuda",
        1.0,
        root_path + "lsc_gain_720x1280_RGGB.bin",
        "RGGB",
    )
    obj_lsc_post = lsc(context)
    T2 = time.time()
    print("LSC post init time cost", (T2 - T1) * 1000, "ms")

    # img (N C H W) bayer
    img_pre = (
        img_pre.reshape((img_height // 2, 2, img_width // 2, 2))
        .permute([1, 3, 0, 2])
        .reshape((1, 4, img_height // 2, img_width // 2))
    )

    # post(FPGA) function, do lsc to img
    T1 = time.time()
    dst = obj_lsc_post.lsc_post(
        img_input=img_pre,
        img_bit_width=bw,
        golden_path=root_path,
        golden_gen_symbol=False,
    )
    T2 = time.time()
    print("LSC time cost", (T2 - T1) * 1000, "ms")

    # do domosaic and gamma for show
    dst = (
        dst.reshape((2, 2, img_height // 2, img_width // 2))
        .permute([2, 0, 3, 1])
        .reshape((img_height, img_width))
    )
    dst = dst.cpu().numpy().astype(np.uint16)

    print(np.max(dst))
    print(np.mean(dst))

    dst = cv2.cvtColor(dst, cv2.COLOR_BAYER_GB2BGR).astype(np.float32)

    dst *= np.max(np.mean(dst, (0, 1))) / np.mean(dst, (0, 1))
    print(np.max(np.mean(dst, (0, 1))) / np.mean(dst, (0, 1)))

    dst = (dst / (2**bw - 1)) ** 0.45 * (2**bw - 1)
    cv2.imwrite(root_path + "dst.png", (dst * 2 ** (16 - bw)).astype(np.uint16))


def test_post_lsc(color):
    bw = 10
    # root_path = (
    #     "/home/lizhan/workbench/1008_LSC_bitwidth/golden/800x800/" + str(bw) + "/"
    # )
    root_path = "/home/lizhan/workbench/1002_ccm/ccm_lsc_0221/mean/"
    # file_name = root_path + "in.png"
    file_name = "/home/lizhan/workbench/1002_ccm/ccm_lsc_0221/mean/" + color + ".png"
    img_pre = cv2.imread(file_name, -1).astype(np.float32)
    # img_pre = rtl_round(np.random.random((800, 800)) * (2**bw - 1))
    # img_pre = np.flip(img_pre, axis=1)
    img_pre = torch.from_numpy(img_pre.copy()).cuda()
    img_height = img_pre.shape[0]
    img_width = img_pre.shape[1]

    # class init
    T1 = time.time()
    context = context_lsc(
        img_height,
        img_width,
        "cuda",
        1.0,
        root_path + "lsc_gain_720x1280_RGGB.bin",
        "RGGB",
    )
    obj_lsc_post = lsc(context)
    T2 = time.time()
    print("LSC post init time cost", (T2 - T1) * 1000, "ms")

    # img (N C H W) bayer
    img_pre = (
        img_pre.reshape((img_height // 2, 2, img_width // 2, 2))
        .permute([1, 3, 0, 2])
        .reshape((1, 4, img_height // 2, img_width // 2))
    )

    # post(FPGA) function, do lsc to img
    T1 = time.time()
    dst = obj_lsc_post.lsc_post(
        img_input=img_pre,
        img_bit_width=bw,
        golden_path=root_path,
        golden_gen_symbol=False,
    )
    T2 = time.time()
    print("LSC time cost", (T2 - T1) * 1000, "ms")

    # do domosaic and gamma for show
    dst = (
        dst.reshape((2, 2, img_height // 2, img_width // 2))
        .permute([2, 0, 3, 1])
        .reshape((img_height, img_width))
    )
    dst = dst.cpu().numpy().astype(np.uint16)

    cv2.imwrite(file_name.replace(".png", "_lsc.png"), (dst).astype(np.uint16))
    cv2.imwrite(
        file_name.replace(".png", "_lsc_vis.png"), (dst * 2**6).astype(np.uint16)
    )

    print(np.max(dst))
    print(np.mean(dst))

    dst = cv2.cvtColor(dst, cv2.COLOR_BAYER_GB2BGR).astype(np.float32)

    # dst *= np.max(np.mean(dst, (0, 1))) / np.mean(dst, (0, 1))
    # print(np.max(np.mean(dst, (0, 1))) / np.mean(dst, (0, 1)))

    dst = (dst / (2**bw - 1)) ** 0.45 * (2**bw - 1)
    cv2.imwrite(root_path + "dst.png", (dst * 2 ** (16 - bw)).astype(np.uint16))


# def lsc(img, bw=10):
#     # BGGR, (1, 4, 200, 200)
#     obj_lsc_post = len_shading_co rrect_bits(400, 400, 1.0, "BGGR", "lsc_gain.bin")
#     img = obj_lsc_post.lsc_post(
#         img_input=img, img_bit_width=bw, golden_path="", golden_gen_symbol=False
#     )
#     return img


def img_resize():
    h = 800
    w = 800
    size_str = str(w) + "x" + str(h)
    img_path = "/home/lizhan/workbench/1008_LSC_bitwidth/data/in.png"
    im = cv2.imread(img_path, -1)
    im = im.reshape((200, 2, 200, 2)).transpose([0, 2, 1, 3]).reshape((200, 200, 4))
    print(im.shape)

    im1 = np.zeros((h // 2, w // 2, 4))

    im1[..., 0] = cv2.resize(im[..., 0], (w // 2, h // 2))
    im1[..., 1] = cv2.resize(im[..., 1], (w // 2, h // 2))
    im1[..., 2] = cv2.resize(im[..., 2], (w // 2, h // 2))
    im1[..., 3] = cv2.resize(im[..., 3], (w // 2, h // 2))
    print(im1.shape)
    im1 = im1.reshape((h // 2, w // 2, 2, 2)).transpose((0, 2, 1, 3)).reshape((h, w))
    for ibw in [8, 10, 12]:
        im_ibw = im1.astype(np.float32) * 2 ** (ibw - 10)
        im_ibw = np.clip(im_ibw, 0, 2**ibw - 1).astype(np.uint16)
        cv2.imwrite(
            "/home/lizhan/workbench/1008_LSC_bitwidth/golden/"
            + size_str
            + "/"
            + str(ibw)
            + "/in.png",
            im_ibw.astype(np.uint16),
        )
        cv2.imwrite(
            "/home/lizhan/workbench/1008_LSC_bitwidth/golden/"
            + size_str
            + "/"
            + str(ibw)
            + "/in_vis.png",
            (im_ibw * 2 ** (16 - ibw)).astype(np.uint16),
        )


def calc_test():
    a = torch.tensor([0, 0.1, 0.2, 0.3]).cuda()
    b = torch.tensor([0.5]).cuda()

    # a = a / b
    a /= b
    print(a)


def test_get_gain_from_bin(gain_bin_path, height, width, size_symbol):
    data = torch.from_numpy(np.fromfile(gain_bin_path, np.uint16).astype(np.float32))
    if size_symbol == 3:
        data = data.reshape((24, 32, 4))
        data = data[:, :, [1, 0, 3, 2]]
        return data[0:height, 0:width, :].reshape((-1, 4))
    elif size_symbol == 4:
        data = data.reshape((24, 32, 4))
        data = data[:, :, [1, 0, 3, 2]]
        return data[0:height, 0:width, :].reshape((-1, 4))
    elif height == 24 and width == 41:
        data = data.reshape((height, 64, 4))
        data = data[:, :, [1, 0, 3, 2]]
        return data[0:height, 0:width, :].reshape((-1, 4))
    else:
        print("bin format gain is just support 400x400 and 720x720 image. skip.")


def save_gain_to_bin(local_vertex_gain, height, width, size_symbol, file_path):
    data = []
    rtl_width = 32 if width < 32 else 64
    if size_symbol == 3:
        for i in range(0, height):
            for j in range(0, rtl_width):
                if j < width:
                    gains = local_vertex_gain[i, j]
                    data.append(gains[1])
                    data.append(gains[0])
                    data.append(gains[3])
                    data.append(gains[2])
                else:
                    data.append(0)
                    data.append(0)
                    data.append(0)
                    data.append(0)
        data = np.array(data, dtype=np.uint16)
        data.tofile(file_path)


def cvt_bin_luma():
    gains = test_get_gain_from_bin("oh0fa10.bin", 24, 24, 3).reshape((24, 24, 4)) / 64.0
    print(gains.shape)
    print(gains[0, 0] * 64.0)
    luma_shading = gains[:, :, 0].unsqueeze(-1)
    color_shading = gains / luma_shading
    new_luma_shading = (luma_shading - 1) * 0.25 + 1
    gains = rtl_round(color_shading * new_luma_shading * 64.0)
    print(gains[0, 0])

    print(np.max(gains) / 64.0)

    save_gain_to_bin(gains, 24, 24, 3, "oh0fa10_0.25.bin")


if __name__ == "__main__":
    # img_resize()

    # calc_test()
    # exit()
    pass
    # test_pre()
    # test_post()

    # test_post_lsc("lsc")

    cvt_bin_luma()
    exit()
    test_post_lsc("blue")
    test_post_lsc("green")
    test_post_lsc("red")
    test_post_lsc("yellow")
