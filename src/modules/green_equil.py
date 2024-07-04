import torch
from .quantizer import q_attr, quantize, qmul, qdiv

supported_pattern = {
    "bggr": [3, 2, 1, 0],
    "gbrg": [2, 3, 0, 1],
    "rggb": [0, 1, 2, 3],
    "grbg": [1, 0, 3, 2],
}


class green_equil:
    def __init__(self, context) -> None:
        self._thresh = context.get("thresh")
        self._pattern = context.get("pattern")
        self._device = context.get("device")
        g_idx = [i for i in range(4) if self._pattern.lower()[i] == "g"]
        self._g_h_idx0 = g_idx[0] // 2
        self._g_w_idx0 = g_idx[0] % 2
        self._g_h_idx1 = g_idx[1] // 2
        self._g_w_idx1 = g_idx[1] % 2
        self._max_value = context.get("max_value")
        self._bw = context.get("bit_width")
        self._if_float = context.get("if_float")

    def run(self, x):
        _, _, h, w = x.shape
        order = supported_pattern[self._pattern.lower()]
        x = (
            x[:, order]
            .reshape([2, 2, h, w])
            .permute([2, 0, 3, 1])
            .reshape([h * 2, w * 2])
        )
        o1_1 = torch.roll(x, [1, 1], [-2, -1])
        o1_2 = torch.roll(x, [1, -1], [-2, -1])
        o1_3 = torch.roll(x, [-1, 1], [-2, -1])
        o1_4 = torch.roll(x, [-1, -1], [-2, -1])

        o2_1 = torch.roll(x, [2, 0], [-2, -1])
        o2_2 = torch.roll(x, [-2, 0], [-2, -1])
        o2_3 = torch.roll(x, [0, 2], [-2, -1])
        o2_4 = torch.roll(x, [0, -2], [-2, -1])

        d1 = o1_1 + o1_2 + o1_3 + o1_4
        d2 = o2_1 + o2_2 + o2_3 + o2_4

        c1 = (
            torch.abs(o1_1 - o1_2)
            + torch.abs(o1_1 - o1_3)
            + torch.abs(o1_1 - o1_4)
            + torch.abs(o1_2 - o1_3)
            + torch.abs(o1_2 - o1_4)
            + torch.abs(o1_3 - o1_4)
        )
        c2 = (
            torch.abs(o2_1 - o2_2)
            + torch.abs(o2_1 - o2_3)
            + torch.abs(o2_1 - o2_4)
            + torch.abs(o2_2 - o2_3)
            + torch.abs(o2_2 - o2_4)
            + torch.abs(o2_3 - o2_4)
        )
        c1 = torch.clip(c1, 0, (1 << (self._bw + 2)) - 1)
        c2 = torch.clip(c2, 0, (1 << (self._bw + 2)) - 1)

        gmp2p2 = x - torch.roll(x, [-2, -2], [-2, -1])
        gmm2m2 = x - torch.roll(x, [2, 2], [-2, -1])
        gmm2p2 = x - torch.roll(x, [2, -2], [-2, -1])
        gmp2m2 = x - torch.roll(x, [-2, 2], [-2, -1])

        gse = o1_4 + 0.5 * gmp2p2
        gnw = o1_1 + 0.5 * gmm2m2
        gne = o1_2 + 0.5 * gmm2p2
        gsw = o1_3 + 0.5 * gmp2m2

        eps = 1
        wtse = 1 / (eps + gmp2p2**2 + (torch.roll(x, [-3, -3], [-2, -1]) - o1_4) ** 2)
        wtnw = 1 / (eps + gmm2m2**2 + (torch.roll(x, [3, 3], [-2, -1]) - o1_1) ** 2)
        wtne = 1 / (eps + gmm2p2**2 + (torch.roll(x, [3, -3], [-2, -1]) - o1_2) ** 2)
        wtsw = 1 / (eps + gmp2m2**2 + (torch.roll(x, [-3, 3], [-2, -1]) - o1_3) ** 2)

        wq_attr = q_attr(False, False, 32, self._bw + 4, "rtl_round")
        if not self._if_float:
            wtse = quantize(wtse, wq_attr)
            wtnw = quantize(wtnw, wq_attr)
            wtne = quantize(wtne, wq_attr)
            wtsw = quantize(wtsw, wq_attr)

        denumer = gse * wtse + gnw * wtnw + gne * wtne + gsw * wtsw
        ori_numer = wtse + wtnw + wtne + wtsw
        ori_numer = torch.where(
            ori_numer == 0, torch.tensor(1).float().to(self._device), ori_numer
        )

        ginterpq = q_attr(False, False, self._bw + 32, 32, "rtl_round")
        ginterp = qdiv(denumer, ori_numer, ginterpq)
        ginterp = torch.where(ori_numer == 0, x, ginterp)

        roi = torch.where(
            (c1 + c2 < 6 * self._thresh * torch.abs(d1 - d2))
            & ((ginterp - x) < self._thresh * (ginterp + x))
        )

        tmp = x.clone()

        tmpq = q_attr(False, False, self._bw, 0, "rtl_round")
        tmp[roi] = qmul(0.5, (ginterp + x), tmpq)[roi]

        x[..., self._g_h_idx0 :: 2, self._g_w_idx0 :: 2] = tmp[
            ..., self._g_h_idx0 :: 2, self._g_w_idx0 :: 2
        ]
        x[..., self._g_h_idx1 :: 2, self._g_w_idx1 :: 2] = tmp[
            ..., self._g_h_idx1 :: 2, self._g_w_idx1 :: 2
        ]
        x = (
            x.reshape([h, 2, w, 2])
            .permute([1, 3, 0, 2])
            .reshape([1, 4, h, w])[:, order]
        )
        return x
