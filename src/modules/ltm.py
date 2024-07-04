import torch
import numpy as np


def gamma(data, max_value, gama_reci):
    data = data / max_value
    if gama_reci != 1 / 709:
        data = data ** gama_reci
    else:
        data = torch.where(data < 0.018, data * 4.5, 1.099 * data ** 0.45 - 0.099)

    return data * max_value


class ltm():
    def __init__(self, context) -> None:
        self._gray_max_ratio = context.get('gray_max_ratio')
        self._kernel = context.get('kernel')
        self._stride = context.get('stride')
        self._gama_reci = 1 / context.get('gama')
        self._frac_reci = 1 / context.get('frac')
        self._local_factor = context.get('local_factor')
        self._clahe_thresh = context.get('clahe_thresh')
        self._factor_thresh = context.get('factor_thresh')

        self._device = context.get('device')
        self._max_value = context.get('max_value')

        distance = torch.arange(1, self._kernel + 1, 1).float()
        distance[self._kernel // 2:] = torch.flip(distance[:self._kernel // 2], [0])
        weight_map = torch.unsqueeze(distance, -1) * distance
        self._weight_map = torch.unsqueeze(weight_map, -1).to(self._device)

        h, w = context.get('img_height') // 2, context.get('img_width') // 2
        h_block_num = (h - self._kernel) / self._stride + 1  # result need int
        w_block_num = (w - self._kernel) / self._stride + 1  # result need int
        block_num = int(h_block_num * w_block_num)

        weight_map_expand = weight_map.reshape([1, self._kernel**2, 1]
                                               ).expand(1, -1, block_num)
        self._weight_map_denominator = torch.nn.functional.fold(
            weight_map_expand, [h, w],
            kernel_size=self._kernel, stride=self._stride).to(self._device)
        
        self._log_max = np.log10(self._max_value)
        self._bin_step = 0.0625
        self._bins = int(np.ceil(self._log_max / self._bin_step))

    def run(self, x):
        # 1.1: bggr2gray, x --> gray
        mean_c = torch.mean(x, 1, keepdim=True)
        max_c = torch.max(x, 1, keepdim=True).values  # almost red in endoscope case
        gray = max_c * self._gray_max_ratio + mean_c * (1 - self._gray_max_ratio)

        # 1.2: tm_gama
        gray = gamma(gray, self._max_value, self._gama_reci)
        gray = torch.clip(gray, 1, self._max_value)

        # 2.1: img2log, gray --> gray_log
        gray_log = torch.log10(gray)
        gray_log = gray_log / torch.log10(torch.tensor(self._max_value)) * self._bins * self._bin_step
        
        # 2.2&3: global_hist, gray_log --> global_histogram
        global_histogram = torch.histc(gray_log, self._bins, 0, self._bins * self._bin_step)
        global_histogram = global_histogram / torch.sum(global_histogram)  # prob
        patches = torch.nn.functional.unfold(gray, kernel_size=self._kernel, stride=self._stride)
        patches_log = torch.nn.functional.unfold(gray_log, kernel_size=self._kernel, stride=self._stride)
        
        # 2.4: local_hist, gray_log --> local_hist
        local_hist = []
        for i in range(patches.shape[-1]):
            lh = torch.histc(patches_log[..., i], self._bins, 0, self._bins * self._bin_step)
            lh = lh / torch.sum(lh)  # prob
            local_hist.append(torch.unsqueeze(lh, 0))
        local_hist = torch.cat(local_hist, 0)
        
        # 2.5: weighted_hist, global_histogram+local_hist --> overall_hist
        overall_hist = self._local_factor*local_hist + (1-self._local_factor)*global_histogram
        
        # 2.6&7: cbrt_hist, overall_hist --> overall_hist(cbrt)
        overall_hist = torch.pow(overall_hist, self._frac_reci)
        overall_hist = overall_hist / torch.sum(overall_hist, -1, keepdim=True)
        
        # 2.8: limit_hist, overall_hist(cbrt) --> overall_hist(limit)
        overall_hist = torch.clip(overall_hist, 0, self._clahe_thresh)
        overall_hist = overall_hist + (1-torch.sum(overall_hist,-1,keepdim=True)) / overall_hist.shape[-1]
        
        # 2.9: ltm_lut, overall_hist(limit) --> acc_hist(lut)
        acc_hist = torch.zeros([overall_hist.shape[0], overall_hist.shape[1] + 1]).to(self._device)
        for i in range(1, acc_hist.shape[1]):
            acc_hist[:, i] = acc_hist[:, i - 1] + overall_hist[:, i - 1]

        # 3.1 logl2bin, xs(gl) --> xs(bin)
        xs = torch.arange(0, self._max_value + 1, 1).to(self._device)
        xs[0] = 1
        xs = torch.log10(xs) / self._bin_step
        
        # 3.2 bin2lut, xs(bin) --> patches[..., 120]
        xs_f, xs_c = torch.floor(xs), torch.ceil(xs)
        alpha = xs - xs_f
        block_num = acc_hist.shape[0]
        block_index = torch.arange(0, block_num, 1).reshape([block_num, 1]).long().to(self._device)
        lut = acc_hist[block_index, xs_f.long()] * (1 - alpha) + \
            acc_hist[block_index, xs_c.long()] * alpha
        for i in range(lut.shape[0]):
            patches[..., i] = lut[i][patches[..., i].long()] * self._max_value
        
        # 3.3 gray_adjust, patches[..., 120] --> dst
        patches = patches.reshape([1, self._kernel, self._kernel, -1])
        dst = (patches * self._weight_map).reshape([1, self._kernel ** 2, -1])
        dst = torch.nn.functional.fold(
            dst, [gray.shape[-2], gray.shape[-1]],
            kernel_size=self._kernel, stride=self._stride)
        dst = dst / self._weight_map_denominator
        
        # 3.4 divide_factor, dst --> coef
        coef = dst / gray
        
        # 4.1 sub sample
        coef = torch.nn.functional.interpolate(
            coef, scale_factor=0.5, mode='bilinear')
        
        # 4.2 up sample & factor th
        coef = torch.nn.functional.interpolate(
            coef, scale_factor=4, mode='bilinear')
        coef = (coef
                .reshape([x.shape[-2], 2, x.shape[-1], 2])
                .permute([1, 3, 0, 2])
                .reshape(x.shape))
        coef = torch.clip(coef, 0, self._factor_thresh)
        
        # 5 Bayer Adjust
        x = x * coef
        x = torch.clip(x, 0, self._max_value)

        return x
