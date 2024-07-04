# input image type: raw RGGB 4channels, 3color RGB 3channels
import torch

supported_pattern = {
    "bggr": [3, 2, 1, 0],
    "gbrg": [2, 3, 0, 1],
    "rggb": [0, 1, 2, 3],
    "grbg": [1, 0, 3, 2],
}


class undistortion():
    # __init__ variables for 1ch/3ch/4ch case
    # input:  from config file, self._device/._pattern/._max_value/._undistort_c/._mtx_f/._mtx_c/._dist_k/._dist_p
    # output: inverse mapping calculate, self._x_index/._y_index
    # declar: self._x_index_f/._y_index_f/._lt/._rt/._lb/._rb
    def __init__(self, context) -> None:
        self._device = context.get('device')
        self._pattern = context.get("pattern")
        self._max_value = context.get('max_value')
        ip_depth = context.get('undistort_depth')
        if self._max_value < (1<<ip_depth) - 1:
            self._max_value = ((1<<ip_depth)-1) / ((1<<ip_depth)/(self._max_value+1))
        self._undistort_c = context.get('undistort_channel')

        h, w = context.get('img_height'), context.get('img_width')
        c = self._undistort_c
        variables = ['mtx_f', 'mtx_c', 'dist_k', 'dist_p',
                     'mtx_f', 'mtx_c', 'dist_k', 'dist_p',
                     'mtx_f', 'mtx_c', 'dist_k', 'dist_p',
                     'mtx_f', 'mtx_c', 'dist_k', 'dist_p']
        for var in variables: setattr(self, '_' + var, torch.FloatTensor(context.get(var)).to(self._device))
        
        if c == 1:
            _mtx_cent_ = torch.stack([torch.tensor([[self._mtx_c[n]]]).to(self._device).unsqueeze(2) for n in range(3)])
            _mtx_foca_ = torch.stack([torch.tensor([[self._mtx_f[n]]]).to(self._device).unsqueeze(2) for n in range(2)])
            _dist_p_ = torch.stack([torch.tensor([[self._dist_p[n]]]).to(self._device).unsqueeze(2) for n in range(2)])
            _dist_k_ = torch.stack([torch.tensor([[self._dist_k[n]]]).to(self._device).unsqueeze(2) for n in range(3)])
        if c == 3:
            _mtx_cent_ = torch.stack([torch.tensor([[self._mtx_c[n]], [self._mtx_c[n]], [self._mtx_c[n]]]).to(self._device).unsqueeze(2) for n in range(3)])
            _mtx_foca_ = torch.stack([torch.tensor([[self._mtx_f[n]], [self._mtx_f[n]], [self._mtx_f[n]]]).to(self._device).unsqueeze(2) for n in range(2)])
            _dist_p_ = torch.stack([torch.tensor([[self._dist_p[n]], [self._dist_p[n]], [self._dist_p[n]]]).to(self._device).unsqueeze(2) for n in range(2)])
            _dist_k_ = torch.stack([torch.tensor([[self._dist_k[n]], [self._dist_k[n]], [self._dist_k[n]]]).to(self._device).unsqueeze(2) for n in range(3)])
        
        # Inverse camera lens distortion mapping
        # step1: target image pixel coordinate
        y0 = torch.arange(h).to(self._device).view(-1, 1).repeat(1, w).unsqueeze(2).repeat(1, 1, c).permute(2, 0, 1)
        x0 = torch.arange(w).to(self._device).view(1, -1).repeat(h, 1).unsqueeze(2).repeat(1, 1, c).permute(2, 0, 1)

        # step2: inverse camera coordinate without distortion
        y1 = (y0 - _mtx_cent_[1]) / _mtx_foca_[1]
        x1 = (x0 - _mtx_cent_[0] - y1 * _mtx_cent_[2]) / _mtx_foca_[0]

        x2 = torch.pow(x1, 2) 
        y2 = torch.pow(y1, 2)

        r2 = y2 + x2
        r4 = torch.pow(r2, 2)
        r6 = torch.pow(r2, 3)

        # step3: inverse camera coordinate after distort process
        x2 = x1*(1+_dist_k_[0]*r2+_dist_k_[1]*r4+_dist_k_[2]*r6) + _dist_p_[0]*x1*y1*2+_dist_p_[1]*(r2+2*x2)
        y2 = y1*(1+_dist_k_[0]*r2+_dist_k_[1]*r4+_dist_k_[2]*r6) + _dist_p_[1]*x1*y1*2+_dist_p_[0]*(r2+2*y2)

        # step4: inverse mapping pixel coordinate
        x_i = _mtx_foca_[0] * x2 + _mtx_cent_[2] * y2 + _mtx_cent_[0]
        y_i = _mtx_foca_[1] * y2 + _mtx_cent_[1]

        self._x_index = torch.clip(x_i, 0, w - 2)
        self._y_index = torch.clip(y_i, 0, h - 2)

        # SIM statis distort delta h & w
        print('∆h/2 max :{}'.format(torch.max(torch.abs(self._y_index-y0))))
        print('∆w/2 max :{}'.format(torch.max(torch.abs(self._x_index-x0))))

        # declar self variables, initialize zero
        self._x_index_f = torch.zeros_like(self._x_index)
        self._y_index_f = torch.zeros_like(self._x_index)
        self._lt = torch.zeros_like(self._x_index)
        self._rt = torch.zeros_like(self._x_index)
        self._lb = torch.zeros_like(self._x_index)
        self._rb = torch.zeros_like(self._x_index)

        # 1ch demosaic approach inter
        w0 = (
            torch.FloatTensor(
                [
                    [ 0,  0, -1,  0,  0],
                    [ 0,  0,  2,  0,  0],
                    [-1,  2,  4,  2, -1],
                    [ 0,  0,  2,  0,  0],
                    [ 0,  0, -1,  0,  0],
                ]
            )
            .reshape([1, 1, 5, 5])
            .to(self._device)
        ) / 8
        w1 = (
            torch.FloatTensor(
                [
                    [ 0,  0, 0.5,  0,  0],
                    [ 0, -1, 0  , -1,  0],
                    [-1,  4, 5  ,  4, -1],
                    [ 0, -1, 0  , -1,  0],
                    [ 0,  0, 0.5,  0,  0],
                ]
            )
            .reshape([1, 1, 5, 5])
            .to(self._device)
        ) / 8
        w2 = (
            torch.FloatTensor(
                [
                    [0  ,  0, -1,  0, 0  ],
                    [0  , -1,  4, -1, 0  ],
                    [0.5,  0,  5,  0, 0.5],
                    [0  , -1,  4, -1, 0  ],
                    [0  ,  0, -1,  0, 0  ],
                ]
            )
            .reshape([1, 1, 5, 5])
            .to(self._device)
        ) / 8
        w3 = (
            torch.FloatTensor(
                [
                    [ 0  , 0, -1.5, 0,  0  ],
                    [ 0  , 2,  0  , 2,  0  ],
                    [-1.5, 0,  6  , 0, -1.5],
                    [ 0  , 2,  0  , 2,  0  ],
                    [ 0  , 0, -1.5, 0,  0  ],
                ]
            )
            .reshape([1, 1, 5, 5])
            .to(self._device)
        ) / 8
        self._w = [
            [None, None, None],
            [None, None, None],
            [None, None, None],
            [None, None, None],
        ]
        self._w[0][1] = w0  # r-g
        self._w[0][2] = w3  # r-b
        self._w[1][0] = w1  # g-r
        self._w[1][2] = w2  # g-b
        self._w[2][0] = w2  # g-r
        self._w[2][2] = w1  # g-b
        self._w[3][0] = w3  # b-r
        self._w[3][1] = w0  # b-g

        self.y = torch.zeros([1, 3, h, w]).to(self._device)

    # _func_3ch: bilinear interpolation in r/g/b 3channel
    def _func_3ch(self, img):
        self._x_index_f = torch.floor(self._x_index).long()
        self._y_index_f = torch.floor(self._y_index).long()
        
        x_coef = torch.clip(self._x_index - self._x_index_f, 0, 1)
        y_coef = torch.clip(self._y_index - self._y_index_f, 0, 1)

        self._lt = (1 - x_coef) * (1 - y_coef)
        self._rt = x_coef * (1 - y_coef)
        self._lb = (1 - x_coef) * y_coef
        self._rb = x_coef * y_coef

        for i in range(0, 3):
            img[:,i,:,:] = (img[:, i, self._y_index_f[i], self._x_index_f[i]] * self._lt[i] +
                img[:, i, self._y_index_f[i], self._x_index_f[i] + 1] * self._rt[i] +
                img[:, i, self._y_index_f[i] + 1, self._x_index_f[i]] * self._lb[i] +
                img[:, i, self._y_index_f[i] + 1, self._x_index_f[i] + 1] * self._rb[i])
        
        return img
    
    # _rggb2raw: rggb to actual bayer pattern, only used in _func_1ch func
    def _rggb2raw(self, img):
        _, _, h, w = img.shape
        order = supported_pattern[self._pattern.lower()]
        img = (
            img[:, order]
            .reshape([2, 2, h, w])
            .permute([2, 0, 3, 1])
            .reshape([1, 1, h * 2, w * 2]))
        
        return img
    
    # _raw2rggb: actual bayer pattern to rggb, only used in _func_1ch func
    def _raw2rggb(self, img):
        _, _, h, w = img.shape
        order = supported_pattern[self._pattern.lower()]
        img = (img[0, 0, :, :]
                .reshape([h//2, 2, w//2, 2])
                .permute(0, 2, 1, 3)
                .reshape([h//2, w//2, 4])[..., order]
                .unsqueeze(0)
                .permute([0, 3, 1, 2]))
        
        return img
    
    # _func_1ch_bilinear: bilinear interpolation
    def _func_1ch_bilinear(self, img):
        # rggb to actual bayer pattern
        img = self._rggb2raw(img)

        x_index = self._x_index
        y_index = self._y_index

        x_coef = torch.zeros_like(self._x_index)
        y_coef = torch.zeros_like(self._x_index)

        self._x_index_f[0,:,0::2] = torch.floor(torch.floor(x_index[0, :, 0::2])/2)*2
        self._x_index_f[0,:,1::2] = torch.floor((torch.floor(x_index[0, :, 1::2])+1)/2)*2-1
        self._y_index_f[0,0::2,:] = torch.floor(torch.floor(y_index[0, 0::2, :])/2)*2
        self._y_index_f[0,1::2,:] = torch.floor((torch.floor(y_index[0, 1::2, :])+1)/2)*2-1

        self._x_index_f = torch.floor(self._x_index_f).long()
        self._y_index_f = torch.floor(self._y_index_f).long()

        x_coef[0,:,0::2] = torch.clip((x_index[0,:,0::2]-torch.floor(x_index[0,:,0::2]))/2 + 0.5*torch.remainder(torch.floor(x_index[0,:,0::2]),2), 0, 1)
        x_coef[0,:,1::2] = torch.clip((x_index[0,:,1::2]-torch.floor(x_index[0,:,1::2]))/2 + 0.5*torch.remainder(torch.floor(x_index[0,:,1::2])-1,2), 0, 1)
        y_coef[0,0::2,:] = torch.clip((y_index[0,0::2,:]-torch.floor(y_index[0,0::2,:]))/2 + 0.5*torch.remainder(torch.floor(y_index[0,0::2,:]),2), 0, 1)
        y_coef[0,1::2,:] = torch.clip((y_index[0,1::2,:]-torch.floor(y_index[0,1::2,:]))/2 + 0.5*torch.remainder(torch.floor(y_index[0,1::2,:])-1,2), 0, 1)

        self._lt = (1 - x_coef) * (1 - y_coef)
        self._rt = x_coef * (1 - y_coef)
        self._lb = (1 - x_coef) * y_coef
        self._rb = x_coef * y_coef

        # interpolation process
        for i in range(0, 1):
            img[:,i,:,:] = (img[:, i, self._y_index_f[i], self._x_index_f[i]] * self._lt[i] +
                img[:, i, self._y_index_f[i], self._x_index_f[i] + 2] * self._rt[i] +
                img[:, i, self._y_index_f[i] + 2, self._x_index_f[i]] * self._lb[i] +
                img[:, i, self._y_index_f[i] + 2, self._x_index_f[i] + 2] * self._rb[i])
            
        # actual bayer pattern to rggb
        img = self._raw2rggb(img)
        
        return img
    
    # _func_1ch_bilinearplus: bilinear interpolation after pixel demosaic
    def _func_1ch_bilinearplus(self, img):
        # rggb to actual bayer pattern
        x = self._rggb2raw(img)

        self._x_index_f = torch.floor(self._x_index).long()
        self._y_index_f = torch.floor(self._y_index).long()
        
        x_coef = torch.clip(self._x_index - self._x_index_f, 0, 1)
        y_coef = torch.clip(self._y_index - self._y_index_f, 0, 1)

        self._lt = (1 - x_coef) * (1 - y_coef)
        self._rt = x_coef * (1 - y_coef)
        self._lb = (1 - x_coef) * y_coef
        self._rb = x_coef * y_coef

        # demosaic
        order = supported_pattern[self._pattern.lower()]
        x_padded = torch.nn.functional.pad(x, [1, 1, 1, 1], "reflect")
        for i in range(4):
            for j in range(3):  # 0:r, 1:g, 2:b
                w = self._w[order[i]][j]
                if w is None:
                    self.y[:, j, i // 2 :: 2, i % 2 :: 2] = x[
                        0, 0, i // 2 :: 2, i % 2 :: 2
                    ]
                    continue
                pad_l = (i % 2 + 1) % 2
                pad_r = 1 - pad_l
                pad_t = 1 - (i // 2)
                pad_b = 1 - pad_t
                ch = torch.nn.functional.conv2d(
                    torch.nn.functional.pad(
                        x_padded, [pad_l, pad_r, pad_t, pad_b], "reflect"
                    ),
                    w,
                    stride=2,
                )
                self.y[:, j, i // 2 :: 2, i % 2 :: 2] = ch
        img = torch.clip(self.y, 0, self._max_value)
        
        img_out = torch.zeros_like(x)
        # interpolation process
        for i in range(0, 3):
            img[:,i,:,:] = (img[:, i, self._y_index_f[i//3], self._x_index_f[i//3]] * self._lt[i//3] +
                img[:, i, self._y_index_f[i//3], self._x_index_f[i//3] + 1] * self._rt[i//3] +
                img[:, i, self._y_index_f[i//3] + 1, self._x_index_f[i//3]] * self._lb[i//3] +
                img[:, i, self._y_index_f[i//3] + 1, self._x_index_f[i//3] + 1] * self._rb[i//3])
        
        img_out[0,0,0::2,0::2] = img[0,int((order[0]/2+0.5)//1),0::2,0::2]
        img_out[0,0,0::2,1::2] = img[0,int((order[1]/2+0.5)//1),0::2,1::2]
        img_out[0,0,1::2,0::2] = img[0,int((order[2]/2+0.5)//1),1::2,0::2]
        img_out[0,0,1::2,1::2] = img[0,int((order[3]/2+0.5)//1),1::2,1::2]

        img_out = self._raw2rggb(img_out)
        
        return img_out
    
    def run(self, img):

        if self._undistort_c == 3:
            img = self._func_3ch(img)

        if self._undistort_c == 1:
            # img = self._func_1ch_bilinear(img)
            img = self._func_1ch_bilinearplus(img)
        
        return torch.clip(img, 0, self._max_value)