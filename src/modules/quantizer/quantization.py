import torch


class q_attr:
    def __init__(self, if_float, signed, bw, fp, round_mode) -> None:
        self.if_float = if_float
        self.signed = signed
        self.bw = bw
        self.fp = fp
        self.round_mode = round_mode


def quantize(x, q):
    signed = q.signed
    bw = q.bw
    fp = q.fp
    round_mode = q.round_mode
    x = x * (1 << fp)
    if round_mode == "floor":
        x = torch.floor(x)
    elif round_mode == "ceil":
        x = torch.ceil(x)
    elif round_mode == "rtl_round":
        x = torch.where(x - torch.floor(x) >= 0.5, torch.ceil(x), torch.floor(x))
    if signed:
        x = torch.clip(x, -(1 << (bw - 1)), (1 << (bw - 1)) - 1)
    else:
        x = torch.clip(x, 0, (1 << bw) - 1)
    return x * 2 ** (-fp)


def qadd(a, b, q):
    if q.if_float:
        return a + b
    else:
        return quantize(a + b, q)


def qsub(a, b, q):
    if q.if_float:
        return a - b
    else:
        return quantize(a - b, q)


def qmul(a, b, q):
    if q.if_float:
        return a * b
    else:
        return quantize(a * b, q)


def qdiv(a, b, q):
    if q.if_float:
        return a / b
    else:
        return quantize(a / b, q)
