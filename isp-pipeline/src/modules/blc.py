import torch
from .quantizer import q_attr, qsub


class blc:
    def __init__(self, context) -> None:
        self._black_level = torch.FloatTensor([context.get("black_level")]).to(
            context.get("device")
        )
        self._bw = context.get("bit_width")
        self._max_value = context.get("max_value")
        self._if_float = context.get("if_float")

    def run(self, x):
        output_quantization_attributes = q_attr(
            self._if_float, False, self._bw, 0, "rtl_round"
        )
        x = qsub(x, self._black_level, output_quantization_attributes)
        if self._if_float:
            return torch.clip(x, 0, self._max_value)
        return x
