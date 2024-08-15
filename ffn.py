"""GPT Blocks used for the GPT Model."""
from typing import Any, Optional
import torch
import torch.nn as nn
from .fc import FC_CLASS_REGISTRY
try:
    import transformer_engine.pytorch as te
except:
    te = None

class MPTMLP(nn.Module):

    def __init__(self, hidden_size: int, expansion_ratio: int, fc_type: str='torch', device: Optional[str]=None):
        super().__init__()
        fc_kwargs = {}
        if fc_type != 'te':
            fc_kwargs['device'] = device
        self.up_proj = FC_CLASS_REGISTRY[fc_type](hidden_size, expansion_ratio * hidden_size, **fc_kwargs)
        self.act = nn.GELU(approximate='none')
        self.down_proj = FC_CLASS_REGISTRY[fc_type](expansion_ratio * hidden_size, hidden_size, **fc_kwargs)
        self.down_proj._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))
FFN_CLASS_REGISTRY = {'mptmlp': MPTMLP}
if te is not None:
    te.LayerNormMLP._has_norm = True
    FFN_CLASS_REGISTRY['te_ln_mlp'] = te.LayerNormMLP

def build_ffn(hidden_size: int, expansion_ratio: int, fc_type: str='torch', device: Optional[str]=None, **kwargs: Any) -> nn.Module:
    ffn_type = kwargs.pop('ffn_type')
    if ffn_type == 'mptmlp':
        if len(kwargs) > 0:
            raise ValueError(f'MPTMLP got an unexpected keyword argument: {kwargs}')
        return MPTMLP(hidden_size=hidden_size, expansion_ratio=expansion_ratio, fc_type=fc_type, device=device)
    elif ffn_type == 'te_ln_mlp':
        assert te is not None
        return te.LayerNormMLP(hidden_size=hidden_size, ffn_hidden_size=hidden_size * expansion_ratio, **kwargs)
    raise ValueError(f'ffn_type={ffn_type!r} not recognized.')