"""Fixed, spatial and query pooling from 49 CLIP patches to 16 slots.

Selected from the archived experiments; run-specific I/O has been removed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = (49, 16)
MODES = ('fixed', 'spatial', 'query')

def pool_patches(patches, vision_tokens):
    if vision_tokens not in TOKENS or patches.ndim != 3 or patches.shape[1:] != (49, 768):
        raise ValueError('expected [B,49,768] row-major CLIP patches and 49/16 output tokens')
    if vision_tokens == 49:
        return patches
    grid = patches.float().transpose(1, 2).reshape(len(patches), 768, 7, 7)
    # Average in FP32; keep the existing cache precision at the projector boundary.
    return F.adaptive_avg_pool2d(grid, (4, 4)).flatten(2).transpose(1, 2).to(patches.dtype).contiguous()


def spatial_support():
    mask = torch.zeros(16, 49, dtype=torch.bool)
    for y in range(4):
        for x in range(4):
            grid = mask[y*4+x].reshape(7,7)
            grid[y*7//4:((y+1)*7+3)//4, x*7//4:((x+1)*7+3)//4] = True
    return mask


def position_encoding(side, width=128):
    yy, xx = torch.meshgrid(torch.linspace(0,1,side), torch.linspace(0,1,side), indexing='ij')
    frequency = 1. / (10000 ** (torch.arange(width//4).float()/(width//4)))
    def encode(coords):
        angles = coords.flatten()[:,None] * (2*math.pi) * frequency[None]
        return torch.cat((angles.sin(),angles.cos()),dim=-1)
    return torch.cat((encode(yy),encode(xx)),dim=-1)


class FixedPool(nn.Module):
    def forward(self, patches):
        return pool_patches(patches,16)


class SpatialPool(nn.Module):
    """100 active logits on the original overlapping windows; no content input."""
    def __init__(self):
        super().__init__()
        support = spatial_support()
        self.register_buffer('support', support)
        self.logits = nn.Parameter(torch.zeros(int(support.sum())))
        initial = torch.zeros_like(support, dtype=torch.float32).masked_fill(~support, -torch.inf).softmax(-1)
        self.register_buffer('initial_weights', initial)

    def weights(self):
        scores = self.logits.new_full((16,49), -torch.inf)
        return scores.masked_scatter(self.support, self.logits).softmax(-1)

    def forward(self, patches):
        baseline = pool_patches(patches,16)
        # Explicit FP32 delta preserves exact baseline output at zero initialization.
        with torch.autocast(patches.device.type, enabled=False):
            delta = self.weights().float() - self.initial_weights
            correction = torch.einsum('ij,bjd->bid', delta, patches.float())
            output = baseline.float() + correction
        return output.to(patches.dtype)


class QueryPool(nn.Module):
    """One four-head low-rank content attention, zero-initialized signed residual gates."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.q_proj = nn.Linear(768,128,bias=False)
        self.k_proj = nn.Linear(768,128,bias=False)
        self.queries = nn.Parameter(torch.randn(16,128)*.02)
        self.gates = nn.Parameter(torch.zeros(16))
        self.register_buffer('key_positions',position_encoding(7))
        self.register_buffer('query_positions',position_encoding(4))

    def attend(self,patches,baseline):
        batch = len(patches)
        keys = self.k_proj(self.norm(patches.float())) + self.key_positions
        queries = self.q_proj(self.norm(baseline.float())) + self.queries + self.query_positions
        queries = queries.reshape(batch,16,4,32).transpose(1,2)
        keys = keys.reshape(batch,49,4,32).transpose(1,2)
        scores = torch.matmul(queries,keys.transpose(-1,-2))/math.sqrt(32)
        weights = scores.float().softmax(-1)
        values = patches.float().reshape(batch,49,4,192).transpose(1,2)
        attended = torch.matmul(weights,values).transpose(1,2).reshape(batch,16,768)
        return attended,weights

    def forward(self,patches):
        baseline = pool_patches(patches,16)
        attended,_ = self.attend(patches,baseline)
        output = baseline.float() + self.gates.tanh()[None,:,None] * (attended.float()-baseline.float())
        return output.to(patches.dtype)


class LearnedCaptioner(nn.Module):
    def __init__(self,base,mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('unsupported compressor')
        self.base = base
        self.base.vision_tokens = 16
        self.mode = mode
        self.compressor = {'fixed':FixedPool,'spatial':SpatialPool,'query':QueryPool}[mode]()

    @property
    def llm(self):
        return self.base.llm

    @property
    def projector(self):
        return self.base.projector

    @property
    def vision_tokens(self):
        return self.base.vision_tokens

    def forward(self,input_ids,patches=None,**kwargs):
        if patches is not None:
            if kwargs.get('past_key_values') is not None:
                raise ValueError('compressor may run only in image prefill')
            patches = self.compressor(patches)
        return self.base(input_ids,patches=patches,**kwargs)
