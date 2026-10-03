import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from pytorch_wavelets import DWTForward, DWTInverse
from pywt import WaveletPacket
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # print(f"PatchEmbed2D input shape of x:{x.shape}") # B, C, H, W
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        
        # print(f"PatchEmbed2D output shape of x:{x.shape}") # B, H/4, W/4, 96
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        # print(f"PatchMerging2D input shape of x:{x.shape}")
        
        if x.dim() == 5:
            B, T, H, W, C = x.shape
            has_temporal = True
            x = x.reshape(B*T, H, W, C)
        else:
            B, H, W, C = x.shape
            has_temporal = False
            T = 1

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B*T if has_temporal else B, H//2, W//2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        
        if has_temporal:
            x = x.reshape(B, T, H//2, W//2, -1)
        
        # print(f"PatchMerging2D output shape of x:{x.shape}")
        return x
    

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn

        # Standard SS2D.
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y




    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # print(f"SS2D shape of x:{x.shape}")

        xz = self.in_proj(x)
        # print(f"SS2D shape of xz:{xz.shape}")
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        # print(f"SS2D shape of x:{x.shape}")
        # print(f"SS2D shape of z:{z.shape}")

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        # print(f"SS2D shape of x:{x.shape}")
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        # out = y
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        # print(f"SS2D shape of out:{out.shape}")
        return out


class SS2D_local(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    def local_scan(self, x: torch.Tensor, H: int = 14, W: int = 14, w: int = 7, flip: bool = False, column_first: bool = False):
        """Local windowed scan in LocalMamba
        Input: 
            x: [B, C, H, W]
            H, W: original width and height
            column_first: column-wise scan first (the additional direction in VMamba)
        Return: [B, C, L]
        """
        # Validate dimensions early to provide clearer debugging errors.
        assert len(x.shape) == 4, f"x must be a 4-dimensional tensor [B, C, H, W], current dimension:{x.shape}"
        B, C, _, _ = x.shape
        x = x.view(B, C, H, W)
        Hg, Wg = math.floor(H / w), math.floor(W / w)
        if H % w != 0 or W % w != 0:
            newH, newW = Hg * w, Wg * w
            x = x[:, :, :newH, :newW]
            print(f"Note: H/W cannot be divided by w, and has been trimmed to newH={newH}, newW={newW}")
        if column_first:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
        else:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
        if flip:
            x = x.flip([-1])
        return x
    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x1 = self.local_scan(x, H, W, H//4)
        x2 = self.local_scan(x, H, W, H//4, column_first=True)
        x3 = self.local_scan(x, H, W, H//4, flip=True)
        x4 = self.local_scan(x, H, W, H//4, column_first=True, flip=True)
        
        xs = torch.stack([x1,x2,x3,x4],dim=1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # print(f"SS2D_local shape of x:{x.shape}")

        xz = self.in_proj(x)
        # print(f"SS2D shape of xz:{xz.shape}")
        x, z = xz.chunk(2, dim=-1)
        # print(f"SS2D_local shape of x:{x.shape}")
        # print(f"SS2D_local shape of z:{z.shape}")

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        # print(f"SS2D_local shape of out:{out.shape}")
        return out


class SS2D_spiral(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        
        # spiral scan
        self.forward_core = self.forward_core_spiral

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_core_spiral(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        
        pos_i = torch.empty(0, dtype=torch.long)
        pos_j = torch.empty(0, dtype=torch.long)
        
        tl_h, tl_w = 0, 0
        br_h, br_w = H - 1, W - 1 
        
        while True:
            pos_i_ = torch.cat([
                torch.full((br_w - tl_w,), tl_h, dtype=torch.long),
                torch.arange(tl_h, br_h, dtype=torch.long),
                torch.full((br_w - tl_w,), br_h, dtype=torch.long),
                torch.arange(br_h, tl_h, -1, dtype=torch.long)
            ])
            pos_j_ = torch.cat([
                torch.arange(tl_w, br_w, dtype=torch.long),
                torch.full((br_h - tl_h,), br_w, dtype=torch.long),
                torch.arange(br_w, tl_w, -1, dtype=torch.long),
                torch.full((br_h - tl_h,), tl_w, dtype=torch.long)
            ])
            
            pos_i = torch.cat([pos_i, pos_i_])
            pos_j = torch.cat([pos_j, pos_j_])
            
            tl_h, tl_w, br_h, br_w = tl_h + 1, tl_w + 1, br_h - 1, br_w - 1
            
            if tl_h > br_h or tl_w > br_w:
                break
            elif tl_h == br_h and tl_w == br_w:
                pos_i = torch.cat([pos_i, torch.LongTensor([tl_h])])
                pos_j = torch.cat([pos_j, torch.LongTensor([tl_w])])
        
        pos_i = pos_i[:L]
        pos_j = pos_j[:L]
        
        x_spiral = x[:, :, pos_i, pos_j]
        
        K = 1
        xs = x_spiral.reshape(B, -1, L).unsqueeze(1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight[:K])
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight[:K])
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)[:C]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)[:C]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)[:C]
        
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float
        
        out_spiral = out_y[:, 0]
        
        inv_idx = torch.argsort(pos_i * W + pos_j)
        out_inv = out_spiral[:, :, inv_idx]
        
        spiral_2d = out_spiral.reshape(B, -1, H, W)
        wh_y = torch.transpose(spiral_2d, dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        inv_wh_2d = out_inv.reshape(B, -1, H, W)
        invwh_y = torch.transpose(inv_wh_2d, dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        return out_spiral, out_inv, wh_y, invwh_y


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # print(f"SS2D shape of x:{x.shape}")

        xz = self.in_proj(x)
        # print(f"SS2D shape of xz:{xz.shape}")
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        # print(f"SS2D shape of x:{x.shape}")
        # print(f"SS2D shape of z:{z.shape}")

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        # print(f"SS2D shape of x:{x.shape}")
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        # out = y
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        # print(f"SS2D shape of out:{out.shape}")
        return out


class SS2D_local_temporal(nn.Module):
    """
    Time-spatial version of Local SS2D
    Supports 5D input [B, T, H, W, C] or 4D input [B, H, W, C]
    Scanning order: For each spatial position, first traverse all time steps, and then move the spatial position according to the order of Local_SS2D
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        self.window_size = 7

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def local_scan_spatial(self, x: torch.Tensor, H: int, W: int, w: int = 7, flip: bool = False, column_first: bool = False):
        """
        Local scan in spatial dimensions (reusing the logic of SS2D_local)
        x: [B, C, H, W]
        Return: [B, C, L_spatial]
        """
        B, C, _, _ = x.shape
        # print(f"x.shape: {x.shape}")
        Hg, Wg = math.floor(H / w), math.floor(W / w)
        # print(f"Hg = {Hg}, Wg = {Wg}")
        if H % w != 0 or W % w != 0:
            newH, newW = Hg * w, Wg * w
            x = x[:, :, :newH, :newW]
            # print(f"newH = {newH}, newW = {newW}")
        
        if column_first:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
        else:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
        if flip:
            x = x.flip([-1])
        # print(f"x.shape after local_scan_spatial: {x.shape}")
        
        return x

    def forward_core_temporal_spatial(self, x: torch.Tensor, T: int, H: int, W: int):
        """
        Core Logic of Time-Space Scanning
        x: [B, C, T, H, W]

        Scanning sequence:
        1. For each time step, traverse all spatial positions
        2. Then move to the next time step

        Final sequence length: L_new = T * L_spatial
        """
        B, C, _, _, _ = x.shape
        L_spatial = H * W
        L_total = T * L_spatial
        K = 4

        # print(f"DEBUG forward_core_temporal_spatial: x.shape={x.shape}, B={B}, C={C}, T={T}, H={H}, W={W}")
        # print(f"DEBUG: L_spatial={L_spatial}, L_total={L_total}")
        # print(f"DEBUG: self.x_proj_weight.shape={self.x_proj_weight.shape}")
        
        xs_list = []
        
        for direction in range(4):
            flip = (direction >= 2)
            column_first = (direction % 2 == 1)
            
            direction_sequences = []
            
            for t in range(T):
                x_t = x[:, :, t, :, :]
                seq_t = self.local_scan_spatial(x_t, H, W, H//4, flip=flip, column_first=column_first)
                direction_sequences.append(seq_t)
            
            direction_stack = torch.stack(direction_sequences, dim=2)
            direction_flat = direction_stack.reshape(B, C, T * L_spatial)
            xs_list.append(direction_flat)
        
        xs = torch.stack(xs_list, dim=1)
        # print(f"DEBUG: xs.shape after stack={xs.shape}")
        # print(f"DEBUG: xs.view(B, K, -1, L_total).shape={xs.view(B, K, -1, L_total).shape}")
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L_total), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L_total), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L_total)
        dts = dts.contiguous().float().view(B, -1, L_total)
        Bs = Bs.float().view(B, K, -1, L_total)
        Cs = Cs.float().view(B, K, -1, L_total)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L_total)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L_total)
        
        out_merged = out_y[:, 0] + inv_y[:, 0]
        
        return out_merged

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: [B, H, W, C] or [B, T, H, W, C]
        """
        if x.dim() == 5:
            B, T, H, W, C = x.shape
            has_temporal = True
        else:
            B, H, W, C = x.shape
            T = 1
            has_temporal = False

        xz = self.in_proj(x)
        x_feat, z = xz.chunk(2, dim=-1)

        if has_temporal:
            x_feat_4d = x_feat.reshape(B*T, H, W, -1)
            x_feat_4d = x_feat_4d.permute(0, 3, 1, 2).contiguous()
            x_feat_4d = self.act(self.conv2d(x_feat_4d))
            x_feat_5d = x_feat_4d.reshape(B, T, -1, H, W)
            
            x_feat_5d = x_feat_5d.permute(0, 2, 1, 3, 4).contiguous()
            
            out_merged = self.forward_core_temporal_spatial(x_feat_5d, T, H, W)
            out_merged = out_merged.reshape(B, -1, T, H, W)
            
            # Process each time step independently.
            out_list = []
            for t in range(T):
                out_t = out_merged[:, :, t, :, :]  # (B, C_new, H, W)
                out_t = out_t.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C_new)
                out_t = self.out_norm(out_t)
                
                z_t = z[:, t, :, :, :]  # (B, H, W, C_z)
                out_t = out_t * F.silu(z_t)
                out_t = self.out_proj(out_t)
                out_list.append(out_t)
            
            out = torch.stack(out_list, dim=1)  # (B, T, H, W, C_out)
        else:
            x_feat = x_feat.permute(0, 3, 1, 2).contiguous()
            x_feat = self.act(self.conv2d(x_feat))
            
            x_4d = x_feat.unsqueeze(2)
            out_merged = self.forward_core_temporal_spatial(x_4d, 1, H, W)
            
            out_merged = out_merged.squeeze(2)
            out_2d = out_merged.permute(0, 2, 1).contiguous()
            out_2d = out_2d.view(B, H, W, -1)
            out_2d = self.out_norm(out_2d)
            out_2d = out_2d * F.silu(z)
            out = self.out_proj(out_2d)

        if self.dropout is not None:
            out = self.dropout(out)
        return out




class SS_Conv_SSM(nn.Module):
    def __init__(
        self,
        wave: str = 'haar', 
        J: int = 2, 
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        expand: float = 2.,
        use_temporal: bool = False,
        active_branches=("spatial", "frequency", "local"),
        **kwargs,
    ):
        super().__init__()
        self.wave = wave
        self.J = J
        self.use_temporal = use_temporal
        self.active_branches = set(active_branches)
        valid_branches = {"spatial", "frequency", "local"}
        unknown_branches = self.active_branches - valid_branches
        if unknown_branches:
            raise ValueError(f"Unknown branches: {sorted(unknown_branches)}")
        self.ln_1 = norm_layer(hidden_dim//3)
        self.ln_11 = norm_layer(hidden_dim)

        self.self_attention_ssm = SS2D(d_model=hidden_dim//3, dropout=attn_drop_rate, d_state=d_state, **kwargs)

        if use_temporal:
            self.self_attention_dwt = SS2D_local_temporal(d_model=hidden_dim//3, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        else:
            self.self_attention_dwt = SS2D_local(d_model=hidden_dim//3, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)

        self.drop_path = DropPath(drop_path)
        self.drop_path1 = DropPath(drop_path)

        self.skip_scale1= nn.Parameter(torch.ones(hidden_dim//3))

        self.PWConv = nn.Conv2d(in_channels=2*hidden_dim//3, out_channels=2*hidden_dim//3, kernel_size=1, stride=1)
        self.linear_out = nn.Linear(2 * hidden_dim, 2 * hidden_dim // 3)

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 3),
            nn.Conv2d(in_channels=hidden_dim//3,out_channels=hidden_dim//3,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_dim//3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 3, out_channels=hidden_dim // 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 3, out_channels=hidden_dim // 3, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.se_fusion = SEFusion(in_channels = hidden_dim, out_channels=hidden_dim)
        
    def forward(self, input: torch.Tensor, mask=None):
        device = input.device
        # print(f"SF-SSM Block input shape of x: {input.shape}")
        
        if input.dim() == 5:
            B, T, H, W, C = input.shape
            has_temporal = True
        else:
            B, H, W, C = input.shape
            T = 1
            has_temporal = False
        
        input_left, input_middle, input_right = input.chunk(3, dim=-1)
        # print(f"input_left shape: {input_left.shape}")
        # print(f"input_middle shape: {input_middle.shape}")
        # print(f"input_right shape: {input_right.shape}")

        ## Spatial-SSM branch
        # print(f"In Spatial-SSM branch, has_temporal: {has_temporal}")
        if "spatial" not in self.active_branches:
            x_ssm = torch.zeros_like(input_middle)
        elif has_temporal:
            input_middle_4d = input_middle.reshape(B*T, H, W, -1)
            x_ssm = self.drop_path(self.self_attention_ssm(self.ln_1(input_middle_4d)))
            x_ssm = x_ssm.reshape(B, T, H, W, -1)
            # print(f"DEBUG: shape of x_ssm: {x_ssm.shape}")
        else:
            x_ssm = self.drop_path(self.self_attention_ssm(self.ln_1(input_middle)))

        ## Frequency-SSM branch
        # print(f"In Frequency-SSM branch, has_temporal: {has_temporal}")
        if "frequency" not in self.active_branches:
            x_dwt = torch.zeros_like(input_right)
        elif has_temporal:
            # print(f"DEBUG: input_right shape: {input_right.shape}")
            # print(f"DEBUG: B={B}, T={T}, H={H}, W={W}")
            if mask is not None:
                # print(f"DEBUG: mask shape: {mask.shape}")
                max_valid_len = mask.sum(dim=1).max().item()  # The maximum effective length in the batch
                
                h00_list = []
                for t in range(T):
                    if mask[:, t].any():  # At least one sample is valid at this frame.
                        prepare_t = rearrange(input_right[:, t], "b h w c -> b c h w").contiguous()
                        # print(f"DEBUG t={t}: prepare_t shape: {prepare_t.shape}")
                        h00_t = self._process_single_frame_dwt(prepare_t, device)
                        # print(f"DEBUG t={t}: h00_t shape: {h00_t.shape}")
                    else:
                        # For frames with complete padding, directly generate zero tensors and skip the DWT calculation.
                        h00_t = torch.zeros(B, input_right.shape[-1], H, W, dtype=torch.float, device=device)
                    h00_list.append(h00_t)
                
                h00_stack = torch.stack(h00_list, dim=1)
                # print(f"DEBUG: h00_stack shape after stack: {h00_stack.shape}")
                h00_stack = rearrange(h00_stack, "b t c h w -> b c t h w")
                # print(f"DEBUG: h00_stack shape after rearrange1: {h00_stack.shape}")
                
                h00_for_attn = rearrange(h00_stack, "b c t h w -> b t h w c")
                # print(f"DEBUG: h00_for_attn shape: {h00_for_attn.shape}")
                
                # Apply the mask to ensure that the padding frame features are zero.
                mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
                # print(f"DEBUG: mask_expanded shape: {mask_expanded.shape}")
                h00_for_attn = h00_for_attn * mask_expanded
                
                h11_processed = self.drop_path1(self.self_attention_dwt(self.ln_1(h00_for_attn)))
                # h11_processed shape: (B, T, H, W, C)
                
                # Perform IDWT for each time step separately
                x_dwt_list = []
                for t in range(T):
                    h11_t = rearrange(h11_processed[:, t], "b h w c -> b c h w")
                    recons2_t = self._idwt_from_h11(h11_t, device)
                    recons2_t = rearrange(recons2_t, "b c h w -> b h w c")
                    x_dwt_list.append(recons2_t)
                
                x_dwt = torch.stack(x_dwt_list, dim=1)  # (B, T, H, W, C)
                # print(f"DEBUG: x_dwt shape before mask: {x_dwt.shape}")
                # print(f"DEBUG: mask shape for x_dwt: {mask.shape}")
                
                # Apply the mask to the output.
                mask_spatial = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
                # print(f"DEBUG: mask_spatial shape: {mask_spatial.shape}")
                x_dwt = x_dwt * mask_spatial
            else:
                # Preserve the original behavior when no mask is provided.
                h00_list = []
                for t in range(T):
                    prepare_t = rearrange(input_right[:, t], "b h w c -> b c h w").contiguous()
                    h00_t = self._process_single_frame_dwt(prepare_t, device)
                    h00_list.append(h00_t)
                
                h00_stack = torch.stack(h00_list, dim=1)
                h00_stack = rearrange(h00_stack, "b t c h w -> b c t h w")
                
                h00_for_attn = rearrange(h00_stack, "b c t h w -> b t h w c")
                h11_processed = self.drop_path1(self.self_attention_dwt(self.ln_1(h00_for_attn)))
                # h11_processed shape: (B, T, H, W, C)
                
                # Apply the IDWT independently at each time step.
                x_dwt_list = []
                for t in range(T):
                    h11_t = rearrange(h11_processed[:, t], "b h w c -> b c h w")
                    recons2_t = self._idwt_from_h11(h11_t, device)
                    recons2_t = rearrange(recons2_t, "b c h w -> b h w c")
                    x_dwt_list.append(recons2_t)
                
                x_dwt = torch.stack(x_dwt_list, dim=1)  # (B, T, H, W, C)
            
        else:
            prepare = rearrange(input_right, "b h w c -> b c h w").contiguous()

            if self.wave == 'haar' and self.J == 2:
                xfm = DWTForward(J=2, mode='zero', wave='haar').to(device)
                ifm = DWTInverse(mode='zero', wave='haar').to(device)

                Yl, Yh = xfm(prepare)
                h00 = torch.zeros_like(prepare, dtype=torch.float, device=device)
                
                for i in range(len(Yh)):
                  if i == len(Yh) - 1:
                    h00[:, :, :Yl.size(2), :Yl.size(3)] = Yl
                    h00[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 0, :, :]
                    h00[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] = Yh[i][:, :, 1, :, :]
                    h00[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 2, :, :]
                  else:
                    h00[:, :, :Yh[i].size(3), Yh[i].size(4):] = Yh[i][:, :, 0, :, :h00.shape[3] - Yh[i].size(4)]
                    h00[:, :, Yh[i].size(3):, :Yh[i].size(4)] = Yh[i][:, :, 1, :h00.shape[2] - Yh[i].size(3), :]
                    h00[:, :, Yh[i].size(3):, Yh[i].size(4):] = Yh[i][:, :, 2, :h00.shape[2] - Yh[i].size(3), :h00.shape[3] - Yh[i].size(4)]

                h00 = rearrange(h00, "b c h w -> b h w c").contiguous()
                h11 = self.ln_1(h00)
                h11 = h00*self.skip_scale1 + self.drop_path1(self.self_attention_dwt(h11))
                h11 = rearrange(h11, "b h w c -> b c h w").contiguous()

                for i in range(len(Yh)):
                  if i == len(Yh) - 1:
                    Yl = h11[:, :, :Yl.size(2), :Yl.size(3)] 
                    Yh[i][:, :, 0, :, :] = h11[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] 
                    Yh[i][:, :, 1, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] 
                    Yh[i][:, :, 2, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] 
                  else:
                    Yh[i][:, :, 0, :, :h11.shape[3] - Yh[i].size(4)] = h11[:, :, :Yh[i].size(3), Yh[i].size(4):] 
                    Yh[i][:, :, 1, :h11.shape[2] - Yh[i].size(3), :] = h11[:, :, Yh[i].size(3):, :Yh[i].size(4)] 
                    Yh[i][:, :, 2, :h11.shape[2] - Yh[i].size(3), :h11.shape[3] - Yh[i].size(4)] = h11[:, :, Yh[i].size(3):, Yh[i].size(4):]

                temp = ifm((Yl, [Yh[1]]))
                recons2 = ifm((temp, [Yh[0]]))

            elif self.J != 2:
                xfm = DWTForward(J=self.J, mode='zero', wave=self.wave).to(device)
                ifm = DWTInverse(mode='zero', wave=self.wave).to(device)

                Yl, Yh = xfm(prepare)
                B_loc, C_loc, H_prep, W_prep = prepare.shape
                h00 = torch.zeros_like(prepare, dtype=torch.float, device=device)

                yl_h, yl_w = Yl.shape[2], Yl.shape[3]
                total_levels = self.J

                for level_idx in range(total_levels):
                    if level_idx >= len(Yh):
                        continue
                    yh = Yh[level_idx]
                    yh_subbands = [yh[:, :, 0, :, :], yh[:, :, 1, :, :], yh[:, :, 2, :, :]]
                    yh_h, yh_w = yh_subbands[0].shape[2], yh_subbands[0].shape[3]
                    
                    level_offset = total_levels - level_idx - 1
                    offset_h = yl_h * (2 ** level_offset)
                    offset_w = yl_w * (2 ** level_offset)
                    
                    valid_h = min(yh_h, h00.shape[2] - offset_h)
                    valid_w = min(yh_w, h00.shape[3] - offset_w)
                    
                    if level_idx == total_levels - 1:
                        h00[:, :, :yl_h, :yl_w] = Yl[:, :, :yl_h, :yl_w]
                        h00[:, :, :valid_h, offset_w:offset_w+valid_w] = yh_subbands[0][:, :, :valid_h, :valid_w]
                        h00[:, :, offset_h:offset_h+valid_h, :valid_w] = yh_subbands[1][:, :, :valid_h, :valid_w]
                        h00[:, :, offset_h:offset_h+valid_h, offset_w:offset_w+valid_w] = yh_subbands[2][:, :, :valid_h, :valid_w]
                    else:
                        h00[:, :, :valid_h, offset_w:offset_w+valid_w] = yh_subbands[0][:, :, :valid_h, :valid_w]
                        h00[:, :, offset_h:offset_h+valid_h, :valid_w] = yh_subbands[1][:, :, :valid_h, :valid_w]
                        h00[:, :, offset_h:offset_h+valid_h, offset_w:offset_w+valid_w] = yh_subbands[2][:, :, :valid_h, :valid_w]

                h00 = rearrange(h00, "b c h w -> b h w c").contiguous()
                h11 = self.ln_1(h00)
                h11 = h00 * self.skip_scale1 + self.drop_path1(self.self_attention_dwt(h11))
                h11 = rearrange(h11, "b h w c -> b c h w").contiguous()

                yl_h, yl_w = Yl.shape[2], Yl.shape[3]
                total_levels = self.J

                for level_idx in range(total_levels):
                    if level_idx >= len(Yh):
                        continue
                    yh = Yh[level_idx]
                    yh_subbands = [yh[:, :, 0, :, :].clone(), yh[:, :, 1, :, :].clone(), yh[:, :, 2, :, :].clone()]
                    yh_h, yh_w = yh_subbands[0].shape[2], yh_subbands[0].shape[3]
                    
                    level_offset = total_levels - level_idx - 1
                    offset_h = yl_h * (2 ** level_offset)
                    offset_w = yl_w * (2 ** level_offset)
                    
                    valid_h = min(yh_h, h11.shape[2] - offset_h)
                    valid_w = min(yh_w, h11.shape[3] - offset_w)
                    
                    if level_idx == total_levels - 1:
                        Yl = h11[:, :, :yl_h, :yl_w].clone()
                        yh_subbands[0][:, :, :valid_h, :valid_w] = h11[:, :, :valid_h, offset_w:offset_w+valid_w]
                        yh_subbands[1][:, :, :valid_h, :valid_w] = h11[:, :, offset_h:offset_h+valid_h, :valid_w]
                        yh_subbands[2][:, :, :valid_h, :valid_w] = h11[:, :, offset_h:offset_h+valid_h, offset_w:offset_w+valid_w]
                    else:
                        yh_subbands[0][:, :, :valid_h, :valid_w] = h11[:, :, :valid_h, offset_w:offset_w+valid_w]
                        yh_subbands[1][:, :, :valid_h, :valid_w] = h11[:, :, offset_h:offset_h+valid_h, :valid_w]
                        yh_subbands[2][:, :, :valid_h, :valid_w] = h11[:, :, offset_h:offset_h+valid_h, offset_w:offset_w+valid_w]
                    
                    Yh[level_idx][:, :, 0, :, :] = yh_subbands[0]
                    Yh[level_idx][:, :, 1, :, :] = yh_subbands[1]
                    Yh[level_idx][:, :, 2, :, :] = yh_subbands[2]

                current_yl = Yl.to(device)
                for level_idx in reversed(range(total_levels)):
                    if level_idx >= len(Yh):
                        continue
                    current_yl = ifm((current_yl, [Yh[level_idx]]))
                recons2 = current_yl.to(device)

            elif self.wave != 'haar':
                haar_target_sizes = {
                    32: {"yl": (8, 8), "yh0": (16, 16), "yh1": (8, 8)},
                    64: {"yl": (4, 4), "yh0": (8, 8), "yh1": (4, 4)},
                    128: {"yl": (2, 2), "yh0": (4, 4), "yh1": (2, 2)},
                    256: {"yl": (1, 1), "yh0": (2, 2), "yh1": (1, 1)},
                }

                xfm = DWTForward(J=self.J, mode='zero', wave=self.wave).to(device)
                ifm = DWTInverse(mode='zero', wave=self.wave).to(device)

                Yl, Yh = xfm(prepare)
                B_loc, C_loc, H_prep, W_prep = prepare.shape
                
                current_channel = Yl.shape[1]
                yl_ori_h, yl_ori_w = Yl.shape[2], Yl.shape[3]
                yh0_ori_h, yh0_ori_w = Yh[0].shape[3], Yh[0].shape[4]
                yh1_ori_h, yh1_ori_w = Yh[1].shape[3], Yh[1].shape[4]

                if current_channel not in haar_target_sizes:
                    raise ValueError(
                        f"No matching Haar target size for {current_channel} channels."
                    )
                yl_target_h, yl_target_w = haar_target_sizes[current_channel]["yl"]
                yh0_target_h, yh0_target_w = haar_target_sizes[current_channel]["yh0"]
                yh1_target_h, yh1_target_w = haar_target_sizes[current_channel]["yh1"]

                Yl_compress = Yl[:, :, :yl_target_h, :yl_target_w]
                if yl_ori_h > yl_target_h:
                    Yl_compress[:, :, -1, :] += Yl[:, :, yl_target_h:, :yl_target_w].sum(dim=2)
                if yl_ori_w > yl_target_w:
                    Yl_compress[:, :, :, -1] += Yl[:, :, :yl_target_h, yl_target_w:].sum(dim=3)

                Yh0_compress = Yh[0][:, :, :, :yh0_target_h, :yh0_target_w]
                if yh0_ori_h > yh0_target_h:
                    Yh0_compress[:, :, :, -1, :] += Yh[0][:, :, :, yh0_target_h:, :yh0_target_w].sum(dim=3)
                if yh0_ori_w > yh0_target_w:
                    Yh0_compress[:, :, :, :, -1] += Yh[0][:, :, :, :yh0_target_h, yh0_target_w:].sum(dim=4)

                Yh1_compress = Yh[1][:, :, :, :yh1_target_h, :yh1_target_w]
                if yh1_ori_h > yh1_target_h:
                    Yh1_compress[:, :, :, -1, :] += Yh[1][:, :, :, yh1_target_h:, :yh1_target_w].sum(dim=3)
                if yh1_ori_w > yh1_target_w:
                    Yh1_compress[:, :, :, :, -1] += Yh[1][:, :, :, :yh1_target_h, yh1_target_w:].sum(dim=4)

                Yl = Yl_compress
                Yh[0] = Yh0_compress
                Yh[1] = Yh1_compress

                h00 = torch.zeros_like(prepare, dtype=torch.float, device=device)

                for i in range(len(Yh)):
                    if i == len(Yh) - 1:
                        h00[:, :, :Yl.size(2), :Yl.size(3)] = Yl
                        h00[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 0, :, :]
                        h00[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] = Yh[i][:, :, 1, :, :]
                        h00[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 2, :, :]
                    else:
                        h00[:, :, :Yh[i].size(3), Yh[i].size(4):] = Yh[i][:, :, 0, :, :h00.shape[3] - Yh[i].size(4)]
                        h00[:, :, Yh[i].size(3):, :Yh[i].size(4)] = Yh[i][:, :, 1, :h00.shape[2] - Yh[i].size(3), :]
                        h00[:, :, Yh[i].size(3):, Yh[i].size(4):] = Yh[i][:, :, 2, :h00.shape[2] - Yh[i].size(3), :h00.shape[3] - Yh[i].size(4)]

                h00 = rearrange(h00, "b c h w -> b h w c")
                if not h00.is_contiguous():
                    h00 = h00.contiguous()

                h11 = self.ln_1(h00)
                h11 = h00*self.skip_scale1 + self.drop_path1(self.self_attention_dwt(h11))

                h11 = rearrange(h11, "b h w c -> b c h w")
                if not h11.is_contiguous():
                    h11 = h11.contiguous()

                for i in range(len(Yh)):
                    if i == len(Yh) - 1:
                        Yl = h11[:, :, :Yl.size(2), :Yl.size(3)] 
                        Yh[i][:, :, 0, :, :] = h11[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] 
                        Yh[i][:, :, 1, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] 
                        Yh[i][:, :, 2, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] 
                    else:
                        Yh[i][:, :, 0, :, :h11.shape[3] - Yh[i].size(4)] = h11[:, :, :Yh[i].size(3), Yh[i].size(4):] 
                        Yh[i][:, :, 1, :h11.shape[2] - Yh[i].size(3), :] = h11[:, :, Yh[i].size(3):, :Yh[i].size(4)] 
                        Yh[i][:, :, 2, :h11.shape[2] - Yh[i].size(3), :h11.shape[3] - Yh[i].size(4)] = h11[:, :, Yh[i].size(3):, Yh[i].size(4):]

                Yl_restore = torch.zeros((B_loc, C_loc, yl_ori_h, yl_ori_w), dtype=Yl.dtype, device=device)
                Yl_restore[:, :, :yl_target_h, :yl_target_w] = Yl
                row_diff = yl_ori_h - yl_target_h
                col_diff = yl_ori_w - yl_target_w
                if row_diff > 0:
                    Yl_restore[:, :, yl_target_h:, :yl_target_w] = Yl[:, :, -1:, :] / row_diff
                if col_diff > 0:
                    Yl_restore[:, :, :yl_target_h, yl_target_w:] = Yl[:, :, :, -1:] / col_diff

                Yh0_restore = torch.zeros((B_loc, C_loc, 3, yh0_ori_h, yh0_ori_w), dtype=Yh[0].dtype, device=device)
                Yh0_restore[:, :, :, :yh0_target_h, :yh0_target_w] = Yh[0]
                yh0_row_diff = yh0_ori_h - yh0_target_h
                yh0_col_diff = yh0_ori_w - yh0_target_w
                if yh0_row_diff > 0:
                    Yh0_restore[:, :, :, yh0_target_h:, :yh0_target_w] = Yh[0][:, :, :, -1:, :] / yh0_row_diff
                if yh0_col_diff > 0:
                    Yh0_restore[:, :, :, :yh0_target_h, yh0_target_w:] = Yh[0][:, :, :, :, -1:] / yh0_col_diff

                Yh1_restore = torch.zeros((B_loc, C_loc, 3, yh1_ori_h, yh1_ori_w), dtype=Yh[1].dtype, device=device)
                Yh1_restore[:, :, :, :yh1_target_h, :yh1_target_w] = Yh[1]
                yh1_row_diff = yh1_ori_h - yh1_target_h
                yh1_col_diff = yh1_ori_w - yh1_target_w
                if yh1_row_diff > 0:
                    Yh1_restore[:, :, :, yh1_target_h:, :yh1_target_w] = Yh[1][:, :, :, -1:, :] / yh1_row_diff
                if yh1_col_diff > 0:
                    Yh1_restore[:, :, :, :yh1_target_h, yh1_target_w:] = Yh[1][:, :, :, :, -1:] / yh1_col_diff

                Yl = Yl_restore
                Yh[0] = Yh0_restore
                Yh[1] = Yh1_restore

                temp = ifm((Yl, [Yh[1]]))
                recons2 = ifm((temp, [Yh[0]]))

            recons2 = rearrange(recons2, "b c h w -> b h w c").contiguous()
            x_dwt = recons2

        ## concat
        # print(f"In Local-Conv branch, has_temporal: {has_temporal}")
        if has_temporal:
            x = torch.cat((x_ssm, x_dwt), dim=-1)
            x_final = x
            # print(f"output of SSM and DWT branches before fusion shape: {x_final.shape}")
            
            # Local-Conv Branch
            input_left_4d = input_left.reshape(B*T, H, W, -1)
            input_left_4d = input_left_4d.permute(0,3,1,2).contiguous()
            input_left_4d = self.conv33conv33conv11(input_left_4d)
            input_left_4d = input_left_4d.permute(0,2,3,1).contiguous()
            input_left = input_left_4d.reshape(B, T, H, W, -1)
            if "local" not in self.active_branches:
                input_left = torch.zeros_like(input_left)
            # print(f"output of Local-Conv branch shape: {input_left.shape}")
            
            output = torch.cat((input_left, x_final), dim=-1)
            output = channel_shuffle(output, groups=2)
            # print(f"output of after fusion Local-Conv branch shape: {output.shape}")
            
            # Apply the mask to the final output as well.
            if mask is not None:
                mask_full = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
                # print(f"DEBUG: mask_full shape for final output: {mask_full.shape}")
                output = output * mask_full
            
            return output + input
        else:
            x = torch.cat((x_ssm, x_dwt), dim=-1)
            x_final = x
            # print(f"output of SSM and DWT branches before fusion shape: {x_final.shape}")
            
            input_left = input_left.permute(0,3,1,2).contiguous()
            input_left = self.conv33conv33conv11(input_left)
            input_left = input_left.permute(0,2,3,1).contiguous()
            if "local" not in self.active_branches:
                input_left = torch.zeros_like(input_left)
            # print(f"output of Local-Conv branch shape: {input_left.shape}")
            output = torch.cat((input_left,x_final),dim=-1)
            output = channel_shuffle(output,groups=2)
            # print(f"output of after fusion Local-Conv branch shape: {output.shape}")

            return output+input

    def _process_single_frame_dwt(self, prepare: torch.Tensor, device):
        if self.wave == 'haar' and self.J == 2:
            xfm = DWTForward(J=2, mode='zero', wave='haar').to(device)
            Yl, Yh = xfm(prepare)
            
            h00 = torch.zeros_like(prepare, dtype=torch.float, device=device)
            
            for i in range(len(Yh)):
                if i == len(Yh) - 1:
                    h00[:, :, :Yl.size(2), :Yl.size(3)] = Yl
                    h00[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 0, :, :]
                    h00[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] = Yh[i][:, :, 1, :, :]
                    h00[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 2, :, :]
                else:
                    h00[:, :, :Yh[i].size(3), Yh[i].size(4):] = Yh[i][:, :, 0, :, :h00.shape[3] - Yh[i].size(4)]
                    h00[:, :, Yh[i].size(3):, :Yh[i].size(4)] = Yh[i][:, :, 1, :h00.shape[2] - Yh[i].size(3), :]
                    h00[:, :, Yh[i].size(3):, Yh[i].size(4):] = Yh[i][:, :, 2, :h00.shape[2] - Yh[i].size(3), :h00.shape[3] - Yh[i].size(4)]
            
            return h00
        else:
            xfm = DWTForward(J=self.J, mode='zero', wave=self.wave).to(device)
            Yl, Yh = xfm(prepare)
            B_loc, C_loc, H_prep, W_prep = prepare.shape
            h00 = torch.zeros_like(prepare, dtype=torch.float, device=device)

            yl_h, yl_w = Yl.shape[2], Yl.shape[3]
            total_levels = self.J

            for level_idx in range(total_levels):
                if level_idx >= len(Yh):
                    continue
                yh = Yh[level_idx]
                yh_subbands = [yh[:, :, 0, :, :], yh[:, :, 1, :, :], yh[:, :, 2, :, :]]
                yh_h, yh_w = yh_subbands[0].shape[2], yh_subbands[0].shape[3]
                
                level_offset = total_levels - level_idx - 1
                offset_h = yl_h * (2 ** level_offset)
                offset_w = yl_w * (2 ** level_offset)
                
                valid_h = min(yh_h, h00.shape[2] - offset_h)
                valid_w = min(yh_w, h00.shape[3] - offset_w)
                
                if level_idx == total_levels - 1:
                    h00[:, :, :yl_h, :yl_w] = Yl[:, :, :yl_h, :yl_w]
                    h00[:, :, :valid_h, offset_w:offset_w+valid_w] = yh_subbands[0][:, :, :valid_h, :valid_w]
                    h00[:, :, offset_h:offset_h+valid_h, :valid_w] = yh_subbands[1][:, :, :valid_h, :valid_w]
                    h00[:, :, offset_h:offset_h+valid_h, offset_w:offset_w+valid_w] = yh_subbands[2][:, :, :valid_h, :valid_w]
                else:
                    h00[:, :, :valid_h, offset_w:offset_w+valid_w] = yh_subbands[0][:, :, :valid_h, :valid_w]
                    h00[:, :, offset_h:offset_h+valid_h, :valid_w] = yh_subbands[1][:, :, :valid_h, :valid_w]
                    h00[:, :, offset_h:offset_h+valid_h, offset_w:offset_w+valid_w] = yh_subbands[2][:, :, :valid_h, :valid_w]
            
            return h00

    def _idwt_from_h11(self, h11: torch.Tensor, device):
        if self.wave == 'haar' and self.J == 2:
            ifm = DWTInverse(mode='zero', wave='haar').to(device)
            B, C, H, W = h11.shape
            
            # Extract level-2 frequency bands from h11 (upper-left 16x16 region).
            yl_h, yl_w = H // 4, W // 4  # 8, 8
            Yl = h11[:, :, :yl_h, :yl_w]  # 8x8
            
            # Level-2 high-frequency subbands (also 8x8).
            Yh1 = torch.zeros(B, C, 3, yl_h, yl_w, device=device)
            Yh1[:, :, 0, :, :] = h11[:, :, :yl_h, yl_w:yl_w*2]        # [:8, 8:16]
            Yh1[:, :, 1, :, :] = h11[:, :, yl_h:yl_h*2, :yl_w]        # [8:16, :8]
            Yh1[:, :, 2, :, :] = h11[:, :, yl_h:yl_h*2, yl_w:yl_w*2]  # [8:16, 8:16]
            
            # First IDWT: level 2 to level 1 (8x8 to 16x16).
            temp = ifm((Yl, [Yh1]))  # temp shape: (B, C, 16, 16)
            
            # Extract level-1 frequency bands from h11 (three 16x16 regions).
            Yh0 = torch.zeros(B, C, 3, yl_h*2, yl_w*2, device=device)  # (B, C, 3, 16, 16)
            Yh0[:, :, 0, :, :] = h11[:, :, :yl_h*2, yl_w*2:]           # [:16, 16:] - LH
            Yh0[:, :, 1, :, :] = h11[:, :, yl_h*2:, :yl_w*2]           # [16:, :16] - HL
            Yh0[:, :, 2, :, :] = h11[:, :, yl_h*2:, yl_w*2:]           # [16:, 16:] - HH (bottom-right)
            
            # Second IDWT: level 1 to the original size (16x16 to 32x32).
            recons2 = ifm((temp, [Yh0]))
            return recons2
        else:
            return h11



class VSSLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        wave = 'haar',
        J = 2, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        use_temporal=False,
        active_branches=("spatial", "frequency", "local"),
        **kwargs,
    ):
        super().__init__()
        self.wave = wave
        self.J = J
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.active_branches = active_branches

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                wave = wave,
                J = J, 
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_temporal=use_temporal,
                active_branches=active_branches,
            )
            for i in range(depth)])
        
        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, mask)
            else:
                x = blk(x, mask=mask)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x



class VSSM(nn.Module):
    def __init__(self, wave='haar', J=2, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96,192,384,768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, use_temporal=False, active_branches=("spatial", "frequency", "local"), **kwargs):
        super().__init__()
        self.wave = wave
        self.J = J
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.use_temporal = use_temporal
        self.active_branches = active_branches

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                wave = self.wave, 
                J = self.J, 
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_temporal=use_temporal,
                active_branches=active_branches,
            )
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward_backbone(self, x, mask=None):
        if x.dim() == 5:
            B, T, H, W, C = x.shape
            # print(f"Input shape of x:{x.shape}") # B, T, H, W, C
            has_temporal = True
        else:
            has_temporal = False
            B, H, W, C = x.shape
            T = 1
        
        # 
        if has_temporal:
            x_4d = x.reshape(B*T, H, W, C)
        else:
            x_4d = x
        x_4d = x_4d.permute(0, 3, 1, 2)
        # print(f"In forward backbone shape of x_4d:{x_4d.shape}") # B*T, C, H, W
        
        x_4d = self.patch_embed(x_4d)
        # print(f"After patch_embed shape of x_4d:{x_4d.shape}") # B*T, H/4, W/4, 96
        
        if has_temporal:
            _, H_new, W_new, C_new = x_4d.shape
            x = x_4d.reshape(B, T, H_new, W_new, C_new)
            # print(f"After reshape to 5D shape of x:{x.shape}") # B, T, H/4, W/4, 96
        else:
            x = x_4d
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, mask=mask)
            # print(f"After layer shape of x:{x.shape}")
        
        # print(f"After all layers has_temporal:{has_temporal}")
        
        return x


    def forward(self, x, mask=None):
        x = self.forward_backbone(x, mask=mask)
        # print(f"After forward_backbone shape of x: {x.shape}")
        has_temporal = False
        B, T = 1, 1
        
        if x.dim() == 5:
            B, T, H, W, C = x.shape
            has_temporal = True
            # Merge B and T before applying average pooling.
            x = x.reshape(B*T, H, W, C)
        
        x = x.permute(0, 3, 1, 2)       # [B*T, C, H, W]
        x = self.avgpool(x)             # [B*T, C, 1, 1]
        x = torch.flatten(x, start_dim=1) # [B*T, C]
        
        x = self.head(x)                # [B*T, num_classes]
        # print(f"After head shape of x: {x.shape}") # [B*T, num_classes]
        
        # print(f"before reshape back has_temporal: {has_temporal}")
        if has_temporal:
            # Restore the shape to [B, T, num_classes].
            x = x.view(B, T, -1)
            # print(f"After reshape back shape of x: {x.shape}") # [B, T, num_classes]
        
        return x


class SEFusion(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=16):

        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, f):

        b, c, _, _ = f.shape
        w = f.mean((2,3))
        w = self.attn(w).view(b, c, 1, 1)
        f = f * w
        f = self.proj(f)
        return f


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    if x.dim() == 5:
        # 5D input: (B, T, H, W, C)
        batch_size, time_steps, height, width, num_channels = x.size()
        channels_per_group = num_channels // groups
        
        # reshape
        x = x.view(batch_size, time_steps, height, width, groups, channels_per_group)
        
        x = torch.transpose(x, 4, 5).contiguous()
        
        # flatten
        x = x.view(batch_size, time_steps, height, width, -1)
    else:
        # 4D input: (B, H, W, C)
        batch_size, height, width, num_channels = x.size()
        channels_per_group = num_channels // groups
        
        # reshape
        x = x.view(batch_size, height, width, groups, channels_per_group)
        
        x = torch.transpose(x, 3, 4).contiguous()
        
        # flatten
        x = x.view(batch_size, height, width, -1)
    
    return x



class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        wave = 'haar',
        J = 2, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.wave = wave
        self.J = J
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                wave = wave,
                J = J, 
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    

