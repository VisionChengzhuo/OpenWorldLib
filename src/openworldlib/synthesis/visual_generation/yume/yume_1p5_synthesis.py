import os
import random
import sys
from typing import Optional, Union, Tuple
import torch
import numpy as np
from diffusers.video_processor import VideoProcessor

from ...base_synthesis import BaseSynthesis

# for Yume1p5TI2V
import logging
from functools import partial
import math
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from safetensors.torch import load_file
from ....base_models.diffusion_model.video.wan_2p2.distributed.fsdp import shard_model
from ....base_models.diffusion_model.video.wan_2p1.modules.t5 import T5EncoderModel
from ....base_models.diffusion_model.video.wan_2p2.modules.vae2_2 import Wan2_2_VAE
from ....base_models.diffusion_model.video.wan_2p2.modules.model import WanModel, Head, rope_params, sinusoidal_embedding_1d, WanSelfAttention, WanCrossAttention, WanLayerNorm
from ....base_models.diffusion_model.video.wan_2p1.modules.attention import flash_attention
from ....base_models.diffusion_model.video.wan_2p2.utils.utils import best_output_size
from ....base_models.diffusion_model.video.wan_2p2.configs import (
    WAN_CONFIGS,
    SIZE_CONFIGS,
    MAX_AREA_CONFIGS
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.cuda import amp


def masks_like(tensor, zero=False, generator=None, p=0.2, current_latent_num=8):
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, :-current_latent_num] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, :-current_latent_num]).exp()
                    v[:, :-current_latent_num] = torch.zeros_like(v[:, :-current_latent_num])
                else:
                    u[:, :-current_latent_num] = u[:, :-current_latent_num]
                    v[:, :-current_latent_num] = v[:, :-current_latent_num]
        else:
            for u, v in zip(out1, out2):
                u[:, :-current_latent_num] = torch.zeros_like(u[:, :-current_latent_num])
                v[:, :-current_latent_num] = torch.zeros_like(v[:, :-current_latent_num])

    return out1, out2

@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, b_, s_, n_, d_, mask_token=None, ids_restore=None, ids_keep=None, rand_num_img=None, flag=None):
    if not flag: #False:
        if ids_restore!=None:
            x = x.view(b_,s_,n_*d_)

            mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            x = x.view(b_,ids_restore.shape[1],n_,d_)

        n, c = x.size(2), x.size(3) // 2
        
        # split freqs
        freqs = freqs.squeeze().split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        
        if ids_keep!=None:
            output = torch.stack(output).float()
            output = output.view(b_,ids_restore.shape[1],n_*d_)
            output = torch.gather(
                output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, n_*d_))
            output = output.view(b_,s_,n_,d_)
            return output

        return torch.stack(output).float()
    else:
        if ids_restore!=None:
            x = x.view(b_,s_,n_*d_)

            mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            x = x.view(b_,ids_restore.shape[1],n_,d_)

        n, c = x.size(2), x.size(3) // 2

        # loop over samples
        output = []
        seq_len = x.shape[1]
        x_i = torch.view_as_complex(x[0, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = freqs
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[0, seq_len:]])
        output.append(x_i)


        if ids_keep!=None:
            output = torch.stack(output).float()
            output = output.view(b_,ids_restore.shape[1],n_*d_)
            output = torch.gather(
                output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, n_*d_))
            output = output.view(b_,s_,n_,d_)
            return output

        return torch.stack(output).float()


def upsample_conv3d_weights_auto(conv_small: nn.Conv3d, size: Tuple[int,int,int]):
    # small: (OC_small, IC_small, kT, kH, kW)
    OC, IC, _, _, _ = conv_small.weight.shape
    with torch.no_grad():
        w = F.interpolate(conv_small.weight.data,
                          size=size, mode='trilinear', align_corners=False)
        big = nn.Conv3d(in_channels=IC, out_channels=OC,
                        kernel_size=size, stride=size, padding=0)
        big.weight.copy_(w)
        if conv_small.bias is not None:
            big.bias = nn.Parameter(conv_small.bias.data.clone())
        else:
            big.bias = None
    return big


def convpadd(tensor,pad_num):
    dim = tensor.dim()
    if dim==4:
        tensor = tensor.unsqueeze(2)
    if dim==6:
        tensor = tensor.squeeze(2)
        
    b,c,f,h,w = tensor.shape

    pad_h = (pad_num - h % pad_num) % pad_num  
    pad_w = (pad_num - w % pad_num) % pad_num  
    tensor = torch.cat([tensor,torch.zeros(b,c,f,pad_h,w).to(tensor.device)],dim=3)
    tensor = torch.cat([tensor,torch.zeros(b,c,f,h+pad_h,pad_w).to(tensor.device)],dim=4)
    return tensor

def up_fre(f_1,f_2,f_3,u,f_z,scale=False):
    b1, c1, f1, h1, w1 = u.shape
    freqs_i = torch.cat([
        f_1[f_z:f_z+f1].view(f1, 1, 1, -1).expand(f1, h1, w1, -1),
        f_2[:h1].view(1, h1, 1, -1).expand(f1, h1, w1, -1),
        f_3[:w1].view(1, 1, w1, -1).expand(f1, h1, w1, -1)],
    dim=-1).reshape(f1*h1*w1, 1, -1)
    return freqs_i


class Yume1p5WanSelfAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):

        super().__init__(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps
        )

    def forward(self, x, seq_lens, grid_sizes, freqs, mask_token, ids_restore, ids_keep, flag):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs, b, s, n, d, mask_token, ids_restore, ids_keep, flag = flag),
            k=rope_apply(k, grid_sizes, freqs, b, s, n, d, mask_token, ids_restore, ids_keep, flag = flag),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class Yume1p5WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = Yume1p5WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps) # special to Yume 1.5
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        # specific to Yume 1.5
        ids_keep=None,
        ids_restore=None,
        mask_token=None,
        flag=True,
    ):
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, ids_restore=ids_restore, ids_keep=ids_keep, mask_token=mask_token,flag=flag)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Yume1p5WanModel(WanModel, ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):

        ModelMixin.__init__(self)
        ConfigMixin.__init__(self)

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            Yume1p5WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)
        self.d = d
        
        self.mask_ratio = 0.3
        self.mask_token = None

        self.patch_embedding_2x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,4,4))
        self.patch_embedding_4x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,8,8))
        self.patch_embedding_8x  = upsample_conv3d_weights_auto(self.patch_embedding, (1,16,16))
        self.patch_embedding_16x = upsample_conv3d_weights_auto(self.patch_embedding, (1,32,32))
        self.patch_embedding_2x_f = nn.Conv3d(
            self.patch_embedding.in_channels,
            self.patch_embedding.in_channels,
            kernel_size=(1,4,4), stride=(1,4,4),
        )

        # initialize weights
        self.init_weights()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_unkeep = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def forward_side_interpolater(self, x, mask, ids_restore, kwagrs):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_before = x
        x = self.sideblock(x, **kwagrs)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x
    
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        enable_mask=False,
        y=None,
        current_latent_num=8,
        input_ids=None,
        flag = True
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        #print(x[0].shape,"nscdbf09-vmifjw0")
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        if flag:
            x_pack = []
            # According to Eq. 3, compress historical frames at different ratios.
            for u in x:
                u = u.unsqueeze(0)
                f_num = u.shape[2]
                u1 = u[:,:,:-current_latent_num]
                u2 = u[:,:,-current_latent_num:]
                f_1 = rope_params(1024, self.d - 4 * (self.d // 6)).to(u.device)
                f_2 = rope_params(1024, 2 * (self.d // 6)).to(u.device)
                f_3 = rope_params(1024, 2 * (self.d // 6)).to(u.device)
                if f_num - current_latent_num <= 2 + 4:
                    f_zero = u1.shape[2]

                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))

                    if f_zero - 2 <= 0:
                        u_2 = self.patch_embedding_2x(convpadd(u1[:,:,-1].unsqueeze(2),4))
                    else:
                        u_2 = self.patch_embedding_2x(convpadd(u1[:,:,1:-1],4))

                    u_3 = self.patch_embedding(u1[:,:,-1].unsqueeze(2))
                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    # Generate the corresponding RoPE encoding based on the compressed historical frames.
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2)],dim=0)
                    f_z = f1+f2+u_3.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - current_latent_num <= 2 + 4 + 16:
                    
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))
                    
                    if f_zero-6<=0:
                        u_2 = self.patch_embedding_4x(convpadd(u1[:,:,-5].unsqueeze(2),8))
                    else:
                        u_2 = self.patch_embedding_4x(convpadd(u1[:,:,1:-5],8))


                    u_3 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_4 = self.patch_embedding(u1[:,:,-3:])
                    
                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3)],dim=0)
                    f_z = f1+f2+f3+u_4.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - current_latent_num <= 2 + 4 + 16 + 64:
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))
                    

                    if f_zero-22<=0:
                        u_2 = self.patch_embedding_8x(convpadd(u1[:,:,-21].unsqueeze(2),16))
                    else:
                        u_2 = self.patch_embedding_8x(convpadd(u1[:,:,1:-21],16))

                    u_3 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_4 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4)) 
                    u_5 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4)],dim=0)
                    f_z = f1+f2+f3+f4+u_5.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2),u_5.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - current_latent_num <= 2 + 4 + 16 + 64 + 256: 
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding_2x(convpadd(u1[:,:,0].unsqueeze(2),4))

                    if f_zero-86<=0:
                        u_2 = self.patch_embedding_16x(convpadd(u1[:,:,-85].unsqueeze(2),32))
                    else:
                        u_2 = self.patch_embedding_16x(convpadd(u1[:,:,1:-85],32))


                    u_3 = self.patch_embedding_8x(convpadd(u1[:,:,-85:-21],16))
                    u_4 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_5 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_6 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    f5 = u_5.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4,True),up_fre(f_1,f_2,f_3,u_6,f1+f2+f3+f4+f5)],dim=0)
                    f_z = f1+f2+f3+f4+f5+u_6.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2), \
                                    u_5.flatten(2).transpose(1, 2),u_6.flatten(2).transpose(1, 2)],dim=1)

                elif f_num - current_latent_num <= 2 + 4 + 16 + 64 + 256 + 1024:
                    f_zero = u1.shape[2]

                    u_1 = self.patch_embedding_2x(convpadd(u1[:,:,0].unsqueeze(2),4))

                    if f_zero - 342 <= 0:
                        u_2 = self.patch_embedding_16x(convpadd(self.patch_embedding_2x_f(convpadd(u1[:,:,-341].unsqueeze(2),4)), 32) )
                    else:
                        u_2 = self.patch_embedding_16x(convpadd(self.patch_embedding_2x_f(convpadd(u1[:,:,1:-341],4)), 32) )


                    u_3 = self.patch_embedding_16x(convpadd(u1[:,:,-341:-85],32) )
                    u_4 = self.patch_embedding_8x(convpadd(u1[:,:,-85:-21],16))
                    u_5 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_6 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_7 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    f5 = u_5.shape[2]
                    f6 = u_6.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4,True),up_fre(f_1,f_2,f_3,u_6,f1+f2+f3+f4+f5,True),\
                                            up_fre(f_1,f_2,f_3,u_7,f1+f2+f3+f4+f5+f6)],dim=0)
                    f_z = f1+f2+f3+f4+f5+f6+u_7.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2), \
                                    u_5.flatten(2).transpose(1, 2),u_6.flatten(2).transpose(1, 2),u_7.flatten(2).transpose(1, 2)],dim=1)

                u2 = self.patch_embedding(u2)
                freqs_i = torch.cat([freqs_i, up_fre(f_1,f_2,f_3,u2,f_z)],dim=0)
                seq_lens1 = u1.shape[1]
                grid_sizes = torch.stack([torch.tensor(u2.shape[2:], dtype=torch.long)])
                u2 = u2.flatten(2).transpose(1, 2)
                seq_lens = torch.tensor([u1.shape[1]+u2.shape[1]], dtype=torch.long)

                seq_len = int(seq_lens[0])
                u = torch.cat([u1,u2],dim=1)
                x_pack.append(u)
                t = t.squeeze()
                u1_shape = u1.shape[1]
                u2_shape = u2.shape[1]
                t = torch.cat([
                            t[0:1].new_ones(u1.shape[1]) * t[0],
                            t[-1:].new_ones(u2.shape[1]) * t[-1]
                        ])
                t = t.unsqueeze(0)
    
            self.freqs = freqs_i
            x = x_pack
            x = torch.cat(x)
        else:
            self.freqs = torch.cat([
                    rope_params(1024, self.d - 4 * (self.d // 6)),
                    rope_params(1024, 2 * (self.d // 6)),
                    rope_params(1024, 2 * (self.d // 6))
                ],
                                       dim=1).to(device)    
            # embeddings
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                          dim=1) for u in x
            ])
                
        # We referenced the implementation at https://github.com/sail-sg/MDT
        ids_keep = None
        ids_restore = None
        if self.mask_ratio is not None and enable_mask:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            x_ori = x
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True
            seq_lens = torch.tensor([x.shape[1]], dtype=torch.long)
            seq_len_ori = seq_len
            seq_len = int(seq_lens.item()) 
            
            t_masked = torch.gather(
                t, dim=1, index=ids_keep)
            t_ori = t
            
            
        if self.mask_ratio is not None and enable_mask:
            t = t_masked
        
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
            seq_len = seq_len_ori
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t_ori.size(0)
                t = t_ori.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0_ori = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            # time embeddings
            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        
        len_blocks = (len(self.blocks)+1)//2
        
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            ids_keep=ids_keep,
            ids_restore=ids_restore,
            mask_token=self.mask_token,
            flag=flag)


        cnt_blocks = 0
        for block in self.blocks:
            cnt_blocks += 1
            if cnt_blocks==len_blocks and enable_mask:
                kwargs["ids_keep"]=None
                kwargs["ids_restore"]=None
                kwargs["mask_token"]=None
                kwargs["seq_lens"] = torch.tensor([x.shape[1]], dtype=torch.long)
                kwargs["e"] = e0_ori
                x = self.forward_side_interpolater(x, mask, ids_restore, kwargs)
                x = block(x, **kwargs)
            else:
                kwargs["seq_lens"] = torch.tensor([x.shape[1]], dtype=torch.long)
                x = block(x, **kwargs)


        # head
        x = self.head(x, e)
        
        if flag:
            # unpatchify
            x = self.unpatchify(x[:,seq_lens1:,:], grid_sizes)
        else:
            # unpatchify
            x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x]



def upsample_conv3d_weights(conv_small, size):
    old_weight = conv_small.weight.data 
    new_weight = F.interpolate(
        old_weight,                      
        size=size,              
        mode='trilinear',             
        align_corners=False           
    )
    conv_large = nn.Conv3d(
        in_channels=16,
        out_channels=5120,
        kernel_size=size,
        stride=size,
        padding=0
    )
    conv_large.weight.data = new_weight
    if conv_small.bias is not None:
        conv_large.bias.data = conv_small.bias.data.clone()
    return conv_large


class Yume1p5TI2V():  # adapted from wan2p2.WanTI2V

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        
        
        self.sample_neg_prompt = config.sample_neg_prompt

        # Specific to Yume 1.5
        self.sp_size = 1

        logging.info(f"Creating Yume1p5Model from {checkpoint_dir}")

        config_wan = {
            "_class_name": "WanModel",
            "_diffusers_version": "0.33.0",
            "dim": 3072,
            "eps": 1e-06,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "in_dim": 48,
            "model_type": "ti2v",
            "num_heads": 24,
            "num_layers": 30,
            "out_dim": 48,
            "text_len": 512
            }

        self.model = Yume1p5WanModel.from_config(config_wan)
        self.model.patch_embedding_2x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,4,4))
        self.model.patch_embedding_4x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,8,8))
        self.model.patch_embedding_8x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,16,16))
        self.model.patch_embedding_16x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,32,32))
        self.model.patch_embedding_2x_f = torch.nn.Conv3d(48, 48, kernel_size=(1,4,4), stride=(1,4,4))
        self.model.sideblock = Yume1p5WanAttentionBlock(self.model.dim, self.model.ffn_dim, self.model.num_heads, self.model.window_size, \
                                                    self.model.qk_norm, self.model.cross_attn_norm, self.model.eps)
        self.model.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, self.model.dim, device=self.model.device)
        )

        self.model = Yume1p5WanModel.from_pretrained(checkpoint_dir)
        state_dict = load_file(checkpoint_dir + "/diffusion_pytorch_model.safetensors")
        self.model.load_state_dict(state_dict)

        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 # specific to Yume 1.5
                 current_latent_num=8):

        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                n_prompt=n_prompt,
                seed=seed,
                # specific to Yume 1.5
                current_latent_num=current_latent_num)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            n_prompt="",
            seed=-1,
            offload_model=True):

        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # specific to Yume 1.5
        noise = torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g)

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}
        
        return arg_c, arg_null, noise

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            n_prompt="",
            seed=-1,
            # specific to Yume 1.5
            current_latent_num=8):

        # preprocess
        ih, iw = img.shape[2:] # specific to Yume 1.5
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)


        # Comment for Yume 1.5

        # scale = max(ow / iw, oh / ih)
        # img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # # center-crop
        # x1 = (img.width - ow) // 2
        # y1 = (img.height - oh) // 2
        # img = img.crop((x1, y1, x1 + ow, y1 + oh))
        # assert img.width == ow and img.height == oh

        # # to tensor
        # img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)


        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            # comment for Yume 1.5
            # if offload_model:
            #    self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]


        # specific to Yume 1.5
        z = img
        C, F_z, H, W = z.shape
        _, F_target, _, _ = noise.shape
        
        padding = F_target - F_z
        z = torch.cat([z, torch.zeros_like(z[:, -1:, :, :]).repeat(1, padding, 1, 1)], dim=1)
        z = [z]
        
        # sample videos
        latent = noise
        mask1, mask2 = masks_like([noise], zero=True, current_latent_num=current_latent_num)
        latent = (1. - mask2[0]) * z[0] + mask2[0] * latent
        
        arg_c = {
            'context': [context[0]],
            'seq_len': seq_len,
        }

        arg_null = {
            'context': context_null,
            'seq_len': seq_len,
        }

        return arg_c, arg_null, noise, mask2, z


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma


class Yume1p5Synthesis(BaseSynthesis):

    def __init__(
        self,
        model,
        device,
        weight_dtype
    ) -> None:
        
        super().__init__()

        self.model = model
        self.weight_dtype = weight_dtype
        self.device = device


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device,
        weight_dtype,
        fsdp
    ) -> "Yume1p5Synthesis":
        
        torch.backends.cuda.matmul.allow_tf32 = True
        
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            from huggingface_hub import snapshot_download
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")
        

        # set device and distributed settings
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        import torch.distributed as dist
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(rank)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        cfg = WAN_CONFIGS["ti2v-5B"]
        model = Yume1p5TI2V(
            config=cfg, 
            checkpoint_dir=model_root,
            device_id=rank,
            dit_fsdp=fsdp
        )

        model.model.eval().requires_grad_(False).to(weight_dtype)
        if not fsdp:
            model.model.to(device)

        return cls(
            model=model,
            device=device,
            weight_dtype=weight_dtype
        )
    
    @torch.no_grad()
    def predict_per_interaction(
        self, 
        prompt,
        image,
        video,
        interaction_idx,
        interaction,
        interaction_caption,
        interaction_speed,
        interaction_distance,
        task_type,
        size,
        seed,
        max_area,
        current_latent_num,
        current_frame_num,
        num_euler_timesteps,
        history_latents=None
    ):

        # prepare input caption
        caption = "First-person perspective." + interaction_caption

        INTERACTION_SPEED_and_DISTANCE_2_CAPTION_DICT = {
            "movement": "Actual distance moved: {distance} at {speed} meters per second.",
            "rotation": "View rotation speed: {speed}."
        }

        if "camera_" not in interaction.lower():
            interaction_type = "movement"
            caption += INTERACTION_SPEED_and_DISTANCE_2_CAPTION_DICT[interaction_type].format(
                speed=interaction_speed,
                distance=interaction_distance
            )
        else:
            interaction_type = "rotation"
            caption += INTERACTION_SPEED_and_DISTANCE_2_CAPTION_DICT[interaction_type].format(
                speed=interaction_speed
            )
        
        prompt = prompt if prompt else ""
        caption += prompt


        if interaction_idx == 0: # first interaction
            if task_type != "t2v":

                if task_type == "i2v":
                    video_tensor = torch.zeros(image.shape[0], 1+current_frame_num, size[0], size[1]) # (C, 33, H, W)
                    video_tensor[:, 0] = (image - 0.5) * 2
                    video = video_tensor.permute(1, 0, 2, 3) # (33, C, H, W)


                visual_content = video.squeeze().permute(1, 0, 2, 3).contiguous().to(self.device) # (C, F, H, W)

                visual_content_extended = torch.cat([visual_content[:, 0].unsqueeze(1).repeat(1, 16, 1, 1), visual_content[:,:33]], dim=1)
                history_current_frame_num = visual_content_extended.shape[1]

                visual_latents = torch.cat(
                    [
                        self.model.vae.encode([visual_content_extended.to(self.device)[:,:-32].to(self.device)])[0], \
                        self.model.vae.encode([visual_content_extended.to(self.device)[:,-32:].to(self.device)])[0]
                    ],
                    dim=1
                ) 
                history_latents = visual_latents[:, :-current_latent_num]

            else:
                history_current_frame_num = current_frame_num
                history_latents = None

        else: # continuation sampling
            history_current_frame_num = (history_latents.shape[1]-1)*4+1+32     

        if task_type != "t2v" or interaction_idx > 0:
            arg_c, arg_null, noise, mask2, img = self.model.generate(
                caption,
                frame_num=history_current_frame_num,
                max_area=max_area,
                current_latent_num=current_latent_num,
                img=history_latents,
                seed=seed
            )
        else:
            arg_c, arg_null, noise = self.model.generate(
                caption,
                frame_num=history_current_frame_num,
                max_area=max_area,
                current_latent_num=current_latent_num,
                seed=seed
            )

        if interaction_idx == 0: # first interaction
            latent = noise
        else: # continuation sampling
            history_zero_latents = torch.cat([history_latents, torch.zeros(48, current_latent_num, history_latents.shape[2], history_latents.shape[3]).to(self.device)], dim=1)
            noise = torch.randn_like(history_zero_latents)
            latent = noise.clone()


        sample_step_num = num_euler_timesteps
        sampling_sigmas = get_sampling_sigmas(sample_step_num, 7.0)


        if task_type != "t2v" or interaction_idx > 0:
            latent = torch.cat([img[0][:, :-current_latent_num, :, :], latent[:, -current_latent_num:, :, :]], dim=1)
    

        # update current latents
        with torch.autocast("cuda", dtype=self.weight_dtype):
            for i in range(sample_step_num):
                latent_model_input = [latent.squeeze(0)]

                if task_type != "t2v" or interaction_idx > 0:

                    timestep = [sampling_sigmas[i] * 1000]
                    timestep = torch.tensor(timestep).to(self.device)
                    temp_ts = (mask2[0][0][:-current_latent_num, ::2, ::2]).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(arg_c['seq_len'] - temp_ts.size(0)) * timestep
                    ])
                    timestep = temp_ts.unsqueeze(0)

                    noise_pred_cond = self.model.model(latent_model_input, t=timestep, **arg_c)[0]

                    if i + 1 == sample_step_num:
                        temp_x0 = latent[:, -current_latent_num:, :, :] + (0 - sampling_sigmas[i]) * noise_pred_cond[:, -current_latent_num:, :, :]
                    else:
                        temp_x0 = latent[:, -current_latent_num:, :, :] + (sampling_sigmas[i + 1] - sampling_sigmas[i]) * noise_pred_cond[:, -current_latent_num:, :, :]

                else:
                    timestep = [sampling_sigmas[i] * 1000]
                    timestep = torch.tensor(timestep).to(self.device)

                    noise_pred_cond = self.model.model(latent_model_input, t=timestep, flag=False, **arg_c)[0]

                    if i + 1 == sample_step_num:
                        latent = latent + (0 - sampling_sigmas[i]) * noise_pred_cond
                    else:
                        latent = latent + (sampling_sigmas[i + 1] - sampling_sigmas[i]) * noise_pred_cond

                if interaction_idx > 0:
                    latent = torch.cat([history_zero_latents[:, :-current_latent_num, :, :], temp_x0], dim=1)
                elif task_type != "t2v":
                    latent = torch.cat([visual_latents[:, :-current_latent_num, :, :], temp_x0], dim=1)


        if interaction_idx > 0:
            history_current_latents = torch.cat([history_latents, latent[:, -current_latent_num:, :, :]], dim=1)
        else:
            if task_type != "t2v":
                history_current_latents = torch.cat([visual_latents[:, :-current_latent_num, :, :], latent[:, -current_latent_num:, :, :]], dim=1)
            else:
                history_current_latents = latent

        with torch.autocast("cuda", dtype=torch.bfloat16):
            history_current_video = self.model.vae.decode([history_current_latents[:, -current_latent_num:, :, :].to(torch.float32)])[0]
            current_video = history_current_video[:, -current_frame_num:]

        return current_video, history_current_latents
            
    
    @torch.no_grad()
    def predict(
        self, 
        prompt, 
        image, 
        video, 
        interactions,
        interaction_captions, 
        interaction_speeds, 
        interaction_distances, 
        task_type, 
        size,
        seed,
        num_euler_timesteps
    ):
        # configs
        current_latent_num = 8 # time compression ratio is 4, so 8 latents corresponds to 32 frames
        current_frame_num = 32


        # inference per interaction
        output_video_list = []

        for interaction_idx, interaction_caption in enumerate(interaction_captions):
            output_video_per_interaction, history_latents = self.predict_per_interaction(
                prompt=prompt,
                image=image,
                video=video if interaction_idx == 0 else output_video_list[-1],
                interaction_idx=interaction_idx,
                interaction=interactions[interaction_idx],
                interaction_caption=interaction_caption, 
                interaction_speed=interaction_speeds[interaction_idx],
                interaction_distance=interaction_distances[interaction_idx], 
                task_type=task_type,
                size=SIZE_CONFIGS[size], 
                seed=seed,
                num_euler_timesteps=num_euler_timesteps,
                max_area=MAX_AREA_CONFIGS[size],
                current_latent_num=current_latent_num,
                current_frame_num=current_frame_num,
                history_latents=history_latents if interaction_idx > 0 else None
            )
            output_video_list.append(output_video_per_interaction)


        # postprocess output video
        vae_spatial_scale_factor = 8
        video_processor = VideoProcessor(vae_scale_factor=vae_spatial_scale_factor)

        output_video = video_processor.postprocess_video(torch.cat(output_video_list, dim=1).unsqueeze(0), output_type="pil")[0]

        return output_video
