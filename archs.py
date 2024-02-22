import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

__all__ = ['DPMNet']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
import pdb

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

from torchsummary import summary


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    # torch.roll(input,shifts,dims)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.GELU(),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4,
             expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    #print('x.shape',x.shape)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    h_patch_num = 8,
    w_patch_num = 8,
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(

            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.Linear(dim, (patch_size ** 2) * channels),

    )


def shift(dim):
    print("dim", dim)
    x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
    print("x_c", x_c)
    print("x_shift", x_shift)
    print("xs", xs)
    x_cat = torch.cat(x_shift, 1)
    print("x_cat1", x_cat)
    x_cat = torch.narrow(x_cat, 2, self.pad, H)
    print("x_cat2", x_cat)
    x_cat = torch.narrow(x_cat, 3, self.pad, W)
    print("x_cat3", x_cat)
    return x_cat


class shiftmlp1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5,
                 modelnumber=1, shift_num=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # self.shift_size=shift_size
        self.shift_size = shift_size
        self.shift_num = shift_num
        self.pad = shift_size // 2
        self.modelnumber = modelnumber
        self.apply(self._init_weights)
        self.model1 = MLPMixer(
            image_size=16,
            channels=160,
            patch_size=16,
            dim=32,
            depth=12,
            num_classes=50
        )
        self.model2 = MLPMixer(
            image_size=8,
            channels=256,
            patch_size=8,
            dim=16,
            depth=12,
            num_classes=50
        )
        self.model3 = MLPMixer(
            image_size=32,
            channels=128,
            patch_size=32,
            dim=64,
            depth=12,
            num_classes=50
        )
        self.model4 = MLPMixer(
            image_size=8,
            channels=160,
            patch_size=8,
            dim=16,
            depth=12,
            num_classes=50
        )
        self.model5 = MLPMixer(
            image_size=4,
            channels=256,
            patch_size=4,
            dim=8,
            depth=12,
            num_classes=50
        )
        self.model6 = MLPMixer(
            image_size=16,
            channels=128,
            patch_size=16,
            dim=32,
            depth=12,
            num_classes=50
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, H, W):
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)

        x_s = x
        ori = x
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x = self.fc1(x_shift_r)
        x = x + ori
        ori1 = x
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        mixer = x_s


        if (self.modelnumber == 1):

            print('mixer22-1',mixer.shape)
            m = self.model1(mixer)
            x_s = m.view((-1, 160, 16, 16))
        if (self.modelnumber == 2):
            m = self.model2(mixer)
            print('model2',mixer.shape)
            x_s = m.view((-1, 256, 8, 8))
        if (self.modelnumber == 3):
            print('model3',mixer.shape)
            m = self.model3(mixer)
            x_s = m.view((-1, 128, 32, 32))
        if (self.modelnumber == 4):
            print('model4,',mixer.shape)
            m = self.model4(mixer)
            x_s = m.view((-1, 160, 8, 8))
        if (self.modelnumber == 5):
            m = self.model5(mixer)
            print('model5,', mixer.shape)
            x_s = m.view((-1, 256, 4, 4))
        if (self.modelnumber == 6):
            m = self.model6(mixer)
            print('model6,', mixer.shape)
            x_s = m.view((-1, 128, 16, 16))

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        x = x + ori1
        return x

class shiftmlp0(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5,
                 modelnumber=1, shift_num=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.shift_num = shift_num
        self.pad = shift_size // 2
        self.modelnumber = modelnumber
        self.apply(self._init_weights)
        self.model1 = MLPMixer(
            image_size=16,
            channels=160,
            patch_size=16,
            dim=64,
            depth=12,
            num_classes=30
        )
        self.model2 = MLPMixer(
            image_size=8,
            channels=256,
            patch_size=8,
            dim=64,
            depth=12,
            num_classes=30
        )
        self.model3 = MLPMixer(
            image_size=32,
            channels=128,
            patch_size=32,
            dim=64,
            depth=12,
            num_classes=30
        )

        self.model4 = MLPMixer(
            image_size=8,
            channels=160,
            patch_size=8,
            dim=16,
            depth=12,
            num_classes=50
        )
        self.model5 = MLPMixer(
            image_size=4,
            channels=256,
            patch_size=4,
            dim=8,
            depth=12,
            num_classes=50
        )
        self.model6 = MLPMixer(
            image_size=16,
            channels=128,
            patch_size=16,
            dim=32,
            depth=12,
            num_classes=50
        )


    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape
        print("1this:", x.shape)
        ori = x
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        print("2this:", xn.shape)
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        print("3this:", xn.shape)
        #shift_size = 5
        xs = torch.chunk(xn, self.shift_size, 1)
        print("4this:",xs[1].shape)

        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]

        print("this:", x_shift[1].shape)

        x_cat = torch.cat(x_shift, 1)
        print("5this:", x_cat.shape)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)

        print("6this:", x_cat.shape)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        print('11',x_s.shape)
        mixer = x_s
        if (self.modelnumber == 1):
            m = self.model1(mixer)
            print('mixer',mixer.shape)
            x_s = m.view((-1, 160, 16, 16))
        if (self.modelnumber == 2):
            m = self.model2(mixer)
            print('model22,', mixer.shape)
            x_s = m.view((-1, 256, 8, 8))
        if (self.modelnumber == 3):
            m = self.model3(mixer)
            print('model33,', mixer.shape)
            x_s = m.view((-1, 128, 32, 32))
        if (self.modelnumber == 4):
            print('model44,',mixer.shape)
            print(next(self.model4.parameters()).device)
            m = self.model4(mixer)
            x_s = m.view((-1, 160, 8, 8))
        if (self.modelnumber == 5):
            m = self.model5(mixer)
            print('model55,', mixer.shape)
            x_s = m.view((-1, 256, 4, 4))
        if (self.modelnumber == 6):
            m = self.model6(mixer)
            print('model66,', mixer.shape)
            x_s = m.view((-1, 128, 16, 16))
            print('yes')
        x_s = x_s.reshape(B, C, H * W).contiguous()
        # print("8this:", x_s.shape)
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)
        x = x + ori
        ori1 = x
        x = self.dwconv(x, H, W)

        x = self.act(x)
        x = self.drop(x)
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_s = x
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        x = x + ori1
        # print('yeah')
        return x


class shiftedBlock1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, modelnum=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp0(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             modelnumber=modelnum)
        self.mlp1 = shiftmlp1(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              modelnumber=modelnum)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        m = self.drop_path(self.mlp(self.norm2(x), H, W))
        x = self.drop_path(self.mlp1(self.norm2(m), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


pair = lambda x: x if isinstance(x, tuple) else (x, x)


class DPMNet(nn.Module):

    ## Conv 3 + ARCMLP 2

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=256, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(128)#16

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([shiftedBlock1(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, modelnum=1,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock1(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, modelnum=2,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock1(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, modelnum=1,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock1(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, modelnum=3,
            sr_ratio=sr_ratios[0])])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.decodernew = nn.Conv2d(128,16,3,stride=1,padding=1)
        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbnnew=nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)
        self.encoder1_p = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2_p = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3_p = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1_p = nn.BatchNorm2d(16)
        self.ebn2_p = nn.BatchNorm2d(32)
        self.ebn3_p = nn.BatchNorm2d(128)
        self.norm3_p = norm_layer(embed_dims[1])
        self.norm4_p = norm_layer(embed_dims[2])
        self.dnorm3_p = norm_layer(160)
        self.dnorm4_p = norm_layer(128)

        self.block1_p = nn.ModuleList([shiftedBlock1(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, modelnum=4,
            sr_ratio=sr_ratios[0])])

        self.block2_p = nn.ModuleList([shiftedBlock1(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, modelnum=5,
            sr_ratio=sr_ratios[0])])

        self.dblock1_p = nn.ModuleList([shiftedBlock1(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, modelnum=4,
            sr_ratio=sr_ratios[0])])

        self.dblock2_p = nn.ModuleList([shiftedBlock1(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, modelnum=6,
            sr_ratio=sr_ratios[0])])
        img_size_p = img_size // 4  # 还是没用到

        self.patch_embed3_p = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[0],
                                                embed_dim=embed_dims[1])
        self.patch_embed4_p = OverlapPatchEmbed(img_size=img_size // 32, patch_size=3, stride=2, in_chans=embed_dims[1],
                                                embed_dim=embed_dims[2])

        self.decoder1_p = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2_p = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1_p = nn.BatchNorm2d(160)
        self.dbn2_p = nn.BatchNorm2d(128)
        self.dbn3_p = nn.BatchNorm2d(32)
        self.dbn4_p = nn.BatchNorm2d(16)


        self.final_p = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft_p = nn.Softmax(dim=1)

    def forward(self, x):
        xin = x.clone()
        B = x.shape[0]

        ### Encoder
        ### Conv Stage

        ### Stage 1

        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        print(t1.shape)

        ### ARC-MLP Stage
        ### Stage 2

        out, H, W = self.patch_embed3(out)

        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        print('beforefinal', out.shape)
        out = F.relu(self.decodernew(out))
        print('afternew', out.shape)
        x_loc = out.clone()

        for i in range(0, 2):
            for j in range(0, 2):
                x_p = xin[:, :, 128 * i:128 * (i + 1), 128 * j:128 * (j + 1)]
                B_p = x_p.shape[0]
                ### Encoder
                ### Conv Stage

                ### Stage 1
                out_p = F.relu(F.max_pool2d(self.ebn1_p(self.encoder1_p(x_p)), 2, 2))
                t1_p = out_p
                ### Stage 2
                out_p = F.relu(F.max_pool2d(self.ebn2_p(self.encoder2_p(out_p)), 2, 2))
                t2_p = out_p
                ### Stage 3
                out_p = F.relu(F.max_pool2d(self.ebn3_p(self.encoder3_p(out_p)), 2, 2))
                t3_p = out_p
                ### Tokenized MLP Stage
                ### Stage 4
                H_p = H / 2
                W_p = W / 2
                out_p, H_p, W_p = self.patch_embed3_p(out_p)
                for m, blk in enumerate(self.block1_p):  # for i,blk改成了for m，blk
                    out_p = blk(out_p, H_p, W_p)
                out_p = self.norm3_p(out_p)
                out_p = out_p.reshape(B_p, H_p, W_p, -1).permute(0, 3, 1, 2).contiguous()
                t4_p = out_p

                ### Bottleneck

                out_p, H_p, W_p = self.patch_embed4_p(out_p)
                for m, blk in enumerate(self.block2_p):
                    out_p = blk(out_p, H_p, W_p)
                out_p = self.norm4_p(out_p)
                out_p = out_p.reshape(B_p, H_p, W_p, -1).permute(0, 3, 1, 2).contiguous()

                ### Stage 4

                out_p = F.relu(F.interpolate(self.dbn1_p(self.decoder1_p(out_p)), scale_factor=(2, 2), mode='bilinear'))
                # 功能：利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整。

                out_p = torch.add(out_p, t4_p)
                _, _, H_p, W_p = out_p.shape
                out_p = out_p.flatten(2).transpose(1, 2)
                for m, blk in enumerate(self.dblock1_p):
                    out_p = blk(out_p, H_p, W_p)

                ### Stage 3

                out_p = self.dnorm3_p(out_p)
                out_p = out_p.reshape(B_p, H_p, W_p, -1).permute(0, 3, 1, 2).contiguous()
                out_p = F.relu(F.interpolate(self.dbn2_p(self.decoder2_p(out_p)), scale_factor=(2, 2), mode='bilinear'))
                out_p = torch.add(out_p, t3_p)
                _, _, H_p, W_p = out_p.shape
                out_p = out_p.flatten(2).transpose(1, 2)

                for m, blk in enumerate(self.dblock2_p):
                    out_p = blk(out_p, H_p, W_p)

                out_p = self.dnorm4_p(out_p)
                out_p = out_p.reshape(B_p, H_p, W_p, -1).permute(0, 3, 1, 2).contiguous()

                out_p = F.relu(F.interpolate(self.dbn3_p(self.decoder3_p(out_p)), scale_factor=(2, 2), mode='bilinear'))
                out_p = torch.add(out_p, t2_p)
                out_p = F.relu(F.interpolate(self.dbn4_p(self.decoder4_p(out_p)), scale_factor=(2, 2), mode='bilinear'))
                out_p = torch.add(out_p, t1_p)
                out_p = F.relu(F.interpolate(self.decoder5_p(out_p), scale_factor=(2, 2), mode='bilinear'))

                x_loc[:, :, 128 * i:128 * (i + 1), 128 * j:128 * (j + 1)] = out_p

            out = torch.add(out, x_loc)
            out = F.relu(out)
            out = self.final(out)

            return out


