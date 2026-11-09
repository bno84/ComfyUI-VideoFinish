"""RIFE 4.26 frame interpolation network (IFNet), inference only.

Reduced to the 4.26 architecture from Fannovel16/ComfyUI-Frame-Interpolation
vfi_models/rife/rife_arch.py at 26545cc2dd95bc3d27f056016300673bdeee78f5 (MIT, Copyright (c) 2023 Fannovel16),
itself based on hzwer/Practical-RIFE (MIT, Copyright (c) 2021 hzwer) and HolyWu/vs-rife (MIT,
Copyright (c) 2021 HolyWu). See THIRD_PARTY_NOTICES.md. Changes: only the 4.26 path is kept, training and
ensemble branches are removed, and the warp grid is built on the input's device instead of a device chosen at
import time. Parameter names are unchanged, so the published flownet.pkl loads with strict=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_grids = {}


def warp(image, flow):
    key = (str(flow.device), flow.dtype, tuple(flow.shape))
    grid = _grids.get(key)
    if grid is None:
        n, _, h, w = flow.shape
        horizontal = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype).view(1, 1, 1, w).expand(n, -1, h, -1)
        vertical = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype).view(1, 1, h, 1).expand(n, -1, -1, w)
        grid = _grids[key] = torch.cat([horizontal, vertical], 1)
    flow = torch.cat([flow[:, 0:1] / ((image.shape[3] - 1.0) / 2.0),
                      flow[:, 1:2] / ((image.shape[2] - 1.0) / 2.0)], 1)
    g = (grid + flow).permute(0, 2, 3, 1).to(image.dtype)
    return F.grid_sample(image, g, mode="bilinear", padding_mode="border", align_corners=True)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*(ResConv(c) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class IFNet(nn.Module):
    SCALES = (16, 8, 4, 2, 1)

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(self, img0, img1, timestep=0.5):
        """img0, img1: [N, 3, H, W] in 0..1. Returns the frame at `timestep` (0 = img0, 1 = img1)."""
        img0, img1 = img0.clamp(0, 1), img1.clamp(0, 1)
        n, c, h, w = img0.shape
        ph, pw = ((h - 1) // 64 + 1) * 64, ((w - 1) // 64 + 1) * 64
        img0 = F.pad(img0, (0, pw - w, 0, ph - h))
        img1 = F.pad(img1, (0, pw - w, 0, ph - h))
        t = torch.full_like(img0[:, :1], float(timestep))
        f0, f1 = self.encode(img0), self.encode(img1)
        blocks = (self.block0, self.block1, self.block2, self.block3, self.block4)
        flow = mask = feat = None
        warped0, warped1 = img0, img1
        for block, scale in zip(blocks, self.SCALES):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, t), 1), None, scale=scale)
            else:
                wf0, wf1 = warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4])
                fd, mask, feat = block(torch.cat((warped0, warped1, wf0, wf1, t, mask, feat), 1), flow, scale=scale)
                flow = flow + fd
            warped0, warped1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return (warped0 * mask + warped1 * (1 - mask))[:, :, :h, :w]
