# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

M_POS, M_NEG = 5.0, 1.0     # 式(13)：正、负样本目标间隔
LAM1, LAM2 = 0.01, 0.001    # 式(12)：λ1=0.01（作用于 w），λ2=0.001（作用于 θ）


class DWCN(nn.Module):
    def __init__(self, in_channels=3, img_height=70, img_width=70):
        super().__init__()
        self.H, self.W = img_height, img_width
        self.num_pixels = self.H * self.W
	self.use_median_filter = True 
        # 位置专属加权层：每像素位置一个 in_channels 维权重向量，无偏置
        self.pixel_conv = nn.Conv2d(in_channels, self.num_pixels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.pixel_conv.weight, mode='fan_out', nonlinearity='relu')
        self.pixel_conv.weight.data = torch.relu(self.pixel_conv.weight.data)  # 非负初始化
        idx = torch.tensor([(k // self.W, k % self.W) for k in range(self.num_pixels)], dtype=torch.long)
        self.register_buffer('pixel_indices', idx)  # 位置索引：第 k 通道绑定坐标 (⌊k/W⌋, k mod W)

    def forward(self, x):
        w = torch.relu(self.pixel_conv.weight)      # 式(10)：w = ReLU(θ)，每次前向执行
        conv_out = F.conv2d(x, w)                   # F ∈ R^{B×4900×70×70}
        B = conv_out.shape[0]
        k = torch.arange(self.num_pixels, device=conv_out.device)
        i = self.pixel_indices[:, 0].expand(B, -1)
        j = self.pixel_indices[:, 1].expand(B, -1)
        pixel_wise_out = conv_out[torch.arange(B)[:, None], k, i, j]  # 第k通道仅在其绑定坐标被读取
        return pixel_wise_out.sum(dim=1)            # 式(11)：整图得分 S
	
   @staticmethod
    def _median3(x):                    # x: B×70×70，3×3中值滤波（可微）
        xp = F.pad(x.unsqueeze(1), (1, 1, 1, 1), mode='reflect').squeeze(1)
        patches = xp.unfold(1, 3, 1).unfold(2, 3, 1)
       	return patches.reshape(*patches.shape[:3], 9).median(dim=-1).values



def dwcn_loss(S, y, model):
    """论文式(12)(13)：合页损失 + λ1‖w‖1 + λ2‖min(0,θ)‖1（θ 为潜变量，非负软约束作用于 θ）"""
    hinge = y * torch.clamp(M_POS - S, min=0) + (1 - y) * torch.clamp(S - M_NEG, min=0)
    w = torch.relu(model.pixel_conv.weight)
    l1 = LAM1 * w.sum()                                             # λ1‖w‖_1（w≥0）
    non_neg = LAM2 * (-model.pixel_conv.weight).clamp(min=0).sum()  # λ2‖min(0,θ)‖_1
    return hinge.mean() + l1 + non_neg
