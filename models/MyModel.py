import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import pc_denormalize, InitPcs, pattern_in_sc, pc_normalize, pc_denormalize
from utils.geometry_utils import find_near_pcd_param, compute_attenuation_with_kdtree_gpu
import math



class GreenLayer(nn.Module):

    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.pai = 3.1415926535897931
        self.omega = 2*self.pai*self.freq
        self.mu0 = 4.0*self.pai*1.0e-07
        self.c = 2.99792458e8
        self.epsilon0 = 1.0e0/(self.c**2*self.mu0)
        self.k = self.omega * (self.mu0 * self.epsilon0)**0.5

    def forward(self, x):
        tloc = x[0]
        rloc = x[1]

        r = ((tloc[...,0]-rloc[...,0])**2 + (tloc[...,1]-rloc[...,1])**2 + (tloc[...,2]-rloc[...,2])**2)**0.5
        g_real = torch.cos(-self.k*r) / (4*self.pai*r)
        g_imag = torch.sin(-self.k*r) / (4*self.pai*r)
        coeff_imag = -self.omega*self.mu0
        h1_real = r**2 - 1/self.k**2
        h1_imag = -r/self.k
        h2_real = -1 + 3/(self.k**2*r**2)
        h2_imag = 3/(self.k*r)

        Gxx_real = h1_real + (tloc[...,0]-rloc[...,0])**2 * h2_real
        Gxx_imag = h1_imag + (tloc[...,0]-rloc[...,0])**2 * h2_imag
        Gyy_real = h1_real + (tloc[...,1]-rloc[...,1])**2 * h2_real
        Gyy_imag = h1_imag + (tloc[...,1]-rloc[...,1])**2 * h2_imag
        Gzz_real = h1_real + (tloc[...,2]-rloc[...,2])**2 * h2_real
        Gzz_imag = h1_imag + (tloc[...,2]-rloc[...,2])**2 * h2_imag
        Gxy_real = (tloc[...,0]-rloc[...,0]) * (tloc[...,1]-rloc[...,1]) * h2_real
        Gxy_imag = (tloc[...,0]-rloc[...,0]) * (tloc[...,1]-rloc[...,1]) * h2_imag
        Gyx_real = (tloc[...,0]-rloc[...,0]) * (tloc[...,1]-rloc[...,1]) * h2_real
        Gyx_imag = (tloc[...,0]-rloc[...,0]) * (tloc[...,1]-rloc[...,1]) * h2_imag
        Gxz_real = (tloc[...,0]-rloc[...,0]) * (tloc[...,2]-rloc[...,2]) * h2_real
        Gxz_imag = (tloc[...,0]-rloc[...,0]) * (tloc[...,2]-rloc[...,2]) * h2_imag
        Gzx_real = (tloc[...,0]-rloc[...,0]) * (tloc[...,2]-rloc[...,2]) * h2_real
        Gzx_imag = (tloc[...,0]-rloc[...,0]) * (tloc[...,2]-rloc[...,2]) * h2_imag
        Gyz_real = (tloc[...,1]-rloc[...,1]) * (tloc[...,2]-rloc[...,2]) * h2_real
        Gyz_imag = (tloc[...,1]-rloc[...,1]) * (tloc[...,2]-rloc[...,2]) * h2_imag
        Gzy_real = (tloc[...,1]-rloc[...,1]) * (tloc[...,2]-rloc[...,2]) * h2_real
        Gzy_imag = (tloc[...,1]-rloc[...,1]) * (tloc[...,2]-rloc[...,2]) * h2_imag

        G_real_temp = torch.stack([Gxx_real, Gxy_real, Gxz_real, Gyx_real, Gyy_real, Gyz_real, Gzx_real, Gzy_real, Gzz_real], dim=-1)
        G_imag_temp = torch.stack([Gxx_imag, Gxy_imag, Gxz_imag, Gyx_imag, Gyy_imag, Gyz_imag, Gzx_imag, Gzy_imag, Gzz_imag], dim=-1)
        r = torch.unsqueeze(r, dim=-1)
        g_real = torch.unsqueeze(g_real, dim=-1)
        g_imag = torch.unsqueeze(g_imag, dim=-1)
        G_imag = (g_real / r**2 * G_real_temp - g_imag / r**2 * G_imag_temp) * coeff_imag
        G_real = (g_real / r**2 * G_imag_temp + g_imag / r**2 * G_real_temp) * -coeff_imag

        G = torch.stack([G_real, G_imag], dim=-1) / 1000.0
        # G = torch.stack([G_real, G_imag], dim=-1) / 4000.0
        # G = torch.stack([G_real, G_imag], dim=-1) / 0.01

        return G  # [batch, n_scatter, 9, 2]


class ComputeGsct(nn.Module):

    def __init__(self, freq, is_attenuation):
        super().__init__()
        self.freq = freq
        self.pai = 3.1415926535897931
        self.c = 2.99792458e8
        self.omega = 2 * self.pai * freq
        self.mu0 = 4.0 * self.pai * 1.0e-07
        self.epsilon0 = 1.0e0 / (self.c ** 2 * self.mu0)
        self.is_attenuation = is_attenuation

    def forward(self, x):
        gsr_real = x[0][:, :, :, 0]
        gsr_imag = x[0][:, :, :, 1]
        grf_real = x[1][:, :, :, 0]
        grf_imag = x[1][:, :, :, 1]

        if not self.is_attenuation:
            kai = x[2][:, :, 0] * 10.0

            gsrxx_kai_real = gsr_real[:, :, 0] * kai
            gsrxx_kai_imag = gsr_imag[:, :, 0] * kai

            gsrxy_kai_real = gsr_real[:, :, 1] * kai
            gsrxy_kai_imag = gsr_imag[:, :, 1] * kai

            gsrxz_kai_real = gsr_real[:, :, 2] * kai
            gsrxz_kai_imag = gsr_imag[:, :, 2] * kai

            gsryx_kai_real = gsr_real[:, :, 3] * kai
            gsryx_kai_imag = gsr_imag[:, :, 3] * kai

            gsryy_kai_real = gsr_real[:, :, 4] * kai
            gsryy_kai_imag = gsr_imag[:, :, 4] * kai

            gsryz_kai_real = gsr_real[:, :, 5] * kai
            gsryz_kai_imag = gsr_imag[:, :, 5] * kai

            gsrzx_kai_real = gsr_real[:, :, 6] * kai
            gsrzx_kai_imag = gsr_imag[:, :, 6] * kai

            gsrzy_kai_real = gsr_real[:, :, 7] * kai
            gsrzy_kai_imag = gsr_imag[:, :, 7] * kai

            gsrzz_kai_real = gsr_real[:, :, 8] * kai
            gsrzz_kai_imag = gsr_imag[:, :, 8] * kai
        else:
            kai_real = x[2][:, :, 0] * 10.0
            kai_imag = -x[2][:, :, 1] / (self.omega * self.epsilon0)

            gsrxx_kai_real = gsr_real[:, :, 0] * kai_real - gsr_imag[:, :, 0] * kai_imag
            gsrxx_kai_imag = gsr_imag[:, :, 0] * kai_real + gsr_real[:, :, 0] * kai_imag

            gsrxy_kai_real = gsr_real[:, :, 1] * kai_real - gsr_imag[:, :, 1] * kai_imag
            gsrxy_kai_imag = gsr_imag[:, :, 1] * kai_real + gsr_real[:, :, 1] * kai_imag

            gsrxz_kai_real = gsr_real[:, :, 2] * kai_real - gsr_imag[:, :, 2] * kai_imag
            gsrxz_kai_imag = gsr_imag[:, :, 2] * kai_real + gsr_real[:, :, 2] * kai_imag

            gsryx_kai_real = gsr_real[:, :, 3] * kai_real - gsr_imag[:, :, 3] * kai_imag
            gsryx_kai_imag = gsr_imag[:, :, 3] * kai_real + gsr_real[:, :, 3] * kai_imag

            gsryy_kai_real = gsr_real[:, :, 4] * kai_real - gsr_imag[:, :, 4] * kai_imag
            gsryy_kai_imag = gsr_imag[:, :, 4] * kai_real + gsr_real[:, :, 4] * kai_imag

            gsryz_kai_real = gsr_real[:, :, 5] * kai_real - gsr_imag[:, :, 5] * kai_imag
            gsryz_kai_imag = gsr_imag[:, :, 5] * kai_real + gsr_real[:, :, 5] * kai_imag

            gsrzx_kai_real = gsr_real[:, :, 6] * kai_real - gsr_imag[:, :, 6] * kai_imag
            gsrzx_kai_imag = gsr_imag[:, :, 6] * kai_real + gsr_real[:, :, 6] * kai_imag

            gsrzy_kai_real = gsr_real[:, :, 7] * kai_real - gsr_imag[:, :, 7] * kai_imag
            gsrzy_kai_imag = gsr_imag[:, :, 7] * kai_real + gsr_real[:, :, 7] * kai_imag

            gsrzz_kai_real = gsr_real[:, :, 8] * kai_real - gsr_imag[:, :, 8] * kai_imag
            gsrzz_kai_imag = gsr_imag[:, :, 8] * kai_real + gsr_real[:, :, 8] * kai_imag

        gsrxx_kai_grfxx_real = gsrxx_kai_real * grf_real[:, :, 0] - gsrxx_kai_imag * grf_imag[:, :, 0]
        gsrxx_kai_grfxx_imag = gsrxx_kai_imag * grf_real[:, :, 0] + gsrxx_kai_real * grf_imag[:, :, 0]

        gsrxy_kai_grfyx_real = gsrxy_kai_real * grf_real[:, :, 3] - gsrxy_kai_imag * grf_imag[:, :, 3]
        gsrxy_kai_grfyx_imag = gsrxy_kai_imag * grf_real[:, :, 3] + gsrxy_kai_real * grf_imag[:, :, 3]

        gsrxz_kai_grfzx_real = gsrxz_kai_real * grf_real[:, :, 6] - gsrxz_kai_imag * grf_imag[:, :, 6]
        gsrxz_kai_grfzx_imag = gsrxz_kai_imag * grf_real[:, :, 6] + gsrxz_kai_real * grf_imag[:, :, 6]

        gsctxx_real = torch.sum(gsrxx_kai_grfxx_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrxy_kai_grfyx_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrxz_kai_grfzx_real, dim=-1, keepdims=True)

        gsctxx_imag = torch.sum(gsrxx_kai_grfxx_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrxy_kai_grfyx_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrxz_kai_grfzx_imag, dim=-1, keepdims=True)

        gsrxx_kai_grfxy_real = gsrxx_kai_real * grf_real[:, :, 1] - gsrxx_kai_imag * grf_imag[:, :, 1]
        gsrxx_kai_grfxy_imag = gsrxx_kai_imag * grf_real[:, :, 1] + gsrxx_kai_real * grf_imag[:, :, 1]

        gsrxy_kai_grfyy_real = gsrxy_kai_real * grf_real[:, :, 4] - gsrxy_kai_imag * grf_imag[:, :, 4]
        gsrxy_kai_grfyy_imag = gsrxy_kai_imag * grf_real[:, :, 4] + gsrxy_kai_real * grf_imag[:, :, 4]

        gsrxz_kai_grfzy_real = gsrxz_kai_real * grf_real[:, :, 7] - gsrxz_kai_imag * grf_imag[:, :, 7]
        gsrxz_kai_grfzy_imag = gsrxz_kai_imag * grf_real[:, :, 7] + gsrxz_kai_real * grf_imag[:, :, 7]

        gsctxy_real = torch.sum(gsrxx_kai_grfxy_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrxy_kai_grfyy_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrxz_kai_grfzy_real, dim=-1, keepdims=True)

        gsctxy_imag = torch.sum(gsrxx_kai_grfxy_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrxy_kai_grfyy_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrxz_kai_grfzy_imag, dim=-1, keepdims=True)

        gsrxx_kai_grfxz_real = gsrxx_kai_real * grf_real[:, :, 2] - gsrxx_kai_imag * grf_imag[:, :, 2]
        gsrxx_kai_grfxz_imag = gsrxx_kai_imag * grf_real[:, :, 2] + gsrxx_kai_real * grf_imag[:, :, 2]

        gsrxy_kai_grfyz_real = gsrxy_kai_real * grf_real[:, :, 5] - gsrxy_kai_imag * grf_imag[:, :, 5]
        gsrxy_kai_grfyz_imag = gsrxy_kai_imag * grf_real[:, :, 5] + gsrxy_kai_real * grf_imag[:, :, 5]

        gsrxz_kai_grfzz_real = gsrxz_kai_real * grf_real[:, :, 8] - gsrxz_kai_imag * grf_imag[:, :, 8]
        gsrxz_kai_grfzz_imag = gsrxz_kai_imag * grf_real[:, :, 8] + gsrxz_kai_real * grf_imag[:, :, 8]

        gsctxz_real = torch.sum(gsrxx_kai_grfxz_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrxy_kai_grfyz_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrxz_kai_grfzz_real, dim=-1, keepdims=True)

        gsctxz_imag = torch.sum(gsrxx_kai_grfxz_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrxy_kai_grfyz_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrxz_kai_grfzz_imag, dim=-1, keepdims=True)

        gsryx_kai_grfxx_real = gsryx_kai_real * grf_real[:, :, 0] - gsryx_kai_imag * grf_imag[:, :, 0]
        gsryx_kai_grfxx_imag = gsryx_kai_imag * grf_real[:, :, 0] + gsryx_kai_real * grf_imag[:, :, 0]

        gsryy_kai_grfyx_real = gsryy_kai_real * grf_real[:, :, 3] - gsryy_kai_imag * grf_imag[:, :, 3]
        gsryy_kai_grfyx_imag = gsryy_kai_imag * grf_real[:, :, 3] + gsryy_kai_real * grf_imag[:, :, 3]

        gsryz_kai_grfzx_real = gsryz_kai_real * grf_real[:, :, 6] - gsryz_kai_imag * grf_imag[:, :, 6]
        gsryz_kai_grfzx_imag = gsryz_kai_imag * grf_real[:, :, 6] + gsryz_kai_real * grf_imag[:, :, 6]

        gsctyx_real = torch.sum(gsryx_kai_grfxx_real, dim=-1, keepdims=True) + \
                      torch.sum(gsryy_kai_grfyx_real, dim=-1, keepdims=True) + \
                      torch.sum(gsryz_kai_grfzx_real, dim=-1, keepdims=True)

        gsctyx_imag = torch.sum(gsryx_kai_grfxx_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsryy_kai_grfyx_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsryz_kai_grfzx_imag, dim=-1, keepdims=True)

        gsryx_kai_grfxy_real = gsryx_kai_real * grf_real[:, :, 1] - gsryx_kai_imag * grf_imag[:, :, 1]
        gsryx_kai_grfxy_imag = gsryx_kai_imag * grf_real[:, :, 1] + gsryx_kai_real * grf_imag[:, :, 1]

        gsryy_kai_grfyy_real = gsryy_kai_real * grf_real[:, :, 4] - gsryy_kai_imag * grf_imag[:, :, 4]
        gsryy_kai_grfyy_imag = gsryy_kai_imag * grf_real[:, :, 4] + gsryy_kai_real * grf_imag[:, :, 4]

        gsryz_kai_grfzy_real = gsryz_kai_real * grf_real[:, :, 7] - gsryz_kai_imag * grf_imag[:, :, 7]
        gsryz_kai_grfzy_imag = gsryz_kai_imag * grf_real[:, :, 7] + gsryz_kai_real * grf_imag[:, :, 7]

        gsctyy_real = torch.sum(gsryx_kai_grfxy_real, dim=-1, keepdims=True) + \
                      torch.sum(gsryy_kai_grfyy_real, dim=-1, keepdims=True) + \
                      torch.sum(gsryz_kai_grfzy_real, dim=-1, keepdims=True)

        gsctyy_imag = torch.sum(gsryx_kai_grfxy_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsryy_kai_grfyy_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsryz_kai_grfzy_imag, dim=-1, keepdims=True)

        gsryx_kai_grfxz_real = gsryx_kai_real * grf_real[:, :, 2] - gsryx_kai_imag * grf_imag[:, :, 2]
        gsryx_kai_grfxz_imag = gsryx_kai_imag * grf_real[:, :, 2] + gsryx_kai_real * grf_imag[:, :, 2]

        gsryy_kai_grfyz_real = gsryy_kai_real * grf_real[:, :, 5] - gsryy_kai_imag * grf_imag[:, :, 5]
        gsryy_kai_grfyz_imag = gsryy_kai_imag * grf_real[:, :, 5] + gsryy_kai_real * grf_imag[:, :, 5]

        gsryz_kai_grfzz_real = gsryz_kai_real * grf_real[:, :, 8] - gsryz_kai_imag * grf_imag[:, :, 8]
        gsryz_kai_grfzz_imag = gsryz_kai_imag * grf_real[:, :, 8] + gsryz_kai_real * grf_imag[:, :, 8]

        gsctyz_real = torch.sum(gsryx_kai_grfxz_real, dim=-1, keepdims=True) + \
                      torch.sum(gsryy_kai_grfyz_real, dim=-1, keepdims=True) + \
                      torch.sum(gsryz_kai_grfzz_real, dim=-1, keepdims=True)

        gsctyz_imag = torch.sum(gsryx_kai_grfxz_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsryy_kai_grfyz_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsryz_kai_grfzz_imag, dim=-1, keepdims=True)

        gsrzx_kai_grfxx_real = gsrzx_kai_real * grf_real[:, :, 0] - gsrzx_kai_imag * grf_imag[:, :, 0]
        gsrzx_kai_grfxx_imag = gsrzx_kai_imag * grf_real[:, :, 0] + gsrzx_kai_real * grf_imag[:, :, 0]

        gsrzy_kai_grfyx_real = gsrzy_kai_real * grf_real[:, :, 3] - gsrzy_kai_imag * grf_imag[:, :, 3]
        gsrzy_kai_grfyx_imag = gsrzy_kai_imag * grf_real[:, :, 3] + gsrzy_kai_real * grf_imag[:, :, 3]

        gsrzz_kai_grfzx_real = gsrzz_kai_real * grf_real[:, :, 6] - gsrzz_kai_imag * grf_imag[:, :, 6]
        gsrzz_kai_grfzx_imag = gsrzz_kai_imag * grf_real[:, :, 6] + gsrzz_kai_real * grf_imag[:, :, 6]

        gsctzx_real = torch.sum(gsrzx_kai_grfxx_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrzy_kai_grfyx_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrzz_kai_grfzx_real, dim=-1, keepdims=True)

        gsctzx_imag = torch.sum(gsrzx_kai_grfxx_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrzy_kai_grfyx_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrzz_kai_grfzx_imag, dim=-1, keepdims=True)

        gsrzx_kai_grfxy_real = gsrzx_kai_real * grf_real[:, :, 1] - gsrzx_kai_imag * grf_imag[:, :, 1]
        gsrzx_kai_grfxy_imag = gsrzx_kai_imag * grf_real[:, :, 1] + gsrzx_kai_real * grf_imag[:, :, 1]

        gsrzy_kai_grfyy_real = gsrzy_kai_real * grf_real[:, :, 4] - gsrzy_kai_imag * grf_imag[:, :, 4]
        gsrzy_kai_grfyy_imag = gsrzy_kai_imag * grf_real[:, :, 4] + gsrzy_kai_real * grf_imag[:, :, 4]

        gsrzz_kai_grfzy_real = gsrzz_kai_real * grf_real[:, :, 7] - gsrzz_kai_imag * grf_imag[:, :, 7]
        gsrzz_kai_grfzy_imag = gsrzz_kai_imag * grf_real[:, :, 7] + gsrzz_kai_real * grf_imag[:, :, 7]

        gsctzy_real = torch.sum(gsrzx_kai_grfxy_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrzy_kai_grfyy_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrzz_kai_grfzy_real, dim=-1, keepdims=True)

        gsctzy_imag = torch.sum(gsrzx_kai_grfxy_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrzy_kai_grfyy_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrzz_kai_grfzy_imag, dim=-1, keepdims=True)

        gsrzx_kai_grfxz_real = gsrzx_kai_real * grf_real[:, :, 2] - gsrzx_kai_imag * grf_imag[:, :, 2]
        gsrzx_kai_grfxz_imag = gsrzx_kai_imag * grf_real[:, :, 2] + gsrzx_kai_real * grf_imag[:, :, 2]

        gsrzy_kai_grfyz_real = gsrzy_kai_real * grf_real[:, :, 5] - gsrzy_kai_imag * grf_imag[:, :, 5]
        gsrzy_kai_grfyz_imag = gsrzy_kai_imag * grf_real[:, :, 5] + gsrzy_kai_real * grf_imag[:, :, 5]

        gsrzz_kai_grfzz_real = gsrzz_kai_real * grf_real[:, :, 8] - gsrzz_kai_imag * grf_imag[:, :, 8]
        gsrzz_kai_grfzz_imag = gsrzz_kai_imag * grf_real[:, :, 8] + gsrzz_kai_real * grf_imag[:, :, 8]

        gsctzz_real = torch.sum(gsrzx_kai_grfxz_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrzy_kai_grfyz_real, dim=-1, keepdims=True) + \
                      torch.sum(gsrzz_kai_grfzz_real, dim=-1, keepdims=True)

        gsctzz_imag = torch.sum(gsrzx_kai_grfxz_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrzy_kai_grfyz_imag, dim=-1, keepdims=True) + \
                      torch.sum(gsrzz_kai_grfzz_imag, dim=-1, keepdims=True)

        Gsct_real = -self.omega * self.epsilon0 * torch.concat(
            [gsctxx_imag, gsctxy_imag, gsctxz_imag, gsctyx_imag, gsctyy_imag,
             gsctyz_imag, gsctzx_imag, gsctzy_imag, gsctzz_imag], dim=-1)  #

        Gsct_imag = self.omega * self.epsilon0 * torch.concat(
            [gsctxx_real, gsctxy_real, gsctxz_real, gsctyx_real, gsctyy_real,
             gsctyz_real, gsctzx_real, gsctzy_real, gsctzz_real], dim=-1)

        Gsct = torch.stack([Gsct_real, Gsct_imag], dim=-1) / 1000.0
        # Gsct = torch.stack([Gsct_real, Gsct_imag], dim=-1) / 4000.0
        # Gsct = torch.stack([Gsct_real, Gsct_imag], dim=-1) / 0.01

        return Gsct




class GreenNet(nn.Module):
    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.Gsr = GreenLayer(self.freq)
        self.Grf = GreenLayer(self.freq)
        self.Gsf = GreenLayer(self.freq)
        self.Gsct = ComputeGsct(self.freq, True)

    def forward(self, tloc, rloc, r, kai, n_scatters=128):  # rcube ckai
        r = torch.concat(r, dim=1)
        kai = torch.concat(kai, dim=1)
        s = torch.tile(tloc.unsqueeze(1), (1, n_scatters, 1))
        f = torch.tile(rloc.unsqueeze(1), (1, n_scatters, 1))
        gsr = self.Gsr((s, r))  # [batch, n_point, 9, 2]
        grf = self.Grf((r, f))  # [batch, n_scatter, 9, 2]
        gsf = self.Gsf((tloc, rloc))  # [batch, n_scatter, 9, 2]
        gsct = self.Gsct((gsr, grf, kai)) # [batch, 9, 2]
        gtot = gsct + gsf
        return gtot, kai, r



class GreenLayer_simple(nn.Module):

    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.pai = 3.1415926535897931
        self.omega = 2*self.pai*self.freq
        self.mu0 = 4.0*self.pai*1.0e-07
        self.c = 2.99792458e8
        self.epsilon0 = 1.0e0/(self.c**2*self.mu0)
        self.k = self.omega * (self.mu0 * self.epsilon0)**0.5

    def forward(self, x):

        tloc = x[0]
        rloc = x[1]

        r = ((tloc[..., 0] - rloc[..., 0]) ** 2 + (tloc[..., 1] - rloc[..., 1]) ** 2 + (
                    tloc[..., 2] - rloc[..., 2]) ** 2) ** 0.5
        g_real = torch.cos(-self.k * r) / (4 * self.pai * r)
        g_imag = torch.sin(-self.k * r) / (4 * self.pai * r)
        coeff_imag = -self.omega * self.mu0
        h1_real = r ** 2 - 1 / self.k ** 2
        h1_imag = -r / self.k
        h2_real = -1 + 3 / (self.k ** 2 * r ** 2)
        h2_imag = 3 / (self.k * r)

        Gxx_real = h1_real + (tloc[..., 0] - rloc[..., 0]) ** 2 * h2_real
        Gxx_imag = h1_imag + (tloc[..., 0] - rloc[..., 0]) ** 2 * h2_imag
        Gyy_real = h1_real + (tloc[..., 1] - rloc[..., 1]) ** 2 * h2_real
        Gyy_imag = h1_imag + (tloc[..., 1] - rloc[..., 1]) ** 2 * h2_imag
        Gzz_real = h1_real + (tloc[..., 2] - rloc[..., 2]) ** 2 * h2_real
        Gzz_imag = h1_imag + (tloc[..., 2] - rloc[..., 2]) ** 2 * h2_imag
        Gxy_real = (tloc[..., 0] - rloc[..., 0]) * (tloc[..., 1] - rloc[..., 1]) * h2_real
        Gxy_imag = (tloc[..., 0] - rloc[..., 0]) * (tloc[..., 1] - rloc[..., 1]) * h2_imag
        Gyx_real = (tloc[..., 0] - rloc[..., 0]) * (tloc[..., 1] - rloc[..., 1]) * h2_real
        Gyx_imag = (tloc[..., 0] - rloc[..., 0]) * (tloc[..., 1] - rloc[..., 1]) * h2_imag
        Gxz_real = (tloc[..., 0] - rloc[..., 0]) * (tloc[..., 2] - rloc[..., 2]) * h2_real
        Gxz_imag = (tloc[..., 0] - rloc[..., 0]) * (tloc[..., 2] - rloc[..., 2]) * h2_imag
        Gzx_real = (tloc[..., 0] - rloc[..., 0]) * (tloc[..., 2] - rloc[..., 2]) * h2_real
        Gzx_imag = (tloc[..., 0] - rloc[..., 0]) * (tloc[..., 2] - rloc[..., 2]) * h2_imag
        Gyz_real = (tloc[..., 1] - rloc[..., 1]) * (tloc[..., 2] - rloc[..., 2]) * h2_real
        Gyz_imag = (tloc[..., 1] - rloc[..., 1]) * (tloc[..., 2] - rloc[..., 2]) * h2_imag
        Gzy_real = (tloc[..., 1] - rloc[..., 1]) * (tloc[..., 2] - rloc[..., 2]) * h2_real
        Gzy_imag = (tloc[..., 1] - rloc[..., 1]) * (tloc[..., 2] - rloc[..., 2]) * h2_imag

        G_real_temp = torch.stack(
            [Gxx_real, Gxy_real, Gxz_real, Gyx_real, Gyy_real, Gyz_real, Gzx_real, Gzy_real, Gzz_real], dim=-1)
        G_imag_temp = torch.stack(
            [Gxx_imag, Gxy_imag, Gxz_imag, Gyx_imag, Gyy_imag, Gyz_imag, Gzx_imag, Gzy_imag, Gzz_imag], dim=-1)
        r = torch.unsqueeze(r, dim=-1)
        g_real = torch.unsqueeze(g_real, dim=-1)
        g_imag = torch.unsqueeze(g_imag, dim=-1)
        G_imag = (g_real / r ** 2 * G_real_temp - g_imag / r ** 2 * G_imag_temp) * coeff_imag
        G_real = (g_real / r ** 2 * G_imag_temp + g_imag / r ** 2 * G_real_temp) * -coeff_imag

        # G = torch.stack([G_real, G_imag], dim=-1) / 1000.0
        # # G = torch.stack([G_real, G_imag], dim=-1) / 4000.0
        # # G = torch.stack([G_real, G_imag], dim=-1) / 0.01

        G = torch.stack([G_real, G_imag], dim=-1) #  / 1000   # pazhou

        G = G[..., -1, :] # VV

        # G = G[..., 6, :] # VH  z-x
        # G = G[..., 0, :] # HH  x-x
        # G = G[..., 2, :] # HV  x-z

        # G = G[..., 2, :] # VH  z-x
        # G = G[..., 0, :] # HH  x-x
        # G = G[..., 6, :] # HV  x-z

        # g_real = torch.cos(-self.k*r) / (4*self.pai*r) * self.mu0 * self.omega
        # g_imag = torch.sin(-self.k*r) / (4*self.pai*r) * self.mu0 * self.omega
        # G = torch.stack([g_imag, -1 * g_real], dim=-1) / 1000.0

        return G

class ComputeGsct_simple(nn.Module):

    def __init__(self, freq, is_attenuation):
        super().__init__()
        self.freq = freq
        self.pai = 3.1415926535897931
        self.c = 2.99792458e8
        self.omega = 2 * self.pai * freq
        self.mu0 = 4.0 * self.pai * 1.0e-07
        self.epsilon0 = 1.0e0 / (self.c ** 2 * self.mu0)
        self.is_attenuation = is_attenuation

    def forward(self, x):
        gsr_real = x[0][:, :, 0] # x[0]: batch, n_points, 2
        gsr_imag = x[0][:, :, 1]
        grf_real = x[1][:, :, 0]
        grf_imag = x[1][:, :, 1]
        # ckai_real = x[2][:, :, 1] # pazhou 1219
        ckai_imag = x[2][:, :, 1]
        ckai_real = x[2][:, :, 0]


        gsr = (gsr_real + 1j * gsr_imag).clone().to(torch.complex64)
        grf = (grf_real + 1j * grf_imag).clone().to(torch.complex64)
        ckai = (ckai_real + 1j * ckai_imag).clone().to(torch.complex64)

        Gsct = gsr * grf * ckai # cube feko bar


        return Gsct

class AttentionLayer(nn.Module):
    def __init__(self, in_dim):
        super(AttentionLayer, self).__init__()
        self.q = nn.Linear(in_dim, in_dim)
        self.k = nn.Linear(in_dim, in_dim)
        self.v = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch_size, dims, num_points)
        returns: weighted features of shape (batch_size, dims, num_points)
        """
        # (batch_size, dims, num_points) -> (batch_size, num_points, dims) after transpose
        q = self.q(x.transpose(1, 2))  # (batch_size, num_points, dims)
        k = self.k(x.transpose(1, 2))  # (batch_size, num_points, dims)
        v = self.v(x.transpose(1, 2))  # (batch_size, num_points, dims)

        # Attention matrix multiplication (q * k^T)
        attn = torch.bmm(q, k.transpose(1, 2))  # (batch_size, num_points, num_points)
        attn = self.softmax(attn)  # Normalize the attention scores

        # Weighted sum of values (attn * v)
        out = torch.bmm(attn, v)  # (batch_size, num_points, dims)

        # Return output in the original input shape (batch_size, dims, num_points)
        return out.transpose(1, 2)  # (batch_size, dims, num_points)

# d_model = 96 # 512  # Embedding Size  Embedding后的长度
# # d_ff = 2048  # FeedForward dimension  前馈神经网络的中间维度
# d_k = d_v = 32  # dimension of K(=Q), V
# # n_layers = 6  # number of Encoder of Decoder Layer   Encoder和Decoder N的个数
# n_heads = 2

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()

#     def forward(self, Q, K, V): #, attn_mask):
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
#         # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
#         attn = nn.Softmax(dim=-1)(scores)
#         context = torch.matmul(attn, V)
#         return context, attn

# class MultiHeadAttention1(nn.Module):
#     def __init__(self):
#         super(MultiHeadAttention1, self).__init__()



#         self.W_Q = nn.Linear(d_model, d_k * n_heads)
#         self.W_K = nn.Linear(d_model, d_k * n_heads)
#         self.W_V = nn.Linear(d_model, d_v * n_heads)
#         self.linear = nn.Linear(n_heads * d_v, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, Q, K, V): #, attn_mask):
#         # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
#         residual, batch_size = Q, Q.size(0)
#         # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#         q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
#         k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
#         v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

#         # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

#         # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
#         context, attn = ScaledDotProductAttention()(q_s, k_s, v_s) #, attn_mask)
#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
#         output = self.linear(context)
#         # return self.layer_norm(output + residual) # , attn # output: [batch_size x len_q x d_model]
#         return self.layer_norm(output)

# class MultiHeadAttention2(nn.Module):
#     def __init__(self):
#         super(MultiHeadAttention2, self).__init__()

#         self.W_Q = nn.Linear(d_model, d_k * n_heads)
#         self.W_K = nn.Linear(d_model, d_k * n_heads)
#         self.W_V = nn.Linear(d_model, d_v * n_heads)
#         self.linear = nn.Linear(n_heads * d_v, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, Q, K, V): #, attn_mask):
#         # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
#         residual, batch_size = Q, Q.size(0)
#         # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#         q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
#         k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
#         v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

#         # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

#         # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
#         context, attn = ScaledDotProductAttention()(q_s, k_s, v_s) #, attn_mask)
#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
#         output = self.linear(context)
#         # return self.layer_norm(output + residual) # , attn # output: [batch_size x len_q x d_model]
#         return self.layer_norm(output)


class GreenNet_simple(nn.Module):
    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.Gsr = GreenLayer_simple(self.freq)
        self.Grf = GreenLayer_simple(self.freq)
        self.Gsf = GreenLayer_simple(self.freq)
        self.Gsct = ComputeGsct_simple(self.freq, True)
        self.freq = freq
        self.pai = 3.1415926535897931
        self.c = 2.99792458e8
        self.omega = 2 * self.pai * freq
        self.mu0 = 4.0 * self.pai * 1.0e-07
        self.epsilon0 = 1.0e0 / (self.c ** 2 * self.mu0)

        self.sf_decay_1 = nn.Sequential(*[torch.nn.Conv1d(24*2, 16, 1), torch.nn.ReLU(), nn.BatchNorm1d(16), AttentionLayer(16)])  # 注意力机制

        self.sf_decay_2 = nn.Sequential(*[torch.nn.Conv1d(16 + 24*2, 64, 1), torch.nn.ReLU(), nn.BatchNorm1d(64),
                                          torch.nn.Conv1d(64, 128, 1), torch.nn.ReLU(), nn.BatchNorm1d(128),
                                          torch.nn.Conv1d(128, 256, 1), torch.nn.ReLU(), nn.BatchNorm1d(256),
                                          torch.nn.Conv1d(256, 512, 1), torch.nn.ReLU(), nn.BatchNorm1d(512),
                                          torch.nn.Conv1d(512, 256, 1), torch.nn.ReLU(), nn.BatchNorm1d(256),
                                          torch.nn.Conv1d(256, 128, 1), torch.nn.ReLU(), nn.BatchNorm1d(128),
                                          torch.nn.Conv1d(128, 64, 1), torch.nn.ReLU(), nn.BatchNorm1d(64),
                                          torch.nn.Conv1d(64, 16, 1), torch.nn.ReLU(), nn.BatchNorm1d(16),
                                          torch.nn.Conv1d(16, 2, 1)])

        self.point_feature = nn.Sequential(*[torch.nn.Conv1d(24*3 + 0, 32, 1), torch.nn.ReLU(), nn.BatchNorm1d(32), # , AttentionLayer(32), # 注意力机制
                                      torch.nn.Conv1d(32, 16, 1), torch.nn.ReLU()])

        self.point_feature_2 = nn.Sequential(*[torch.nn.Conv1d(24*1 + 0 + 16, 64, 1), torch.nn.ReLU(), nn.BatchNorm1d(64),   # 注意力机制
                                               torch.nn.Conv1d(64, 128, 1), torch.nn.ReLU(), nn.BatchNorm1d(128),
                                               torch.nn.Conv1d(128, 256, 1), torch.nn.ReLU(), nn.BatchNorm1d(256),
                                               torch.nn.Conv1d(256, 512, 1), torch.nn.ReLU(), nn.BatchNorm1d(512),
                                               torch.nn.Conv1d(512, 256, 1), torch.nn.ReLU(), nn.BatchNorm1d(256),
                                               torch.nn.Conv1d(256, 128, 1), torch.nn.ReLU(), nn.BatchNorm1d(128),
                                               torch.nn.Conv1d(128, 64, 1), torch.nn.ReLU(), nn.BatchNorm1d(64),
                                               torch.nn.Conv1d(64, 32, 1), torch.nn.ReLU(), nn.BatchNorm1d(32),
                                      torch.nn.Conv1d(32, 16, 1), torch.nn.ReLU(), nn.BatchNorm1d(16)])

        feature_dims = 16
        self.q = nn.Linear(feature_dims, feature_dims)  # 查询
        self.k = nn.Linear(feature_dims, feature_dims)  # 键
        self.v = nn.Linear(feature_dims, feature_dims)  # 值
        self.softmax = nn.Softmax(dim=-1)  # 用于归一化
        self.out_param = nn.Linear(feature_dims, 1)


    def forward(self, tloc, rloc, r, kai, n_scatters=128, if_Ez_theta=False, if_s_pattern=False, rcube_sh=None, deg=0, pcd=None, block_pcd=None):
        r = torch.concat(r, dim=1)
        kai = torch.concat(kai, dim=1) # / 10 # / 1000  # pazhou /10  # cube feko 不除
        kai_ones = torch.ones_like(kai)
        s = torch.tile(tloc.unsqueeze(1), (1, n_scatters, 1))
        f = torch.tile(rloc.unsqueeze(1), (1, n_scatters, 1))
        gsr = self.Gsr((s, r)) # batch, n_points, 2
        grf = self.Grf((r, f))
        # grf = self.Grf((r, f)) / (self.omega * self.mu0)
        gsf = self.Gsf((tloc, rloc))
        if block_pcd is not None:
            _, pcd_center, pcd_scale = pc_normalize(block_pcd[0,...])
            s_to_encode = (s - torch.tensor(pcd_center, device='cuda'))/torch.tensor(pcd_scale, device='cuda')
            f_to_encode = (f - torch.tensor(pcd_center, device='cuda')) / torch.tensor(pcd_scale, device='cuda')
            r_to_encode = (r - torch.tensor(pcd_center, device='cuda')) / torch.tensor(pcd_scale, device='cuda')

            s_pos_encoding = self.positional_encoding_3d(s_to_encode[:,0,:].unsqueeze(1), 4)
            f_pos_encoding = self.positional_encoding_3d(f_to_encode[:,0,:].unsqueeze(1), 4)
            Dsf = torch.cat([s_pos_encoding, f_pos_encoding], dim=-1)  # 将位置编码拼接到特征上
            Dsf = Dsf.permute(0, 2, 1)  # (batch_size, feature_dim, num_points)
            Dsf = self.sf_decay_1(Dsf)
            Dsf = Dsf.permute(0, 2, 1)
            Dsf = torch.cat([Dsf, s_pos_encoding, f_pos_encoding], dim=-1)  # 将位置编码拼接到特征上
            Dsf = Dsf.permute(0, 2, 1)
            gsf *= self.sf_decay_2(Dsf).squeeze(-1)
            gsf *= torch.tanh(self.sf_decay_2(Dsf)).squeeze(-1)  # -1 ~ +1


            # origin
            s_pos_encoding = self.positional_encoding_3d(s_to_encode, 4)
            f_pos_encoding = self.positional_encoding_3d(f_to_encode, 4)
            rcube_pos_encoding = self.positional_encoding_3d(r_to_encode, 4)
            csf = torch.concat([rcube_pos_encoding, s_pos_encoding, f_pos_encoding], dim=-1)
            csf = csf.permute(0, 2, 1)
            scat_feature = self.point_feature(csf)
            q = self.q(scat_feature.transpose(1,2))  # (batch_size, feature_dims, num_points)
            k = self.k(scat_feature.transpose(1,2))  # (batch_size, feature_dims, num_points)
            v = self.v(scat_feature.transpose(1,2))  # (batch_size, feature_dims, num_points)
            # 计算注意力分数  origin
            attn_weights = torch.bmm(q, k.transpose(1, 2))  # (batch_size, num_points, num_points)
            attn_weights = self.softmax(attn_weights)  # (batch_size, num_points, num_points)
            weighted_feats = torch.bmm(attn_weights, v)   # 自注意力机制  batch, points, feature_dims
            out_param = self.out_param(weighted_feats).squeeze(-1)

        g_per_sct = self.Gsct((gsr, grf, kai)) # batch, n_points  dtype=complex64

        g_per_sct *= out_param

        if if_s_pattern:
            vector_inc = s - r
            vector_sca = f - r
            norm_test = vector_inc / torch.norm(vector_inc, dim=-1, keepdim=True)  # [batch, 128, 1]
            norm_gt = vector_sca / torch.norm(vector_sca, dim=-1, keepdim=True)  # [batch, 4, 1]
            if rcube_sh is not None:
                g_per_sct = g_per_sct * pattern_in_sc(norm_test, norm_gt, rcube_sh, deg)
            elif pcd is not None:
                # pcd=[pcd_xyz, pcd_param])  # [pcd_xyz, pcd_areas, pcd_normals]
                pcd_nearest_param = find_near_pcd_param(scatter_xyz=r, pcd_xyz=pcd[0], pcd_param=pcd[1]) # 返回scatter最近的pcd param
                g_per_sct = g_per_sct * pattern_in_sc(norm_test, norm_gt, pcd_nearest_param=pcd_nearest_param)
            else:
                g_per_sct = g_per_sct * pattern_in_sc(norm_test, norm_gt)

        g_per_sct_p = torch.stack([g_per_sct.real, g_per_sct.imag]).permute(1,2,0)


        gsct = torch.sum(g_per_sct, dim=1) / g_per_sct.shape[1]
        gsct = torch.stack([gsct.real, gsct.imag]).permute(1, 0)
        gtot = (gsct + gsf) / 1000 # / 2 pazhou   # /20000 # * 200 
        
        if if_Ez_theta:
            return gtot, g_per_sct_p, r
        else:
            return gtot, kai, r


    def positional_encoding_3d(self, positions, L):
        """
        为每个点生成3D位置编码，位置编码使用正弦和余弦函数。
        positions: (batch_size, num_points, 3)，每个点的坐标
        d_model: 位置编码的维度，通常是特征维度
        """

        batch_size, num_points, _ = positions.size()

        # 生成 2^i 的频率序列: [2^0, 2^1, ..., 2^(L-1)]
        freq_bands = 2 ** torch.arange(L, dtype=torch.float32, device=positions.device)  # (L,)

        # 初始化位置编码矩阵 (batch_size, num_points, 6 * L)
        pe = torch.zeros(batch_size, num_points, 6 * L, device=positions.device)

        # 分别对 x, y, z 三个维度计算正弦和余弦
        for i in range(3):  # 对应 x, y, z
            for j, freq in enumerate(freq_bands):  # 对应不同频率
                pe[:, :, i * 2 * L + 2 * j] = torch.sin(freq * math.pi * positions[:, :, i])  # sin(2^j * π * p_i)
                pe[:, :, i * 2 * L + 2 * j + 1] = torch.cos(freq * math.pi * positions[:, :, i])  # cos(2^j * π * p_i)

        return pe


class Scatters_Green_simple(nn.Module):
    def __init__(self, opt):  # opt: [centroid, m, batch_size, freq, n_scatters]
        super().__init__()

        [self.centroid, self.m, self.batch_size, self.freq, self.n_scatters] = opt
        self.GreenNet = GreenNet_simple(freq=self.freq)

    def forward(self, tloc, rloc, if_Ez_theta=False, if_s_pattern=False, sh_module=False, deg=0, pcd=None, block_pcd=None, rcube_=None, ckai_=None):

        rcube_ = rcube_.repeat(tloc.shape[0], 1, 1)
        ckai = ckai_.unsqueeze(0).repeat(tloc.shape[0], 1, 1)
        # ckai = ckai_.unsqueeze(0).repeat(tloc.shape[0], 1, 2)
        rcube = pc_denormalize(rcube_, torch.tensor(self.centroid, dtype=torch.float32).to("cuda"),
                               torch.tensor(self.m, dtype=torch.float32).to("cuda"))

        if sh_module:
            rcube_sh = self.Sh_Module(tloc, rloc, rcube_, deg, self.n_scatters) # 每个散射点球谐
            gtot, gsct, _ = self.GreenNet.forward(tloc, rloc, [rcube], [ckai], self.n_scatters, if_Ez_theta, if_s_pattern, rcube_sh, deg, block_pcd=block_pcd)
        elif pcd:
            gtot, gsct, _ = self.GreenNet.forward(tloc, rloc, [rcube], [ckai], self.n_scatters, if_Ez_theta, if_s_pattern, pcd=pcd)
        elif block_pcd is not None:
            gtot, gsct, _ = self.GreenNet.forward(tloc, rloc, [rcube], [ckai], self.n_scatters, if_Ez_theta, if_s_pattern, block_pcd=block_pcd)
        else:
            gtot, gsct, _ = self.GreenNet.forward(tloc, rloc, [rcube], [ckai], self.n_scatters, if_Ez_theta, if_s_pattern)

        if if_Ez_theta:
            return [gtot, [rcube_, gsct], rcube, ckai]  # 信号；散射点坐标；散射点大小
        else:
            return [gtot, rcube_, rcube, ckai]




class MLP_Tx_Rx_E(nn.Module):
    def __init__(self, opt):  # opt: [centroid, m, batch_size, freq, n_scatters]
        super().__init__()
        [self.centroid, self.m, self.batch_size, self.freq, self.n_scatters] = opt
        self.MLP_Etot_module = nn.ModuleList(
            [nn.Linear(24*2, 128)] +  # Input: tloc + rloc (6 features)
            [nn.ReLU(), nn.Linear(128, 128)] * 4 +  # Hidden layers
            [nn.ReLU(), nn.Linear(128, 256)] +
            [nn.ReLU(), nn.Linear(256, 512)] +
            [nn.ReLU(), nn.Linear(512, 256)] +
            [nn.ReLU(), nn.Linear(256, 128)] +
            [nn.ReLU(), nn.Linear(128, 64)] +
            [nn.ReLU(), nn.Linear(64, 2)]  # Output: 2 features (e.g., real and imaginary parts)
        )
    # (inputs[0], inputs[1], Ez_theta, s_pattern, sh_module=if_sh_module, deg=deg, block_pcd=block_pcd, rcube_=rcube_, ckai_=ckai_)
    def forward(self, tloc, rloc, if_Ez_theta=False, if_s_pattern=False, sh_module=False, deg=0, block_pcd=None, rcube_=None, ckai_=None):
        # input_features = torch.cat([(tloc + torch.tensor([7, -6.5, -1.4], device='cuda')) / torch.tensor([4, 9, 2.8], device='cuda'), (rloc + torch.tensor([7, -6.5, -1.4], device='cuda')) / torch.tensor([4, 9, 2.8], device='cuda')], dim=1)

        _, pcd_center, pcd_scale = pc_normalize(block_pcd[0, ...])
        s_to_encode = (tloc - torch.tensor(pcd_center, device='cuda')) / torch.tensor(pcd_scale, device='cuda')
        f_to_encode = (rloc - torch.tensor(pcd_center, device='cuda')) / torch.tensor(pcd_scale, device='cuda')
        tloc = self.positional_encoding_3d(s_to_encode[:, :].unsqueeze(1), 4)
        rloc = self.positional_encoding_3d(f_to_encode[:, :].unsqueeze(1), 4)


        # tloc = self.positional_encoding_3d(tloc[:, :].unsqueeze(1), 4)
        # rloc = self.positional_encoding_3d(rloc[:, :].unsqueeze(1), 4)

        input_features = torch.cat((tloc, rloc), dim=-1)

        x = input_features.squeeze()
        for layer in self.MLP_Etot_module:
            x = layer(x)  # Apply each layer (linear + ReLU)

        # gtot output (predicted Etot)
        gtot = x

        rcube_ = torch.zeros([self.batch_size, self.n_scatters, 3]).to('cuda')
        ckai = torch.zeros([self.batch_size, self.n_scatters, 2]).to('cuda')

        return [gtot, rcube_, rcube_, ckai]

    def positional_encoding_3d(self, positions, L):
        """
        为每个点生成3D位置编码，位置编码使用正弦和余弦函数。
        positions: (batch_size, num_points, 3)，每个点的坐标
        d_model: 位置编码的维度，通常是特征维度
        """

        batch_size, num_points, _ = positions.size()

        # 生成 2^i 的频率序列: [2^0, 2^1, ..., 2^(L-1)]
        freq_bands = 2 ** torch.arange(L, dtype=torch.float32, device=positions.device)  # (L,)

        # 初始化位置编码矩阵 (batch_size, num_points, 6 * L)
        pe = torch.zeros(batch_size, num_points, 6 * L, device=positions.device)

        # 分别对 x, y, z 三个维度计算正弦和余弦
        for i in range(3):  # 对应 x, y, z
            for j, freq in enumerate(freq_bands):  # 对应不同频率
                pe[:, :, i * 2 * L + 2 * j] = torch.sin(freq * math.pi * positions[:, :, i])  # sin(2^j * π * p_i)
                pe[:, :, i * 2 * L + 2 * j + 1] = torch.cos(freq * math.pi * positions[:, :, i])  # cos(2^j * π * p_i)

        return pe

class Model_select():
    def __init__(self, model_type):
        self.model_type = model_type
        if self.model_type == 'Scatters_Green_simple':
            self.model = Scatters_Green_simple
        elif self.model_type == 'MLP_Tx_Rx_E':
            self.model = MLP_Tx_Rx_E
        else:
            raise ValueError('model_type error')


    def __call__(self, *args, **kwargs):
        # 在__call__方法中返回相应模型
        return self.model(*args, **kwargs)






