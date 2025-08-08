import torch
from torch import nn
import torch.nn.functional as F

from ..utils import init_weights


class EqualizedLR(nn.Module):
    # "weight" scaling at run time !!! sweeet
    def __init__(self):
        super().__init__()
    
    def _init_weights(self):
        nn.init.normal_(self.module.weight)
        if self.module.bias is not None:
            nn.init.zeros_(self.module.bias)

    def forward(self, x):
        return self.module(x * self.scale)


class EqualizedLinear(EqualizedLR):
    def __init__(self, in_dim, out_dim, gain=2**-.5):
        super().__init__()
        self.module = nn.Linear(in_dim, out_dim)
        self.gain = gain
        self.fan_in = self.module.weight[0].numel()
        self.scale = self.gain * self.fan_in**-.5

class EqualizedConv(EqualizedLR):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, gain=2**.5):
        super().__init__()
        self.module = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.fan_in = in_channels * kernel_size**2
        self.scale = gain * self.fan_in**-.5



class Mapper(nn.Module):
    def __init__(self, z_dim, w_dim, depth=4):
        super().__init__()
        self.eps = 1e-8
        self.layers = []
        for _ in range(depth):
            self.layers += [
                EqualizedLinear(z_dim, w_dim),
                nn.LeakyReLU(.2),
            ]
            z_dim = w_dim
            
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, z):
        return self.layers(
            z / (torch.norm(z, dim=1, keepdim=True) + self.eps)
        )



class AdaIN(nn.Module):
    ...



class ModConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.style_projector = EqualizedLinear(style_dim, in_channels)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.eps = 1e-8
        
    def forward(self, x, style_vector):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        b, c, h, w = x.shape
        x = x.reshape(1, b*c, h, w)
        
        style = self.style_projector(style_vector).view(b, 1, self.in_channels, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            demod_coef = torch.rsqrt((weight ** 2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod_coef.view(b, self.out_channels, 1, 1, 1)

        weight = weight.reshape(
            b * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        out = F.conv2d(x, weight, padding=self.kernel_size//2, groups=b)
        return out.view(b, self.out_channels, h, w)


class NoiseInjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        b, _, h, w = x.shape
        noise = torch.randn(b, 1, h, w, device=x.device)
        return x + self.weight * noise


class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=False):
        super().__init__()
        self.conv1 = ModConv(in_channels, out_channels, 3, style_dim, upsample=upsample)
        self.noise_injector1 = NoiseInjector(out_channels)
        
        self.conv2 = ModConv(out_channels, out_channels, 3, style_dim)
        self.noise_injector2 = NoiseInjector(out_channels)
        
        self.act = nn.LeakyReLU(.2)
        
        
    def forward(self, x, w1, w2):
        x = self.conv1(x, w1)
        x = self.noise_injector1(x)
        x = self.act(x)
        
        x = self.conv2(x, w2)
        x = self.noise_injector2(x)
        return self.act(x)
    

class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.conv = ModConv(in_channels, 3, 1, style_dim, demodulate=False)

    def forward(self, x, w):
        return self.conv(x, w)
    

class StyleGANG(nn.Module):
    def __init__(self, channels, lat_dim=256, w_dim=256, init_channels=256):
        super().__init__()
        self.lat_dim = lat_dim
        self.w_dim = w_dim
        
        self.mapper = Mapper(lat_dim, w_dim)
        self.init_canvas = nn.Parameter(torch.randn(1, init_channels, 4, 4))

        self.blocks = nn.ModuleList()
        self.rgbs = nn.ModuleList()

        in_channels = init_channels
        for i, out_channels in enumerate(channels):
            upsample = i > 0
            self.blocks.append(
                StyleBlock(in_channels, out_channels, w_dim, upsample=upsample)
            )

            self.rgbs.append(ToRGB(out_channels, w_dim))
            in_channels = out_channels
        
        self.num_styles = len(self.blocks) * 2
        init_weights(self, init_scheme="kaiming")

        for module in self.modules():
            if isinstance(module, EqualizedLR):
                module._init_weights()

    
    def synthesis(self, w):
        b = w.size(0)
        x = self.init_canvas.expand(b, *([-1]*3))
        rgb = None

        for i, (block, to_rgb) in enumerate(zip(self.blocks, self.rgbs)):
            w1 = w[:, 2*i]
            w2 = w[:, 2*i + 1]
            x = block(x, w1, w2)

            if rgb is None:
                rgb = to_rgb(x, w2)
            else:
                rgb = F.interpolate(rgb, scale_factor=2, mode="nearest")
                rgb = rgb + to_rgb(x, w2)

        return torch.tanh(rgb)


    def forward(self, z):
        w = self.mapper(z)
        return self.synthesis(w)



class BatchSTD(nn.Module):
    def forward(self, x):
        _, _, h, w = x.shape
        std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + 1e-7
        ).mean(dim=(2, 3), keepdim=True)

        std = torch.nan_to_num(std, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.cat([
            x,
            std.expand(-1, 1, h, w)
        ], dim=1)



class ResDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv(in_channels, in_channels, 3, padding=1)
        self.conv2 = EqualizedConv(in_channels, out_channels, 3, padding=1)
        self.act = nn.LeakyReLU(.2)
        
        self.res = EqualizedConv(in_channels, out_channels, 1, padding=0)
        self.avg = nn.AvgPool2d(2)
        
    def forward(self, x):
        residual = self.res(x)
        residual = self.avg(residual)
        
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.avg(x)
        
        return (x + residual) * 2.**-.5


class StyleGAND(nn.Module):
    def __init__(self, channels):
        super().__init__()
        in_channels = channels[0]
        self.bstd = BatchSTD()
        blocks = [EqualizedConv(3, in_channels, kernel_size=3, padding=1)]

        for out_channels in channels:
            blocks.append(
                ResDownSampleBlock(in_channels, out_channels)
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            EqualizedConv(out_channels + 1, out_channels, 3, padding=1),
            nn.LeakyReLU(.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 1)
        )
        
        init_weights(self, init_scheme="kaiming")

        for module in self.modules():
            if isinstance(module, EqualizedLR):
                module._init_weights()



    def forward(self, x):
        x = self.blocks(x)
        x = self.bstd(x)
        return self.head(x)