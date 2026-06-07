import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(F_int))
        self.W_l = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi

class DeepDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.block(x)

class VasculatureDualHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = convnext.features
        
        self.box_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(768, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
        nn.init.constant_(self.box_head[-1].weight, 0.0)
        nn.init.constant_(self.box_head[-1].bias, 0.0)
        
        self.upconv4 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=384, F_l=384, F_int=192)
        self.dec4 = DeepDecoderBlock(in_channels=768, out_channels=384)

        self.upconv3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=192, F_l=192, F_int=96)
        self.dec3 = DeepDecoderBlock(in_channels=384, out_channels=192)

        self.upconv2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=96, F_l=96, F_int=48)
        self.dec2 = DeepDecoderBlock(in_channels=192, out_channels=96)

        self.final_upconv = nn.ConvTranspose2d(96, 32, kernel_size=4, stride=4)
        self.mask_head = nn.Conv2d(32, 1, kernel_size=1) 
        
    def forward(self, x):
        e1 = self.features[0:2](x)          
        e2 = self.features[2:4](e1)         
        e3 = self.features[4:6](e2)         
        bottleneck = self.features[6:8](e3) 

        box_preds = self.box_head(bottleneck)

        d4_up = self.upconv4(bottleneck)
        if d4_up.shape[-2:] != e3.shape[-2:]:
            d4_up = F.interpolate(d4_up, size=e3.shape[-2:], mode='bilinear', align_corners=True)
        e3_att = self.att4(g=d4_up, x=e3)
        d4 = self.dec4(torch.cat([d4_up, e3_att], dim=1))

        d3_up = self.upconv3(d4)
        if d3_up.shape[-2:] != e2.shape[-2:]:
            d3_up = F.interpolate(d3_up, size=e2.shape[-2:], mode='bilinear', align_corners=True)
        e2_att = self.att3(g=d3_up, x=e2)
        d3 = self.dec3(torch.cat([d3_up, e2_att], dim=1))

        d2_up = self.upconv2(d3)
        if d2_up.shape[-2:] != e1.shape[-2:]:
            d2_up = F.interpolate(d2_up, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        e1_att = self.att2(g=d2_up, x=e1)
        d2 = self.dec2(torch.cat([d2_up, e1_att], dim=1))
        
        d0 = self.final_upconv(d2)
        mask_preds = self.mask_head(d0)

        return box_preds, mask_preds