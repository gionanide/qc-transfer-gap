"""
2D U-Net for ACDC cardiac segmentation (4 classes: bg, RV, MYO, LV).
Thin wrapper around MONAI's UNet so the rest of the code stays clean and the
uncertainty head can be added later without touching training logic.
"""
import torch.nn as nn
from monai.networks.nets import UNet

import config as C


def build_unet():
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=C.NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,            # kept >0 so MC-Dropout baseline is available later
    )


if __name__ == "__main__":
    import torch
    net = build_unet()
    x = torch.randn(2, 1, *C.PATCH_SIZE)
    y = net(x)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"output: {tuple(y.shape)}  params: {n_params/1e6:.2f}M")
