from PIL import Image
from torchvision import transforms
import torch
from torch import nn

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)

class Down(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.seq(x)

class Up(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, stride=2, kernel_size=2)
        self.conv  = DoubleConv(in_channels, out_channels)
    def forward(self, x, from_residual):
        scaled_up = self.up(x)
        concatenated = torch.concat([from_residual, scaled_up], dim=1)
        return self.conv(concatenated)

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super(UNet, self).__init__()
        self.increase_channels = DoubleConv(in_channels, base)
        self.down1 = Down(base    , base * 2 )
        self.down2 = Down(base * 2, base * 4 )
        self.down3 = Down(base * 4, base * 8 )
        self.down4 = Down(base * 8, base * 16)

        self.up4   = Up(base * 16, base * 8 )
        self.up3   = Up(base * 8 , base * 4 )
        self.up2   = Up(base * 4 , base * 2 )
        self.up1   = Up(base * 2 , base     )

        self.final = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        layer_1 = self.increase_channels(x)
        layer_2 = self.down1(layer_1)
        layer_3 = self.down2(layer_2)
        layer_4 = self.down3(layer_3)
        layer_5 = self.down4(layer_4)

        layer_4 = self.up4(layer_5, layer_4)
        layer_3 = self.up3(layer_4, layer_3)
        layer_2 = self.up2(layer_3, layer_2)
        layer_1 = self.up1(layer_2, layer_1)

        return self.final(layer_1)


if __name__ == "__main__":
    # UNet Test
    model = UNet()

    img = Image.open("image.jpg").convert("RGB")
    img1 = Image.open("image1.jpg").convert("RGB")
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img)
    img1_tensor = to_tensor(img1)
    stacked = torch.stack([img_tensor, img1_tensor])
    print(model(stacked).shape)
