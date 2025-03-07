import torch.nn as nn
import torch, torchvision

# Source: https://amaarora.github.io/posts/2020-09-13-unet.html
class QUBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pad   = nn.ReplicationPad2d(1)  # pad data to avoid edge artefacts
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        x = self.pad(x)  
        x = self.relu(self.conv1(x))
        return self.relu(self.conv2(x))


class QUEncoder(nn.Module):
    def __init__(self, in_channels=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([QUBlock(in_channels[i], in_channels[i+1]) for i in range(len(in_channels)-1)])
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features
    
class QUDecoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) 
            for i in range(len(chs)-1)
        ])
        self.dec_blocks = nn.ModuleList(
            [QUBlock(chs[i], chs[i+1]) 
            for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class QUNet(nn.Module):
    
    def __init__(self, in_channels=3, base_channels=64, depth=4, num_class=1, retain_dim=True, out_sz=(256,256)):
        super().__init__()
        enc_chs, dec_chs = self.generate_channels(in_channels, base_channels, depth)
        self.encoder     = QUEncoder(enc_chs)
        self.decoder     = QUDecoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz
        self.num_class = num_class

    def generate_channels(self, in_ch, base_ch, depth):
        enc_chs = [in_ch] + [base_ch * (2**i) for i in range(depth)]
        dec_chs = enc_chs[::-1][:-1]
        return enc_chs, dec_chs

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = nn.functional.interpolate(out, size=self.out_sz, mode="bilinear", align_corners=False)
        return out