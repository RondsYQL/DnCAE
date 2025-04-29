from collections import OrderedDict
import torch.nn as nn
import torch

# hello
# DnCAE模型的结构
# picture_channel代表图像通道数N，channels代表卷积层通道数C
picture_channel = 3
channels = 64

class SubNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_layers = 4):
        super().__init__()
        self.num_of_layers = num_of_layers
        for i in range(1, num_of_layers+1):
            setattr(self, 'layer'+str(i), nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride = 1, padding=1, bias=True),
                # nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            ))
                
    def forward(self, x):
        for i in range(1, self.num_of_layers+1):
            layer = getattr(self, 'layer'+str(i))
            x = layer(x)
        return x

def make_layer(block):
    layers = []
    for layer_name, param in block.items():
        if 'dconv' in layer_name:
            layers.append(
                nn.ConvTranspose2d(in_channels=param[0],
                                   out_channels=param[1],
                                   kernel_size=param[2],
                                   stride=param[3],
                                   padding=param[4]))
        else:
            layers.append(
                nn.Conv2d(in_channels=param[0],
                          out_channels=param[1],
                          kernel_size=param[2],
                          stride=param[3],
                          padding=param[4]))
        if 'end' not in layer_name:
            # layers.append(nn.BatchNorm2d(param[1]))
            layers.append(nn.LeakyReLU(inplace=True))
    return nn.Sequential(*layers)

encoder_params = [
    [
        OrderedDict({"conv": [picture_channel,  channels, 3, 2, 1]}),
        OrderedDict({"conv": [channels, channels, 3, 2, 1]}),
        OrderedDict({"conv": [channels, channels, 3, 2, 1]})
    ],

    [
        SubNet(channels, channels),
        SubNet(channels, channels),
        SubNet(channels, channels)
    ]
]

decoder_params = [
    [
        OrderedDict({"dconv": [channels*2, channels, 4, 2, 1]}),
        OrderedDict({"dconv": [channels*2, channels, 4, 2, 1]}),
        OrderedDict(
            {"dconv1": [channels*2, channels,  4, 2, 1], 
             "conv2": [channels, channels//2, 3, 1, 1],
             "conv3_end": [channels//2, picture_channel, 3, 1, 1]})
    ],

    [
        SubNet(channels, channels),
        SubNet(channels, channels),
        None
    ]
]

class EncoderNet(nn.Module):
    def __init__(self, sample_nets, sub_nets):
        super().__init__()
        self.layer_nums = len(sample_nets)
        for i, (sample_net, sub_net) in enumerate(zip(sample_nets, sub_nets), 1):
            setattr(self, 'sample_net'+str(i), make_layer(sample_net))
            setattr(self, 'sub_net'+str(i), sub_net)

    def forward(self, x):
        out = []
        for i in range(1, self.layer_nums+1):
            sample_net = getattr(self, 'sample_net'+str(i))
            sub_net = getattr(self, 'sub_net'+str(i))
            x = sample_net(x)
            out.append(x)
            x = sub_net(x)
        out.append(x)
        return out

class DecoderNet(nn.Module):
    def __init__(self, sample_nets, sub_nets):
        super().__init__()
        self.layer_nums = len(sample_nets)
        for i, (sample_net, sub_net) in enumerate(zip(sample_nets, sub_nets), 1):
            setattr(self, 'sample_net'+str(i), make_layer(sample_net))
            setattr(self, 'sub_net'+str(i), sub_net)

    def forward(self, temp):
        x = temp.pop()
        for i in range(1, self.layer_nums+1):
            x = torch.cat([x, temp.pop()], dim=1)
            sample_net = getattr(self, 'sample_net'+str(i))
            sub_net = getattr(self, 'sub_net'+str(i))
            x = sample_net(x)
            if sub_net is not None:
                x = sub_net(x)
        return x


class EDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderNet(encoder_params[0], encoder_params[1])
        self.decoder = DecoderNet(decoder_params[0], decoder_params[1])

    def forward(self, x):
        state = self.encoder(x)
        out = self.decoder(state)
        return out
