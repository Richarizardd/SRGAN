import math

import torch.nn.functional as F
from torch import nn
from torch import cat

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()

        self.growth_rate = 16
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4)
        )
        self.block2 = DenseBlock(16, self.growth_rate)
        self.block3 = TransitionBlock(16,1)
        self.block4 = DenseBlock(16, self.growth_rate)
        self.block5 = TransitionBlock(16,1)
        self.block6 = DenseBlock(16, self.growth_rate)
        self.block11 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1, padding=0, stride=1)
        )
        

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block11 = self.block11(block6)
        out = F.sigmoid(block11)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(TransitionBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels* up_scale ** 2, stride=2, kernel_size=3, padding=0)


    def forward(self, x):
        x = self.deconv(x)
        return x



class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate):
        super(DenseBlock, self).__init__()

        self.relu1b = nn.PReLU()
        self.relu2b = nn.PReLU()
        self.relu3b = nn.PReLU()
        self.relu4b = nn.PReLU()
        self.relu5b = nn.PReLU()

        self.relu1a = nn.PReLU()
        self.relu2a = nn.PReLU()
        self.relu3a = nn.PReLU()
        self.relu4a = nn.PReLU()
        self.relu5a = nn.PReLU()

        self.bn1a = nn.BatchNorm2d(channels*1)
        self.bn2a = nn.BatchNorm2d(channels*2)
        self.bn3a = nn.BatchNorm2d(channels*3)
        self.bn4a = nn.BatchNorm2d(channels*4)
        self.bn5a = nn.BatchNorm2d(channels*5)

        self.bn1b = nn.BatchNorm2d(channels)
        self.bn2b = nn.BatchNorm2d(channels)
        self.bn3b = nn.BatchNorm2d(channels)
        self.bn4b = nn.BatchNorm2d(channels)
        self.bn5b = nn.BatchNorm2d(channels)

        self.conv1a = nn.Conv2d(channels*1, channels, stride=1, padding=0, kernel_size=1)
        self.conv2a = nn.Conv2d(channels*2, channels, stride=1, padding=0, kernel_size=1)
        self.conv3a = nn.Conv2d(channels*3, channels, stride=1, padding=0, kernel_size=1)
        self.conv4a = nn.Conv2d(channels*4, channels, stride=1, padding=0, kernel_size=1)
        self.conv5a = nn.Conv2d(channels*5, channels, stride=1, padding=0, kernel_size=1)

        self.conv1b = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv2b = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv3b = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv4b = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv5b = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)

        self.dropout = nn.Dropout(p=0.25)


    def forward(self, x):
        nodes = []
        nodes.append(x)

        cocat_node = cat(tuple(nodes),1)
        dense = self.bn1a(cocat_node)
        dense = self.relu1a(dense)
        dense = self.conv1a(dense)
        dense = self.bn1b(dense)
        dense = self.relu1b(dense)
        dense = self.conv1b(dense)
        dense = self.dropout(dense)
        nodes.append(dense)

        cocat_node = cat(tuple(nodes),1)
        dense = self.bn2a(cocat_node)
        dense = self.relu2a(dense)
        dense = self.conv2a(dense)
        dense = self.bn2b(dense)
        dense = self.relu2b(dense)
        dense = self.conv2b(dense)
        dense = self.dropout(dense)
        nodes.append(dense)

        cocat_node = cat(tuple(nodes),1)
        dense = self.bn3a(cocat_node)
        dense = self.relu3a(dense)
        dense = self.conv3a(dense)
        dense = self.bn3b(dense)
        dense = self.relu3b(dense)
        dense = self.conv3b(dense)
        dense = self.dropout(dense)
        nodes.append(dense)

        cocat_node = cat(tuple(nodes),1)
        dense = self.bn4a(cocat_node)
        dense = self.relu4a(dense)
        dense = self.conv4a(dense)
        dense = self.bn4b(dense)
        dense = self.relu4b(dense)
        dense = self.conv4b(dense)
        dense = self.dropout(dense)
        nodes.append(dense)

        cocat_node = cat(tuple(nodes),1)
        dense = self.bn5a(cocat_node)
        dense = self.relu5a(dense)
        dense = self.conv5a(dense)
        dense = self.bn5b(dense)
        dense = self.relu5b(dense)
        dense = self.conv5b(dense)
        dense = self.dropout(dense)
    
        return dense



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        # Desired Size: [6, 3, 88, 88]
        # print("netD Input Tensor Size: ", x.size())
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))