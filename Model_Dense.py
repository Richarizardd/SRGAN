import math

import torch.nn.functional as F
from torch import nn
from torch import cat

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()

        self.growth_rate = 16
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1)
        )
        self.block2 = DenseBlock(16, self.growth_rate)
        self.block3 = TransitionBlock(16,1)
        self.block4 = DenseBlock(16, self.growth_rate)
        self.block5 = TransitionBlock(16,1)
        self.block6 = DenseBlock(16, self.growth_rate)
        self.block11 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1, padding=0, stride=1)
        )
        

    def forward(self, x):
        block1 = self.block1(x)
        print "block1", block1.size()
        print

        block2 = self.block2(block1)
        print "block2", block2.size()
        print

        block3 = self.block3(block2)
        print "block3", block3.size()
        print
        
        block4 = self.block4(block3)
        print "block4", block4.size()
        print
        
        block5 = self.block5(block4)
        print "block5", block5.size()
        print
        
        block6 = self.block6(block5)
        print "block6", block6.size()
        print

        block11 = self.block11(block6)
        print block11.size()

        out = F.sigmoid(block11)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(TransitionBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels*up_scale**2, stride=2, kernel_size=3, padding=0)


    def forward(self, x):
        x = self.deconv(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate):
        super(DenseBlock, self).__init__()

        self.relu1b = nn.ReLU()
        self.relu2b = nn.ReLU()
        self.relu3b = nn.ReLU()
        self.relu4b = nn.ReLU()
        self.relu5b = nn.ReLU()

        self.relu1a = nn.ReLU()
        self.relu2a = nn.ReLU()
        self.relu3a = nn.ReLU()
        self.relu4a = nn.ReLU()
        self.relu5a = nn.ReLU()

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

        self.dropout1 = nn.Dropout(p=0.20)
        self.dropout2 = nn.Dropout(p=0.20)
        self.dropout3 = nn.Dropout(p=0.20)
        self.dropout4 = nn.Dropout(p=0.20)
        self.dropout5 = nn.Dropout(p=0.20)


    def forward(self, x):
        nodes = []
        nodes.append(x)

        concat_node = cat(tuple(nodes),1)
        print concat_node.size()
        dense = self.bn1a(concat_node)
        dense = self.relu1a(dense)
        dense = self.conv1a(dense)
        dense = self.bn1b(dense)
        dense = self.relu1b(dense)
        dense = self.conv1b(dense)
        dense = self.dropout1(dense)
        nodes.append(dense)

        concat_node = cat(tuple(nodes),1)
        print concat_node.size()
        dense = self.bn2a(concat_node)
        dense = self.relu2a(dense)
        dense = self.conv2a(dense)
        dense = self.bn2b(dense)
        dense = self.relu2b(dense)
        dense = self.conv2b(dense)
        dense = self.dropout2(dense)
        nodes.append(dense)

        concat_node = cat(tuple(nodes),1)
        print concat_node.size()
        dense = self.bn3a(concat_node)
        dense = self.relu3a(dense)
        dense = self.conv3a(dense)
        dense = self.bn3b(dense)
        dense = self.relu3b(dense)
        dense = self.conv3b(dense)
        dense = self.dropout3(dense)
        nodes.append(dense)

        concat_node = cat(tuple(nodes),1)
        print concat_node.size()
        dense = self.bn4a(concat_node)
        dense = self.relu4a(dense)
        dense = self.conv4a(dense)
        dense = self.bn4b(dense)
        dense = self.relu4b(dense)
        dense = self.conv4b(dense)
        dense = self.dropout4(dense)
        nodes.append(dense)
        
        concat_node = cat(tuple(nodes),1)
        print concat_node.size()
        dense = self.bn5a(concat_node)
        dense = self.relu5a(dense)
        dense = self.conv5a(dense)
        dense = self.bn5b(dense)
        dense = self.relu5b(dense)
        dense = self.conv5b(dense)
        dense = self.dropout5(dense)
        nodes.append(dense)
        print
        
    
        return dense
    
#input_size = (1,3,32,32)
#x = Variable(torch.rand(*input_size))
#test = Generator(4)
#_ = test(x)



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