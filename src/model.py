import torch
import torch.nn as nn

"""
Max-Feature-Map activation layer
Splits channels into two halves and takes element-wise maximum
"""
class MFM(nn.Module):
    def forward(self, x):
        c = x.size(1) // 2
        return torch.max(x[:, :c], x[:, c:])

"""
Light Convolutional Neural Network architecture
"""
class LCNN(nn.Module):
    def __init__(self, c_in=1, n_cls=2):
        super(LCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=5, stride=1, padding=2),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.5)
        self.bn_final = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, n_cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.bn_final(x)
        x = self.fc(x)
        return x

