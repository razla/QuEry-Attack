import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64, 128], fc_dims=[512, 128, 10]):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=3),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(fc_dims[0], fc_dims[1])
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc_dims[1], fc_dims[2])

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.conv3(out)
        out = self.max_pool(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out