class SeNet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SeNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # b,c,h,w -> b,c,1,1
        avg = self.avg_pool(x).view([b, c])
        # b,c -> b,c,1,1
        fc = self.fc(avg).view([b, c, 1, 1])
        return x * fc
