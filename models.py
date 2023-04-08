import torch.nn as nn

class EEGNetELU(nn.Module):
    def __init__(self):
        super(EEGNetELU, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=8,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(736,2,bias=True)
        )

    def forward(self, x):
        out = self.firstConv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out) 
        out = self.classify(out)
        return out
    
class EEGNetReLU(nn.Module):
    def __init__(self):
        super(EEGNetReLU, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=8,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(736,2,bias=True)
        )

    def forward(self, x):
        out = self.firstConv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out) 
        out = self.classify(out)
        return out

class EEGNetLeakyReLU(nn.Module):
    def __init__(self):
        super(EEGNetLeakyReLU, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=8,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.06),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.06),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(736,2,bias=True)
        )

    def forward(self, x):
        out = self.firstConv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out) 
        out = self.classify(out)
        return out
    
class DeepConvNetELU(nn.Module):
    def __init__(self):
        super(DeepConvNetELU, self).__init__()
        self.model = nn.Sequential(
            # Conv2d(1, 25, kernel_size=(1,5),padding='VALID',bias=False),
            # Conv2d(25, 25, kernel_size=(2,1), padding='VALID',bias=False),
            nn.Conv2d(1, 25, kernel_size=(1,5), padding='valid'),
            nn.Conv2d(25, 25, kernel_size=(2,1), padding='valid'),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            # Conv2d(25, 50, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(25, 50, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            # Conv2d(50, 100, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(50, 100, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            # Conv2d(100, 200, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(100, 200, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            nn.Flatten(),
            nn.Linear(8600,2,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out
    
class DeepConvNetReLU(nn.Module):
    def __init__(self):
        super(DeepConvNetReLU, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5), padding='valid'),
            nn.Conv2d(25, 25, kernel_size=(2,1), padding='valid'),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(25, 50, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(50, 100, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(100, 200, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            nn.Flatten(),
            nn.Linear(8600,2,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out
    
class DeepConvNetLeakyReLU(nn.Module):
    def __init__(self):
        super(DeepConvNetLeakyReLU, self).__init__()
        self.model = nn.Sequential(
            # Conv2d(1, 25, kernel_size=(1,5),padding='VALID',bias=False),
            # Conv2d(25, 25, kernel_size=(2,1), padding='VALID',bias=False),
            nn.Conv2d(1, 25, kernel_size=(1,5), padding='valid'),
            nn.Conv2d(25, 25, kernel_size=(2,1), padding='valid'),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            # Conv2d(25, 50, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(25, 50, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            # Conv2d(50, 100, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(50, 100, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            # Conv2d(100, 200, kernel_size=(1,5),padding='VALID',bias=False),
            nn.Conv2d(100, 200, kernel_size=(1,5), padding='valid'),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),

            nn.Flatten(),
            nn.Linear(8600,2,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out