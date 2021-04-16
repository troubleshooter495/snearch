import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential()
        self.seq.add_module('conv-1', nn.Conv2d(3, 64, (5, 5), 1, 2))
        self.seq.add_module('relu-1', nn.ReLU())
        self.seq.add_module('pool-1', nn.MaxPool2d(3, 2))
        self.seq.add_module('conv-2', nn.Conv2d(64, 128, (3, 3), 1, 1))
        self.seq.add_module('relu-2', nn.ReLU())
        self.seq.add_module('pool-2', nn.MaxPool2d(3, 2))
        self.seq.add_module('conv-3', nn.Conv2d(128, 192, (3, 3), 1, 1))
        self.seq.add_module('relu-3', nn.ReLU())
        self.seq.add_module('conv-4', nn.Conv2d(192, 128, (3, 3), 1, 1))
        self.seq.add_module('relu-4', nn.ReLU())
        self.seq.add_module('conv-5', nn.Conv2d(128, 128, (3, 3), 1, 1))
        self.seq.add_module('relu-5', nn.ReLU())
        self.seq.add_module('pool-5', nn.MaxPool2d(3, 2))

        self.seq.add_module('flat', nn.Flatten())

        self.seq.add_module('drop-1', nn.Dropout())
        self.seq.add_module('lin-1', nn.Linear(1152, 512))
        self.seq.add_module('lin-relu-1', nn.ReLU())
        self.seq.add_module('drop-2', nn.Dropout())
        self.seq.add_module('lin-2', nn.Linear(512, 256))
        self.seq.add_module('lin-relu-2', nn.ReLU())
        self.seq.add_module('lin-3', nn.Linear(256, 2))

    def forward(self, x):
        return self.seq(x)


def load_classifier(path):
    return torch.load("../" + path, map_location=torch.device('cpu'))
