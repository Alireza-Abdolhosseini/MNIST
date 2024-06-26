import torch.nn as nn
from torch import relu

class Net(nn.Module):

    def __init__(self, layers):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.hidden = nn.ModuleList()
        self.cnn = nn.ModuleList()
        self.maxpooling = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.cnn.append(nn.Conv2d(in_channels=1, # Gray pictures
                                        out_channels=32,
                                        kernel_size=5,
                                        stride=1,
                                        padding=0))


        self.maxpooling.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.bn.append(nn.BatchNorm2d(32))

        self.cnn.append(nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=0))


        self.maxpooling.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.bn.append(nn.BatchNorm2d(64))

        self.cnn.append(nn.Conv2d(in_channels=64,
                                        out_channels=128,
                                        kernel_size=3,
                                        stride=1,
                                        padding=0))


        self.maxpooling.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.bn.append(nn.BatchNorm2d(128))

        layers[0] = 128 # Number of nodes after convolution and maxpooling

        for input, output in zip(layers, layers[1:]):
            layer = nn.Linear(input, output)
            nn.init.kaiming_uniform_(layer.weight,
                                        nonlinearity="relu") # "He" method
            self.hidden.append(layer)


    def forward(self, x):

        l = len(self.maxpooling)
        for i, layer in zip(range(l), self.cnn):
            x = relu(layer(x))
            x = self.maxpooling[i](x)
            x = self.bn[i](x)

        x = x.view(x.size(0), -1)
        l = len(self.hidden)
        for i, layer in zip(range(l), self.hidden):
            if i < l - 1:
                x = self.dropout(x)
                x = relu(layer(x))
            else:
                x = nn.functional.log_softmax(layer(x), dim=1)

        return x

