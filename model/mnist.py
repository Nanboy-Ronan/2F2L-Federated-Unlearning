import copy
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import *
import torchvision


LEAK = 0.01

class NTKMnist(nn.Module):
    """
    One forward pass to find the JVP
    Adpated from https://github.com/fmu2/gradfeat20/blob/master/src/model.py
    """

    def __init__(self, num_classes=10, init_weights=True, **kwargs):
        super(NTKMnist, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_classes)

        self.jvpfc1 = self.jvpfc2 = None

        if init_weights:
            self.apply(self.kaiming_init)

    def jvp(self, model):
        # initialize the jvp layers
        self.jvpfc1 = NTKLinear(784, 500)
        self.jvpfc2 = NTKLinear(500, 100)
        self.jvpfc3 = NTKLinear(100, self.num_classes)
        # set the parameter value of jvp layers to be the same as model
        self.jvpfc1.weight.data = copy.deepcopy(model.fc1.weight.data)
        self.jvpfc2.weight.data = copy.deepcopy(model.fc2.weight.data)
        self.jvpfc3.weight.data = copy.deepcopy(model.fc3.weight.data)

        self.jvpfc1.bias.data = copy.deepcopy(model.fc1.bias.data)
        self.jvpfc2.bias.data = copy.deepcopy(model.fc2.bias.data)
        self.jvpfc3.bias.data = copy.deepcopy(model.fc3.bias.data)
    
    def kaiming_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def freeze_jvp(self):
        self.jvpfc1.freeze()
        self.jvpfc2.freeze()
        self.jvpfc3.freeze()

    def forward(self, x):
        x = x.view(-1, 784)
        y = self.fc1(x)
        y = self.fc2(F.leaky_relu(y, LEAK, inplace=True))
        output = self.fc3(F.leaky_relu(y, LEAK, inplace=True))
        # print(y.shape, output.shape)
        # output = y

        if self.jvpfc1 is not None:
            y = self.jvpfc1(x)
            jvp = self.fc1(x) * leaky_relu_grad(y, LEAK)

            jvp = self.jvpfc2(jvp, add_bias=False) + self.fc2(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            # jvp = self.jvpfc2(F.leaky_relu(y1))
            
            y = self.jvpfc2(F.leaky_relu(y, LEAK, inplace=False))

            jvp *= leaky_relu_grad(y, LEAK)

            # five layers
            jvp = self.jvpfc3(jvp, add_bias=False) + self.fc3(
                F.leaky_relu(y, LEAK, inplace=False)
            )

            # y3 = self.jvpfc3(F.leaky_relu(y))
            # jvp *= leaky_relu_grad(y, LEAK)


        else:
            jvp = None

        return output, jvp


class NTKMnistS(nn.Module):
    """
    One forward pass to find the JVP
    Adpated from https://github.com/fmu2/gradfeat20/blob/master/src/model.py
    """

    def __init__(self, num_classes=10, init_weights=True):
        super(NTKMnistS, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_classes)

        self.jvpfc1 = self.jvpfc2 = None

        # Initialize weights
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def jvp(self, model):
        # initialize the jvp layers
        self.jvpfc1 = NTKLinear(784, 500)
        self.jvpfc2 = NTKLinear(500, 100)
        self.jvpfc3 = NTKLinear(100, self.num_classes)
        # set the parameter value of jvp layers to be the same as model
        self.jvpfc1.weight.data = copy.deepcopy(model.fc1.weight.data)
        self.jvpfc2.weight.data = copy.deepcopy(model.fc2.weight.data)
        self.jvpfc3.weight.data = copy.deepcopy(model.fc3.weight.data)

        self.jvpfc1.bias.data = copy.deepcopy(model.fc1.bias.data)
        self.jvpfc2.bias.data = copy.deepcopy(model.fc2.bias.data)
        self.jvpfc3.bias.data = copy.deepcopy(model.fc3.bias.data)

    def freeze_jvp(self):
        self.jvpfc1.freeze()
        self.jvpfc2.freeze()
        self.jvpfc3.freeze()

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(F.leaky_relu(y, LEAK, inplace=True))
        y = self.fc3(F.leaky_relu(y, LEAK, inplace=True))
        output = F.softmax(y, 1)

        if self.jvpfc1 is not None:
            y = self.jvpfc1(x)
            jvp = self.fc1(x) * leaky_relu_grad(y, LEAK)

            jvp = self.jvpfc2(jvp, add_bias=False) + self.fc2(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            # jvp = self.jvpfc2(F.leaky_relu(y1))
            
            y = self.jvpfc2(F.leaky_relu(y, LEAK, inplace=False))

            jvp *= leaky_relu_grad(y, LEAK)

            # five layers
            jvp = self.jvpfc3(jvp, add_bias=False) + self.fc3(
                F.leaky_relu(y, LEAK, inplace=False)
            )

            y = F.softmax(self.jvpfc3(
                    F.leaky_relu(y, LEAK, inplace=False)
                ), 1
            )

            jvp = (softmax_grad(y) * jvp[:, None, :]).sum(-1)


        else:
            jvp = None

        return output, jvp