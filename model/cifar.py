import copy
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import *
import torchvision

LEAK = 0.01

class NTKNet(nn.Module):
    """
    One forward pass to find the JVP
    Adpated from https://github.com/fmu2/gradfeat20/blob/master/src/model.py
    """

    def __init__(self):
        super(NTKNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.fc1 = nn.Linear(6272, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc_last = nn.Linear(512, 10)
        self.jvpconv1 = self.jvpconv2 = self.jvpconv3 = None
        self.jvpfc1 = self.jvpfc2 = self.jvpfc_last = None

    def jvp(self, model):
        # initialize the jvp layers
        self.jvpconv1 = NTKConv2d(3, 64, 5, 1, 2)
        self.jvpconv2 = NTKConv2d(64, 64, 5, 1, 2)
        self.jvpconv3 = NTKConv2d(64, 128, 5, 1, 2)
        self.jvpfc1 = NTKLinear(6272, 2048)
        self.jvpfc2 = NTKLinear(2048, 512)
        self.jvpfc_last = NTKLinear(512, 10)

        # set the parameter value of jvp layers to be the same as model
        self.jvpconv1.weight.data = copy.deepcopy(model.conv1.weight.data)
        self.jvpconv2.weight.data = copy.deepcopy(model.conv2.weight.data)
        self.jvpconv3.weight.data = copy.deepcopy(model.conv3.weight.data)
        self.jvpfc1.weight.data = copy.deepcopy(model.fc1.weight.data)
        self.jvpfc2.weight.data = copy.deepcopy(model.fc2.weight.data)
        self.jvpfc_last.weight.data = copy.deepcopy(model.fc_last.weight.data)

        self.jvpconv1.bias.data = copy.deepcopy(model.conv1.bias.data)
        self.jvpconv2.bias.data = copy.deepcopy(model.conv2.bias.data)
        self.jvpconv3.bias.data = copy.deepcopy(model.conv3.bias.data)
        self.jvpfc1.bias.data = copy.deepcopy(model.fc1.bias.data)
        self.jvpfc2.bias.data = copy.deepcopy(model.fc2.bias.data)
        self.jvpfc_last.bias.data = copy.deepcopy(model.fc_last.bias.data)

    def freeze_jvp(self):
        self.jvpconv1.freeze()
        self.jvpconv2.freeze()
        self.jvpconv3.freeze()
        self.jvpfc1.freeze()
        self.jvpfc2.freeze()
        self.jvpfc_last.freeze()

    def forward(self, x):
        y = F.leaky_relu(F.max_pool2d(self.conv1(x), 2), LEAK, inplace=True)
        y = F.leaky_relu(F.max_pool2d(self.conv2(y), 2), LEAK, inplace=True)
        y = F.leaky_relu(self.conv3(y), LEAK, inplace=True)
        y = y.view(y.shape[0], -1)
        y = F.leaky_relu(self.fc1(y), LEAK, inplace=True)
        y = F.leaky_relu(self.fc2(y), LEAK, inplace=True)
        output = self.fc_last(y)

        if self.jvpconv1 is not None:
            # one layer
            y, idx = F.max_pool2d_with_indices(self.jvpconv1(x), 2)
            jvp = maxpool_grad(y, 2, idx) * self.conv1(x)
            jvp = retrieve_elements_from_indices(jvp, idx) * leaky_relu_grad(y, LEAK)
            jvp = self.jvpconv2(jvp, add_bias=False) + self.conv2(
                F.leaky_relu(y, LEAK, inplace=True)
            )
            # two layers
            y, idx = F.max_pool2d_with_indices(self.jvpconv2(F.leaky_relu(y)), 2)
            jvp = retrieve_elements_from_indices(jvp, idx) * leaky_relu_grad(y, LEAK)
            # three layers
            jvp = self.jvpconv3(jvp, add_bias=False) + self.conv3(
                F.leaky_relu(y, LEAK, inplace=True)
            )
            y = self.jvpconv3(F.leaky_relu(y))
            jvp *= leaky_relu_grad(y, LEAK)
            # four layers
            y = y.view(y.shape[0], -1)
            jvp = jvp.view(jvp.shape[0], -1)
            jvp = self.jvpfc1(jvp, add_bias=False) + self.fc1(
                F.leaky_relu(y, LEAK, inplace=True)
            )
            y = self.jvpfc1(F.leaky_relu(y))
            jvp *= leaky_relu_grad(y, LEAK)
            # five layers
            jvp = self.jvpfc2(jvp, add_bias=False) + self.fc2(
                F.leaky_relu(y, LEAK, inplace=True)
            )
            y = self.jvpfc2(F.leaky_relu(y))
            jvp *= leaky_relu_grad(y, LEAK)
            # six layers
            print(jvp.shape, y.shape)
            jvp = self.jvpfc_last(jvp, add_bias=False) + self.fc_last(
                F.leaky_relu(y, LEAK, inplace=True)
            )

        else:
            jvp = None

        return output, jvp

class NTKCIFAR(nn.Module):
    """
    One forward pass to find the JVP
    Adpated from https://github.com/fmu2/gradfeat20/blob/master/src/model.py
    """

    def __init__(self, num_classes=10, init_weights=True):
        super(NTKCIFAR, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        # self.conv3 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(576, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc_last = nn.Linear(512, self.num_classes)
        self.jvpconv1 = self.jvpconv2 = self.jvpconv3 = None
        self.jvpfc1 = self.jvpfc2 = self.jvpfc_last = None

        # Initialize weights
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.normal_(m.weight, mean=0, std=self.w_sig/(m.in_channels*np.prod(m.kernel_size)))
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                    if m.bias is not None:
                        nn.init.normal_(m.bias, mean=0, std=0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def jvp(self, model):
        # initialize the jvp layers
        self.jvpconv1 = NTKConv2d(3, 6, kernel_size=3)
        self.jvpconv2 = NTKConv2d(6, 16, kernel_size=3)
        # self.jvpconv3 = NTKConv2d(10, 16, 5)
        self.jvpfc1 = NTKLinear(576, 2048)
        self.jvpfc2 = NTKLinear(2048, 1024)
        self.jvpfc3 = NTKLinear(1024, 512)
        self.jvpfc4 = NTKLinear(512, 256)
        self.jvpfc_last = NTKLinear(256, self.num_classes)

        # set the parameter value of jvp layers to be the same as model
        self.jvpconv1.weight.data = copy.deepcopy(model.conv1.weight.data)
        self.jvpconv2.weight.data = copy.deepcopy(model.conv2.weight.data)
        # self.jvpconv3.weight.data = copy.deepcopy(model.conv3.weight.data)
        self.jvpfc1.weight.data = copy.deepcopy(model.fc1.weight.data)
        self.jvpfc2.weight.data = copy.deepcopy(model.fc2.weight.data)
        self.jvpfc3.weight.data = copy.deepcopy(model.fc3.weight.data)
        self.jvpfc4.weight.data = copy.deepcopy(model.fc4.weight.data)
        self.jvpfc_last.weight.data = copy.deepcopy(model.fc_last.weight.data)

        self.jvpconv1.bias.data = copy.deepcopy(model.conv1.bias.data)
        self.jvpconv2.bias.data = copy.deepcopy(model.conv2.bias.data)
        # self.jvpconv3.bias.data = copy.deepcopy(model.conv3.bias.data)
        self.jvpfc1.bias.data = copy.deepcopy(model.fc1.bias.data)
        self.jvpfc2.bias.data = copy.deepcopy(model.fc2.bias.data)
        self.jvpfc3.bias.data = copy.deepcopy(model.fc3.bias.data)
        self.jvpfc4.bias.data = copy.deepcopy(model.fc4.bias.data)
        self.jvpfc_last.bias.data = copy.deepcopy(model.fc_last.bias.data)

    def freeze_jvp(self):
        self.jvpconv1.freeze()
        self.jvpconv2.freeze()
        # self.jvpconv3.freeze()
        self.jvpfc1.freeze()
        self.jvpfc2.freeze()
        self.jvpfc3.freeze()
        self.jvpfc4.freeze()
        self.jvpfc_last.freeze()

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        y = F.leaky_relu(F.max_pool2d(self.conv1(x), 2), LEAK, inplace=True)
        y = F.leaky_relu(F.max_pool2d(self.conv2(y), 2), LEAK, inplace=True)
        # y = F.leaky_relu(self.conv3(y), LEAK, inplace=True)
        y = y.view(y.shape[0], -1)
        y = F.leaky_relu(self.fc1(y), LEAK, inplace=True)
        y = F.leaky_relu(self.fc2(y), LEAK, inplace=True)
        y = F.leaky_relu(self.fc3(y), LEAK, inplace=True)
        y = F.leaky_relu(self.fc4(y), LEAK, inplace=True)
        output = self.fc_last(y)

        if self.jvpconv1 is not None:
            # one layer
            y, idx = F.max_pool2d_with_indices(self.jvpconv1(x), 2)
            jvp = maxpool_grad(y, 2, idx) * self.conv1(x)
            jvp = retrieve_elements_from_indices(jvp, idx) * leaky_relu_grad(y, LEAK)

            # two layers
            jvp = self.jvpconv2(jvp, add_bias=False) + self.conv2(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y, idx = F.max_pool2d_with_indices(
                self.jvpconv2(F.leaky_relu(y, LEAK, inplace=False)), 2
            )
            jvp = maxpool_grad(y, 2, idx) * jvp
            jvp = retrieve_elements_from_indices(jvp, idx) * leaky_relu_grad(y, LEAK)
            # three layers
            # jvp = self.jvpconv3(jvp, add_bias=False) + self.conv3(
            #     F.leaky_relu(y, LEAK, inplace=False)
            # )
            # y = self.jvpconv3(F.leaky_relu(y, LEAK, inplace=False))
            # jvp *= leaky_relu_grad(y, LEAK)
            # four layers
            y = y.view(y.shape[0], -1)
            jvp = jvp.view(jvp.shape[0], -1)
            jvp = self.jvpfc1(jvp, add_bias=False) + self.fc1(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y = self.jvpfc1(F.leaky_relu(y, LEAK, inplace=False))
            jvp *= leaky_relu_grad(y, LEAK)
            # five layers
            jvp = self.jvpfc2(jvp, add_bias=False) + self.fc2(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y = self.jvpfc2(F.leaky_relu(y, LEAK, inplace=False))
            jvp *= leaky_relu_grad(y, LEAK)
            # five layers
            jvp = self.jvpfc3(jvp, add_bias=False) + self.fc3(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y = self.jvpfc3(F.leaky_relu(y, LEAK, inplace=False))
            jvp *= leaky_relu_grad(y, LEAK)
            # five layers
            jvp = self.jvpfc4(jvp, add_bias=False) + self.fc4(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y = self.jvpfc4(F.leaky_relu(y, LEAK, inplace=False))
            jvp *= leaky_relu_grad(y, LEAK)
            # six layers
            jvp = self.jvpfc_last(jvp, add_bias=False) + self.fc_last(
                F.leaky_relu(y, LEAK, inplace=False)
            )

        else:
            jvp = None

        return output, jvp

class NTKCIFAR(nn.Module):
    """
    One forward pass to find the JVP
    Adpated from https://github.com/fmu2/gradfeat20/blob/master/src/model.py
    """

    def __init__(self, num_classes=10):
        super(NTKCIFAR, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_last = nn.Linear(128, self.num_classes)
        self.jvpconv1 = self.jvpconv2 = self.jvpconv3 = None
        self.jvpfc1 = self.jvpfc2 = self.jvpfc_last = None

    def jvp(self, model):
        # initialize the jvp layers
        self.jvpconv1 = NTKConv2d(3, 6, 5)
        self.jvpconv2 = NTKConv2d(16, 32, 5)
        # self.jvpconv3 = NTKConv2d(10, 16, 5)
        self.jvpfc1 = NTKLinear(16*5*5*4, 4096)
        self.jvpfc2 = NTKLinear(4096, 1024)
        self.jvpfc3 = NTKLinear(1024, 512)
        self.jvpfc4 = NTKLinear(512, 256)
        self.jvpfc_last = NTKLinear(256, 8)

        # set the parameter value of jvp layers to be the same as model
        self.jvpconv1.weight.data = copy.deepcopy(model.conv1.weight.data)
        self.jvpconv2.weight.data = copy.deepcopy(model.conv2.weight.data)
        # self.jvpconv3.weight.data = copy.deepcopy(model.conv3.weight.data)
        self.jvpfc1.weight.data = copy.deepcopy(model.fc1.weight.data)
        self.jvpfc2.weight.data = copy.deepcopy(model.fc2.weight.data)
        self.jvpfc_last.weight.data = copy.deepcopy(model.fc_last.weight.data)

        self.jvpconv1.bias.data = copy.deepcopy(model.conv1.bias.data)
        self.jvpconv2.bias.data = copy.deepcopy(model.conv2.bias.data)
        # self.jvpconv3.bias.data = copy.deepcopy(model.conv3.bias.data)
        self.jvpfc1.bias.data = copy.deepcopy(model.fc1.bias.data)
        self.jvpfc2.bias.data = copy.deepcopy(model.fc2.bias.data)
        self.jvpfc_last.bias.data = copy.deepcopy(model.fc_last.bias.data)

    def freeze_jvp(self):
        self.jvpconv1.freeze()
        self.jvpconv2.freeze()
        # self.jvpconv3.freeze()
        self.jvpfc1.freeze()
        self.jvpfc2.freeze()
        self.jvpfc_last.freeze()

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        y = F.leaky_relu(F.max_pool2d(self.conv1(x), 2), LEAK, inplace=True)
        y = F.leaky_relu(F.max_pool2d(self.conv2(y), 2), LEAK, inplace=True)
        # y = F.leaky_relu(self.conv3(y), LEAK, inplace=True)
        y = y.view(y.shape[0], -1)
        y = F.leaky_relu(self.fc1(y), LEAK, inplace=True)
        y = F.leaky_relu(self.fc2(y), LEAK, inplace=True)
        output = self.fc_last(y)

        if self.jvpconv1 is not None:
            # one layer
            y, idx = F.max_pool2d_with_indices(self.jvpconv1(x), 2)
            jvp = maxpool_grad(y, 2, idx) * self.conv1(x)
            jvp = retrieve_elements_from_indices(jvp, idx) * leaky_relu_grad(y, LEAK)

            # two layers
            jvp = self.jvpconv2(jvp, add_bias=False) + self.conv2(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y, idx = F.max_pool2d_with_indices(
                self.jvpconv2(F.leaky_relu(y, LEAK, inplace=False)), 2
            )
            jvp = maxpool_grad(y, 2, idx) * jvp
            jvp = retrieve_elements_from_indices(jvp, idx) * leaky_relu_grad(y, LEAK)

            # four layers
            y = y.view(y.shape[0], -1)
            jvp = jvp.view(jvp.shape[0], -1)
            jvp = self.jvpfc1(jvp, add_bias=False) + self.fc1(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y = self.jvpfc1(F.leaky_relu(y, LEAK, inplace=False))
            jvp *= leaky_relu_grad(y, LEAK)
            # five layers
            jvp = self.jvpfc2(jvp, add_bias=False) + self.fc2(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y = self.jvpfc2(F.leaky_relu(y, LEAK, inplace=False))
            jvp *= leaky_relu_grad(y, LEAK)
            # six layers
            jvp = self.jvpfc_last(jvp, add_bias=False) + self.fc_last(
                F.leaky_relu(y, LEAK, inplace=False)
            )

        else:
            jvp = None

        return output, jvp
    

class NTKNetSmall(nn.Module):
    """
    This model is same as NTKCIFAR and is temperaily used for debugging.
    """

    def __init__(self, num_classes=10):
        super(NTKNetSmall, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_last = nn.Linear(128, self.num_classes)
        self.jvpconv1 = self.jvpconv2 = self.jvpconv3 = None
        self.jvpfc1 = self.jvpfc2 = self.jvpfc_last = None

    def jvp(self, model):
        self.jvpconv1 = NTKConv2d(3, 6, 5)
        self.jvpconv2 = NTKConv2d(16, 32, 5)
        self.jvpfc1 = NTKLinear(16*5*5, 256)
        self.jvpfc2 = NTKLinear(256, 128)
        self.jvpfc_last = NTKLinear(128, self.num_classes)

        self.jvpconv1.weight.data = copy.deepcopy(model.conv1.weight.data)
        self.jvpconv2.weight.data = copy.deepcopy(model.conv2.weight.data)
        self.jvpfc1.weight.data = copy.deepcopy(model.fc1.weight.data)
        self.jvpfc2.weight.data = copy.deepcopy(model.fc2.weight.data)
        self.jvpfc_last.weight.data = copy.deepcopy(model.fc_last.weight.data)

        self.jvpconv1.bias.data = copy.deepcopy(model.conv1.bias.data)
        self.jvpconv2.bias.data = copy.deepcopy(model.conv2.bias.data)
        self.jvpfc1.bias.data = copy.deepcopy(model.fc1.bias.data)
        self.jvpfc2.bias.data = copy.deepcopy(model.fc2.bias.data)
        self.jvpfc_last.bias.data = copy.deepcopy(model.fc_last.bias.data)

    def freeze_jvp(self):
        self.jvpconv1.freeze()
        self.jvpconv2.freeze()
        self.jvpfc1.freeze()
        self.jvpfc2.freeze()
        self.jvpfc_last.freeze()

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        y = F.leaky_relu(F.max_pool2d(self.conv1(x), 2), LEAK, inplace=True)
        y = F.leaky_relu(F.max_pool2d(self.conv2(y), 2), LEAK, inplace=True)
        y = y.view(y.shape[0], -1)
        y = F.leaky_relu(self.fc1(y), LEAK, inplace=True)
        y = F.leaky_relu(self.fc2(y), LEAK, inplace=True)
        output = self.fc_last(y)

        if self.jvpconv1 is not None:
            # one layer
            y, idx = F.max_pool2d_with_indices(self.jvpconv1(x), 2)
            jvp = maxpool_grad(y, 2, idx) * self.conv1(x)
            jvp = retrieve_elements_from_indices(jvp, idx) * leaky_relu_grad(y, LEAK)

            # two layers
            jvp = self.jvpconv2(jvp, add_bias=False) + self.conv2(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y, idx = F.max_pool2d_with_indices(
                self.jvpconv2(F.leaky_relu(y, LEAK, inplace=False)), 2
            )
            jvp = maxpool_grad(y, 2, idx) * jvp
            jvp = retrieve_elements_from_indices(jvp, idx) * leaky_relu_grad(y, LEAK)

            # four layers
            y = y.view(y.shape[0], -1)
            jvp = jvp.view(jvp.shape[0], -1)
            jvp = self.jvpfc1(jvp, add_bias=False) + self.fc1(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y = self.jvpfc1(F.leaky_relu(y, LEAK, inplace=False))
            jvp *= leaky_relu_grad(y, LEAK)
            # five layers
            jvp = self.jvpfc2(jvp, add_bias=False) + self.fc2(
                F.leaky_relu(y, LEAK, inplace=False)
            )
            y = self.jvpfc2(F.leaky_relu(y, LEAK, inplace=False))
            jvp *= leaky_relu_grad(y, LEAK)
            # six layers
            jvp = self.jvpfc_last(jvp, add_bias=False) + self.fc_last(
                F.leaky_relu(y, LEAK, inplace=False)
            )

        else:
            jvp = None

        return output, jvp