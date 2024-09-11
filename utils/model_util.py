from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTKConv2d(nn.Module):
    """Conv2d layer under NTK parametrization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        zero_init=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.bias = None
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.init(zero_init)

    def init(self, zero_init=False):
        if zero_init:
            nn.init.constant_(self.weight, 0.0)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0.0)
        else:
            nn.init.normal_(self.weight, 0.0, 1.0)
            if self.bias is not None:
                nn.init.normal_(self.bias, 0.0, 1.0)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def thaw(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x, add_bias=True):
        # weight = np.sqrt(1.0 / self.out_channels) * self.weight
        if add_bias and self.bias is not None:
            # bias = np.sqrt(0.1) * self.bias
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        else:
            return F.conv2d(x, self.weight, None, self.stride, self.padding)


class NTKLinear(nn.Module):
    """Linear layer under NTK parametrization."""

    def __init__(self, in_features, out_features, bias=True, zero_init=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bias = None
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.init(zero_init)

    def init(self, zero_init=False):
        if zero_init:
            nn.init.constant_(self.weight, 0.0)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0.0)
        else:
            nn.init.normal_(self.weight, 0.0, 1.0)
            if self.bias is not None:
                nn.init.normal_(self.bias, 0.0, 1.0)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def thaw(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x, add_bias=True):
        # weight = np.sqrt(1. / self.out_features) * self.weight
        if add_bias and self.bias is not None:
            # bias = np.sqrt(.1) * self.bias
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, None)


def merge_batchnorm(conv2d, batchnorm):
    """Folds BatchNorm2d into Conv2d."""
    if isinstance(batchnorm, nn.Identity):
        return conv2d
    mean = batchnorm.running_mean
    sigma = torch.sqrt(batchnorm.running_var + batchnorm.eps)
    beta = batchnorm.weight
    gamma = batchnorm.bias

    w = conv2d.weight
    if conv2d.bias is not None:
        b = conv2d.bias
    else:
        b = torch.zeros_like(mean)

    w = w * (beta / sigma).view(conv2d.out_channels, 1, 1, 1)
    b = (b - mean) / sigma * beta + gamma

    fused_conv2d = nn.Conv2d(
        conv2d.in_channels,
        conv2d.out_channels,
        conv2d.kernel_size,
        conv2d.stride,
        conv2d.padding,
    )
    fused_conv2d.weight.data = w
    fused_conv2d.bias.data = b

    return fused_conv2d


def leaky_relu_grad(y, LEAK):
    return (y > 0).float() + (y < 0).float() * LEAK


def softmax_grad(y):
    output = []
    for i in range(y.shape[0]):
        output.append(torch.outer(y[i], -y[i]) + torch.diag(y[i]))
    return torch.stack(output)


def maxpool_grad(y, kernel_size, indices):
    unpool = nn.MaxUnpool2d(kernel_size)
    output = unpool(y, indices)
    output[output != 0] = 1
    return output


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(
        indices
    )
    return output

def load_pretrained_model(args, model):
    """
    Loads the state dictionary from a checkpoint into a model.
    This function will also print out which layers are loaded and which are not.
    
    Args:
    - model (nn.Module): The model to load the state dictionary into.
    - checkpoint (dict): The loaded checkpoint containing the state dictionary.
    - strict (bool): Whether to strictly enforce that the keys in `checkpoint` and `model` match.
    
    Returns:
    - None
    """

    assert(args.imgnet_path != "")
    pretrained_model_ckpt = torch.load(args.imgnet_path)
    print("The best acc for this ckpt on imagenet is {}".format(pretrained_model_ckpt["best_acc1"]))
    model_state_dict = pretrained_model_ckpt['state_dict']

    new_state_dict = OrderedDict()
    for name, para in model_state_dict.items():
        if "module" in name:
            name = name.replace("module.", "", 1)
        if "last" not in name:
            new_state_dict[name] = para

    # Load the state dict with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # Print missing and unexpected layers
    if missing_keys:
        print("\nMissing layers in the checkpoint:")
        for k in missing_keys:
            print(f" - {k}")

    if unexpected_keys:
        print("\nUnexpected layers in the checkpoint:")
        for k in unexpected_keys:
            print(f" - {k}")


