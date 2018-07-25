import torch
import torch.nn as nn


def fuse_bn_sequential(block, is_cpu=False):
    """
    This function takes a sequential block and fuses the batch normalization with convolution

    :param model: nn.Sequential. Source resnet model
    :return: nn.Sequential. Converted block
    """
    if not isinstance(block, nn.Sequential):
        return block
    stack = []
    for m in block.children():
        if isinstance(m, nn.BatchNorm2d):
            if isinstance(stack[-1], nn.Conv2d):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1].state_dict()

                # BatchNorm params
                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']
                gamma = bn_st_dict['weight']

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float()
                    if not is_cpu:
                        beta = beta.cuda()

                # Conv params
                W = conv_st_dict['weight']
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float()
                    if not is_cpu:
                        bias = bias.cuda()

                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                W.mul_(A)
                bias.add_(b)

                stack[-1].weight.data.copy_(W)
                if stack[-1].bias is None:
                    stack[-1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1].bias.data.copy_(bias)

        else:
            stack.append(m)

    if len(stack) > 1:
        return nn.Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model, is_cpu=False):
    for module_name in model._modules:
        module = model._modules[module_name]
        model._modules[module_name] = fuse_bn_sequential(module, is_cpu=is_cpu)
        if module._modules:
            fuse_bn_recursively(module, is_cpu=is_cpu)

    return model
