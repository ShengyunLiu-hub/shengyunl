"""Cosine-prototype classifier head for CL-LoRA (reference: https://github.com/hshustc/CVPR19_Incremental_Learning)."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CosineLinearFeature(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinearFeature, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

    def forward_diagonal(self, input, cur_task, init_cls=10, inc=10, out_dim=768):
        """Pure-diagonal scoring: branch i's classes are scored only against
        branch i's own feature block; the concatenated logits implicitly decide
        the task id via cross-branch argmax."""
        for i in range(cur_task + 1):
            if i == 0:
                start_cls = 0
                end_cls = init_cls
            else:
                start_cls = init_cls + (i - 1) * inc
                end_cls = start_cls + inc
            input1 = F.normalize(input[:, i * out_dim:(i + 1) * out_dim], p=2, dim=1)
            weight1 = F.normalize(self.weight[start_cls:end_cls, i * out_dim:(i + 1) * out_dim], p=2, dim=1)

            out = F.linear(input1, weight1)
            if i == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all, out), dim=1)

        if self.to_reduce:
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all

        return {'logits': out_all}


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)
