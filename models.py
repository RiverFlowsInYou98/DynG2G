import torch
import numpy as np


class InputLinear(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_dim = input_size
        self.output_dim = input_size
        self.out_channels = input_size
        self.in_channels = input_size
        self.weight = torch.nn.Parameter(
            torch.eye(self.input_dim, self.output_dim), requires_grad=True
        )
        self.bias = torch.nn.Parameter(torch.zeros(self.output_dim), requires_grad=True)
        # self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        y = torch.sparse.mm(input, self.weight) + self.bias
        return y


class Linear(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = torch.nn.Parameter(torch.randn(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if input.is_sparse:
            y = torch.sparse.mm(input, self.weight) + self.bias
        else:
            y = torch.mm(input, self.weight) + self.bias
        return y


def wider(m1, m2, new_width):
    w1 = m1.weight.data.transpose(1, 0)
    w2 = m2.weight.data.transpose(1, 0)
    b1 = m1.bias.data
    assert w1.size(0) == w2.size(1), "Module weights are not compatible"
    assert new_width > w1.size(0), "New size should be larger"
    old_width = w1.size(0)
    nw1 = m1.weight.data.transpose(1, 0).clone()
    nw2 = w2.clone()
    nw1.resize_(new_width, nw1.size(1))
    nw2.resize_(nw2.size(0), new_width)
    w2 = w2.transpose(0, 1)
    nw2 = nw2.transpose(0, 1)
    if b1 is not None:
        nb1 = m1.bias.data.clone()
        nb1.resize_(new_width)
    nw1.narrow(0, 0, old_width).copy_(w1)
    nw2.narrow(0, 0, old_width).copy_(w2)
    nb1.narrow(0, 0, old_width).copy_(b1)
    weight_norm = True
    if weight_norm:
        for i in range(old_width):
            norm = w1.select(0, i).norm()
            w1.select(0, i).div_(norm)
    tracking = dict()
    for i in range(old_width, new_width):
        idx = np.random.randint(0, old_width)
        try:
            tracking[idx].append(i)
        except:
            tracking[idx] = [idx]
            tracking[idx].append(i)
        nw1.select(0, i).copy_(w1.select(0, idx).clone())
        nw2.select(0, i).copy_(w2.select(0, idx).clone())
        nb1[i] = b1[idx]
    random_init = False
    if not random_init:
        for idx, d in tracking.items():
            for item in d:
                nw2[item].div_(len(d))
    w2.transpose_(0, 1)
    nw2.transpose_(0, 1)
    m1.out_channels = new_width
    m2.in_channels = new_width
    noise = True
    if noise:
        noise = np.random.normal(scale=5e-2 * nw1.std(), size=list(nw1.size()))
        nw1 += torch.FloatTensor(noise).type_as(nw1)

    m1.weight.data = nw1.transpose(1, 0)
    m2.weight.data = nw2.transpose(1, 0)
    m1.bias.data = nb1

    return m1, m2


def Energy_KL(mu, sigma, pairs, L):
    ij_mu = mu[pairs]
    ij_sigma = sigma[pairs]
    sigma_ratio = ij_sigma[:, 1] / ij_sigma[:, 0]
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), 1)
    mu_diff_sq = torch.sum(torch.square(ij_mu[:, 0] - ij_mu[:, 1]) / ij_sigma[:, 0], 1)
    return 0.5 * (trace_fac + mu_diff_sq - L - log_det)


def build_loss(triplets, scale_terms, mu, sigma, L, scale):
    hop_pos = torch.stack(
        [torch.Tensor(triplets[:, 0]), torch.Tensor(triplets[:, 1])], 1
    ).type(torch.int64)
    hop_neg = torch.stack(
        [torch.Tensor(triplets[:, 0]), torch.Tensor(triplets[:, 2])], 1
    ).type(torch.int64)
    eng_pos = Energy_KL(mu, sigma, hop_pos, L)
    eng_neg = Energy_KL(mu, sigma, hop_neg, L)
    energy = torch.square(eng_pos) + torch.exp(-eng_neg)
    if scale:
        loss = torch.mean(energy * torch.Tensor(scale_terms).cuda())
    else:
        loss = torch.mean(energy)
    return loss


class Graph2Gauss(torch.nn.Module):
    def __init__(self, n_hidden, L, D):
        super(Graph2Gauss, self).__init__()

        self.D = D
        self.n_hidden = n_hidden
        self.sizes = [self.D] + self.n_hidden
        self.L = L
        layer = []
        self.mu_linear = Linear(self.sizes[-1], self.L)
        self.sigma_linear = Linear(self.sizes[-1], self.L)
        for i in range(1, len(self.sizes)):
            temp = Linear(self.sizes[i - 1], self.sizes[i])
            layer.append(temp)

        self.layers = torch.nn.ModuleList(layer)
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()

    def forward(self, input):
        encoded = input
        for i in range(0, len(self.sizes) - 1):
            encoded = self.relu(self.layers[i](encoded))
        mu = self.mu_linear(encoded)
        sigma = self.sigma_linear(encoded)
        sigma = self.elu(sigma) + 1 + 1e-14

        return encoded, mu, sigma

    def reset_parameters(self):
        self.mu_linear.reset_parameters()
        self.sigma_linear.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


class Classifier(torch.nn.Module):
    def __init__(self, L):
        super(Classifier, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * L, out_features=L),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=L, out_features=1),
        )

    def forward(self, x):
        return self.mlp(x)
