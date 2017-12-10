
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import itertools
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        hidden_size = 64 # determine this ...  from below
        self.aff_mu = nn.Linear(hidden_size, hidden_size)
        self.aff_log_std = nn.Linear(hidden_size, hidden_size)

        self.aff2_mu = nn.Linear(hidden_size, 1)
        self.aff2_log_std = nn.Linear(hidden_size, 1)

        self.act = nn.ReLU()
        self.final = nn.Tanh()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        print(out.size()) # you can determine hidden_size from here
        mu = self.final(self.aff2_mu(self.act(self.aff_mu(out))))
        log_std = self.final(self.aff2_log_std(self.act(self.aff_log_std(out))))

        std = torch.exp(log_std)

        return mu, log_std, std

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)