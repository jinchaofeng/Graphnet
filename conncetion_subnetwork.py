import torch.nn as nn
import math


class connection_layer(nn.Module):
    def __init__(self, input, hidden, output):
        super(connection_layer, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.act(x1)
        x2 = self.fc2(x1)
        x2 = self.act(x2)
        return x2


class model_FCN(nn.Module):
    def __init__(self, input, output):
        super(model_FCN, self).__init__()
        hidden1 = math.floor(math.sqrt(input * (math.floor(output / 4) + 2)) + 1)
        self.C_unit1 = connection_layer(input, hidden1, math.floor(output / 4))
        hidden2 = math.floor(math.sqrt(math.floor(output / 4) * (math.floor(output / 2) + 2)) + 1)
        self.C_unit2 = connection_layer(math.floor(output / 4), hidden2, math.floor(output / 2))
        hidden3 = math.floor(math.sqrt(math.floor(output / 2) * (output + 2)) + 1)
        self.C_unit3 = connection_layer(math.floor(output / 2), hidden3, output)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.C_unit1(x)
        x1 = self.act(x1)
        x2 = self.C_unit2(x1)
        x2 = self.act(x2)
        x3 = self.C_unit3(x2)
        x3 = self.act(x3)
        return x3
