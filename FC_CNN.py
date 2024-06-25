from CNN_subnetwork import model_CNN
from connection_subnetwork import model_FCN
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
import torch

mesh = loadmat('../../')  # The path of the mesh
mesh = mesh['xx']  # The name of the mesh
nodes = mesh[0, 0]['nodes']
nodes = torch.FloatTensor(nodes)
num_nodes, _ = np.shape(np.array(nodes))
num_signal = 240  # Number of measurements for each source-detector pair


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.preprocess = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.connection = model_FCN(num_signal, num_nodes)
        self.cnn = model_CNN()

    def forward(self, A, x):
        x0 = self.preprocess(x)
        x1 = x0.squeeze(1)
        x2 = x1.reshape(x1.shape[0], num_signal)
        x3 = self.connection(x2)
        x4 = x3.unsqueeze(2)
        x5 = self.gcn(A, x4)
        x5 = x5.permute(0, 2, 1)
        return x5
    def forward(self,x):
        x0=self.preprocess(x)
        x1=x0.squeeze(1)
        x2=x1.reshape(x1.shape[0],num_signal)
        x3 = self.connection(x2)
        x4=x3.unsqueeze(1)
        x5=self.cnn(x4)
        return x5