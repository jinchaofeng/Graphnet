import torch.nn as nn


class CNN_unit(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(CNN_unit,self).__init__()
        self.cnn1=nn.Conv1d(dim_in, dim_out, kernel_size=3,stride=1,padding=1,dilation=1)
        self.act = nn.ReLU()
        if dim_in !=dim_out:
            self.res_layer = nn.Conv1d(dim_in, dim_out, 1)
        else:
            self.res_layer=None
    def forward(self,x):
        x1 = self.cnn1(x)
        x2 = self.act(x1)
        if self.res_layer is not None:
            residual=self.res_layer(x)
        else:
            residual=x
        out = residual+x2
        return out

class model_CNN(nn.Module):
    def __init__(self):
        super(model_CNN, self).__init__()
        self.CNN_unit1 = CNN_unit(1, 16)
        self.CNN_unit2 = CNN_unit(16, 32)
        self.CNN_unit3 = CNN_unit(32, 16)
        self.CNN_unit4 = CNN_unit(16, 3)

    def forward(self, x):
        # x=torch.unsqueeze(x,1)
        x1 = self.CNN_unit1(x)
        x2 = self.CNN_unit2(x1)
        x3 = self.CNN_unit3(x2)
        x4 = self.CNN_unit4(x3)
        return x4
