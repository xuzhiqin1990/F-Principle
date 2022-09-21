import torch
import torch.nn as nn


def init_weights(model):
    pass

def ricker_actfun(xx):
    a = 0.1
    return 2/((3*a)*(3.1415926**0.25)) * (1 - (xx/a)**2) * K.exp(-0.5*(xx/a)**2) / 5


class RickerAct(nn.Module):
    def __init__(self, a=0.1) -> None:
        super().__init__()
        self.a = a
        
    def forward(self, x):
        return  2/((3*self.a)*(3.1415926**0.25)) * (1 - (x/self.a)**2) * torch.exp(-0.5*(x/self.a)**2) / 5

class SimpleLinear(nn.Module):
    def __init__(self, in_feat, out_feat, layers, act) -> None:
        super().__init__()
        if act in ['ReLU', 'Tanh']:
            act_fn = nn.__dict__[act]()
        elif act == 'Ricker':
            act_fn = RickerAct()
        else:
            raise ValueError(f"{act} not supported!")
        
        model = []
        model += [nn.Linear(in_feat, layers[0]), act_fn]
        for i in range(0, len(layers) - 1):
            model.append(nn.Linear(layers[i], layers[i+1]))
            model.append(act_fn)
        model.append(nn.Linear(layers[-1], out_feat))
        self.model = nn.ModuleList(model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    