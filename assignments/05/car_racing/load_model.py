import numpy as np
import torch as th



seq = th.nn.Sequential(
        th.nn.Linear(in_features=40, out_features=64, bias=True),
        th.nn.ReLU(),
        th.nn.Linear(in_features=64, out_features=64, bias=True),
        th.nn.ReLU(),
        th.nn.Linear(in_features=64, out_features=5, bias=True)
)

sd = seq.state_dict()

wb = np.load('named_params.npy', allow_pickle=True)

w0 = wb[0][1].data
b0 = wb[1][1].data
w1 = wb[2][1].data
b1 = wb[3][1].data
w2 = wb[4][1].data
b2 = wb[5][1].data

sd['0.weight'].copy_(w0)
sd['0.bias'].copy_(b0)
sd['2.weight'].copy_(w1)
sd['2.bias'].copy_(b1)
sd['4.weight'].copy_(w2)
sd['4.bias'].copy_(b2)

