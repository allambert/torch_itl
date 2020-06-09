# junk file for testing
import math
import torch
n_i = 100
n_o = 10
n_d = 600000

x = torch.randn((n_i,n_o))
anchors = torch.randn(n_o, n_d)

a = torch.cos(x @ anchors)
b = torch.sin(x @ anchors)
b.shape
c = 1/torch.sqrt(torch.Tensor([n_d]))*torch.cat((a,b),1)
c.shape

torch.Tensor([60]).item()

x = torch.Tensor([0.3,0.3,0.3,0.3,0.3]).view(1,-1)
y = torch.Tensor([0.6,0.6,0.6,0.6,0.6]).view(1,-1)
gamma = 2
torch.exp(-gamma**2/2*x@x.T)

anchors = gamma*torch.randn(5, n_d)
a=1
b=2
c=1

c @ c.T

def feature_map(x, anchors):
    a = torch.cos(x @ anchors)
    b = torch.sin(x @ anchors)
    return 1/torch.sqrt(torch.Tensor([n_d]))*torch.cat((a,b),1)

feature_map(x,anchors) @ feature_map(y,anchors).T

if not a==b or not b==c:
    print(1)

model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(5, n_h),
    torch.nn.ReLU(),
    torch.nn.Linear(n_h, n_h),
    torch.nn.Linear(n_h, d_out),
)

from torch_itl import kernel
